/* 
The interfaces 
 Copyright (C) 2019 Max-Planck-Society   Author: Martin Reinecke 
*/

#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "cudagridder.h"

using namespace std;
using namespace gridder;
// using gridder::detail::idx_t;
namespace py = pybind11;

namespace
{

    auto None = py::none();

    template <typename T>
    using pyarr = py::array_t<T, 0>;

#if defined(__GNUC__)
#define LOC_ CodeLocation(__FILE__, __LINE__, __PRETTY_FUNCTION__)
#else
#define LOC_ CodeLocation(__FILE__, __LINE__)
#endif

#define myfail(...)                                        \
    do                                                     \
    {                                                      \
        std::ostringstream os;                             \
        streamDump__(os, LOC_, "\n", ##__VA_ARGS__, "\n"); \
        throw std::runtime_error(os.str());                \
    } while (0)

#define myassert(cond, ...)                               \
    do                                                    \
    {                                                     \
        if (cond)                                         \
            ;                                             \
        else                                              \
        {                                                 \
            myfail("Assertion failure\n", ##__VA_ARGS__); \
        }                                                 \
    } while (0)

    template <typename T>
    inline void streamDump__(std::ostream &os, const T &value)
    {
        os << value;
    }

    template <typename T, typename... Args>
    inline void streamDump__(std::ostream &os, const T &value,
                             const Args &...args)
    {
        os << value;
        streamDump__(os, args...);
    }

    // to be replaced with std::source_location once available
    class CodeLocation
    {
    private:
        const char *file, *func;
        int line;

    public:
        CodeLocation(const char *file_, int line_, const char *func_ = nullptr)
            : file(file_), func(func_), line(line_) {}

        ostream &print(ostream &os) const
        {
            os << "file: " << file << ", line: " << line;
            if (func)
                os << ", function: " << func;
            return os;
        }
    };

    inline std::ostream &operator<<(std::ostream &os, const CodeLocation &loc)
    {
        return loc.print(os);
    }

    template <typename T>
    bool isPytype(const py::array &arr)
    {
        auto t1 = arr.dtype();
        auto t2 = pybind11::dtype::of<T>();
        auto k1 = t1.kind();
        auto k2 = t2.kind();
        auto s1 = t1.itemsize();
        auto s2 = t2.itemsize();
        return (k1 == k2) && (s1 == s2);
    }
    template <typename T>
    pyarr<T> getPyarr(const py::array &arr, const string &name)
    {
        auto t1 = arr.dtype();
        auto t2 = pybind11::dtype::of<T>();
        auto k1 = t1.kind();
        auto k2 = t2.kind();
        auto s1 = t1.itemsize();
        auto s2 = t2.itemsize();
        myassert((k1 == k2) && (s1 == s2),
                 "type mismatch for array '", name, "': expected '", k2, s2,
                 "', but got '", k1, s1, "'.");
        return arr.cast<pyarr<T>>();
    }

    template <typename T>
    pyarr<T> makeArray(const vector<size_t> &shape)
    {
        return pyarr<T>(shape);
    }

    void checkArray(const py::array &arr, const string &aname,
                    const vector<size_t> &shape)
    {
        if (size_t(arr.ndim()) != shape.size())
        {
            cerr << "Array '" << aname << "' has " << arr.ndim() << " dimensions; "
                                                                    "expected "
                 << shape.size() << endl;
            throw runtime_error("bad dimensionality");
        }
        for (size_t i = 0; i < shape.size(); ++i)
            if ((shape[i] != 0) && (size_t(arr.shape(i)) != shape[i]))
            {
                cerr << "Dimension " << i << " of array '" << aname << "' has size "
                     << arr.shape(i) << "; expected " << shape[i] << endl;
                throw runtime_error("bad array size");
            }
    }

    template <typename T>
    pyarr<T> provideArray(const py::object &in,
                          const string &name, const vector<size_t> &shape)
    {
        if (in.is_none())
        {
            auto tmp_ = makeArray<T>(shape);
            size_t sz = size_t(tmp_.size());
            auto tmp = tmp_.mutable_data();
            for (size_t i = 0; i < sz; ++i)
                tmp[i] = T(0);
            return tmp_;
        }
        auto tmp_ = getPyarr<T>(in.cast<py::array>(), name);
        checkArray(tmp_, name, shape);
        return tmp_;
    }

    template <typename T>
    pyarr<T> providePotentialArray(const py::object &in,
                                   const string &name, const vector<size_t> &shape)
    {
        if (in.is_none())
            return makeArray<T>(vector<size_t>(shape.size(), 0));
        return getPyarr<T>(in.cast<py::array>(), name);
    }

    template <size_t ndim, typename T>
    mav<T, ndim> make_mav(pyarr<T> &in)
    {
        myassert(ndim == in.ndim(), "dimension mismatch");
        array<size_t, ndim> dims;
        array<ptrdiff_t, ndim> str;
        for (size_t i = 0; i < ndim; ++i)
        {
            dims[i] = in.shape(i);
            str[i] = in.strides(i) / sizeof(T);
            myassert(str[i] * ptrdiff_t(sizeof(T)) == in.strides(i), "weird strides");
        }
        return mav<T, ndim>(in.mutable_data(), dims, str);
    }
    template <size_t ndim, typename T>
    const_mav<T, ndim> make_const_mav(const pyarr<T> &in)
    {
        myassert(ndim == in.ndim(), "dimension mismatch");
        array<size_t, ndim> dims;
        array<ptrdiff_t, ndim> str;
        for (size_t i = 0; i < ndim; ++i)
        {
            dims[i] = in.shape(i);
            str[i] = in.strides(i) / sizeof(T);
            myassert(str[i] * ptrdiff_t(sizeof(T)) == in.strides(i), "weird strides");
        }
        return const_mav<T, ndim>(in.data(), dims, str);
    }

    template <typename T>
    py::array ms2dirty_general2(const py::array &uvw_,
                                const py::array &freq_, const py::array &ms_, const py::object &wgt_,
                                size_t npix_x, size_t npix_y, double pixsize_x, double pixsize_y, size_t nu, size_t nv, double epsilon,
                                bool do_wstacking, size_t nthreads, size_t verbosity)
    {
        auto uvw = getPyarr<double>(uvw_, "uvw");
        auto uvw2 = make_const_mav<2>(uvw);
        auto freq = getPyarr<double>(freq_, "freq");
        auto freq2 = make_const_mav<1>(freq);
        auto ms = getPyarr<complex<T>>(ms_, "ms");
        auto ms2 = make_const_mav<2>(ms);
        auto wgt = providePotentialArray<T>(wgt_, "wgt", {ms2.shape(0), ms2.shape(1)});
        auto wgt2 = make_const_mav<2>(wgt);
        auto dirty = makeArray<T>({npix_x, npix_y});
        auto dirty2 = make_mav<2>(dirty);
        {
            py::gil_scoped_release release;
            ms2dirty_general(uvw2, freq2, ms2, wgt2, pixsize_x, pixsize_y, nu, nv, epsilon, do_wstacking,
                             nthreads, dirty2, verbosity);
        }
        return dirty;
    }

    py::array Pyms2dirty_general(const py::array &uvw,
                                 const py::array &freq, const py::array &ms, const py::object &wgt,
                                 size_t npix_x, size_t npix_y, double pixsize_x, double pixsize_y, size_t nu, size_t nv, double epsilon,
                                 bool do_wstacking, size_t nthreads, size_t verbosity)
    {
        if (isPytype<complex<float>>(ms))
            return ms2dirty_general2<float>(uvw, freq, ms, wgt, npix_x, npix_y,
                                            pixsize_x, pixsize_y, nu, nv, epsilon, do_wstacking, nthreads, verbosity);
        if (isPytype<complex<double>>(ms))
            return ms2dirty_general2<double>(uvw, freq, ms, wgt, npix_x, npix_y,
                                             pixsize_x, pixsize_y, nu, nv, epsilon, do_wstacking, nthreads, verbosity);
        myfail("type matching failed: 'ms' has neither type 'c8' nor 'c16'");
    }
    constexpr auto ms2dirty_DS = R"""(
Converts an MS object to dirty image.

Parameters
==========
uvw: np.array((nrows, 3), dtype=np.float64)
    UVW coordinates from the measurement set
freq: np.array((nchan,), dtype=np.float64)
    channel frequencies
ms: np.array((nrows, nchan,), dtype=np.complex64 or np.complex128)
    the input measurement set data.
    Its data type determines the precision in which the calculation is carried
    out.
wgt: np.array((nrows, nchan), float with same precision as `ms`), optional
    If present, its values are multiplied to the output
npix_x, npix_y: int
    dimensions of the dirty image
pixsize_x, pixsize_y: float
    angular pixel size (in radians) of the dirty image
epsilon: float
    accuracy at which the computation should be done. Must be larger than 2e-13.
    If `ms` has type np.complex64, it must be larger than 1e-5.
do_wstacking: bool
    if True, the full improved w-stacking algorithm is carried out, otherwise
    the w values are assumed to be zero.
nthreads: int
    number of threads to use for the calculation
verbosity: int
    0: no output
    1: some output
    2: detailed output

Returns
=======
np.array((nxdirty, nydirty), dtype=float of same precision as `ms`)
    the dirty image
)""";
    py::array Pyms2dirty(const py::array &uvw,
                         const py::array &freq, const py::array &ms, const py::object &wgt,
                         size_t npix_x, size_t npix_y, double pixsize_x, double pixsize_y, double epsilon,
                         bool do_wstacking, size_t nthreads, size_t verbosity)
    {
        return Pyms2dirty_general(uvw, freq, ms, wgt, npix_x, npix_y,
                                  pixsize_x, pixsize_y, 2 * npix_x, 2 * npix_y, epsilon, do_wstacking, nthreads,
                                  verbosity);
    }

    template <typename T>
    py::array dirty2ms_general2(const py::array &uvw_,
                                const py::array &freq_, const py::array &dirty_, const py::object &wgt_,
                                double pixsize_x, double pixsize_y, size_t nu, size_t nv, double epsilon,
                                bool do_wstacking, size_t nthreads, size_t verbosity)
    {
        auto uvw = getPyarr<double>(uvw_, "uvw");
        auto uvw2 = make_const_mav<2>(uvw);
        auto freq = getPyarr<double>(freq_, "freq");
        auto freq2 = make_const_mav<1>(freq);
        auto dirty = getPyarr<T>(dirty_, "dirty");
        auto dirty2 = make_const_mav<2>(dirty);
        auto wgt = providePotentialArray<T>(wgt_, "wgt", {uvw2.shape(0), freq2.shape(0)});
        auto wgt2 = make_const_mav<2>(wgt);
        auto ms = makeArray<complex<T>>({uvw2.shape(0), freq2.shape(0)});
        auto ms2 = make_mav<2>(ms);
        {
            py::gil_scoped_release release;
            dirty2ms_general(uvw2, freq2, dirty2, wgt2, pixsize_x, pixsize_y, nu, nv, epsilon, do_wstacking,
                             nthreads, ms2, verbosity);
        }
        return ms;
    }
    py::array Pydirty2ms_general(const py::array &uvw,
                                 const py::array &freq, const py::array &dirty, const py::object &wgt,
                                 double pixsize_x, double pixsize_y, size_t nu, size_t nv, double epsilon,
                                 bool do_wstacking, size_t nthreads, size_t verbosity)
    {
        if (isPytype<float>(dirty))
            return dirty2ms_general2<float>(uvw, freq, dirty, wgt,
                                            pixsize_x, pixsize_y, nu, nv, epsilon, do_wstacking, nthreads, verbosity);
        if (isPytype<double>(dirty))
            return dirty2ms_general2<double>(uvw, freq, dirty, wgt,
                                             pixsize_x, pixsize_y, nu, nv, epsilon, do_wstacking, nthreads, verbosity);
        myfail("type matching failed: 'dirty' has neither type 'f4' nor 'f8'");
    }
    constexpr auto dirty2ms_DS = R"""(
Converts a dirty image to an MS object.

Parameters
==========
uvw: np.array((nrows, 3), dtype=np.float64)
    UVW coordinates from the measurement set
freq: np.array((nchan,), dtype=np.float64)
    channel frequencies
dirty: np.array((nxdirty, nydirty), dtype=np.float32 or np.float64)
    dirty image
    Its data type determines the precision in which the calculation is carried
    out.
wgt: np.array((nrows, nchan), same dtype as `dirty`), optional
    If present, its values are multiplied to the output
pixsize_x, pixsize_y: float
    angular pixel size (in radians) of the dirty image
epsilon: float
    accuracy at which the computation should be done. Must be larger than 2e-13.
    If `dirty` has type np.float32, it must be larger than 1e-5.
do_wstacking: bool
    if True, the full improved w-stacking algorithm is carried out, otherwise
    the w values are assumed to be zero.
nthreads: int
    number of threads to use for the calculation
verbosity: int
    0: no output
    1: some output
    2: detailed output

Returns
=======
np.array((nrows, nchan,), dtype=complex of same precision as `dirty`)
    the measurement set data.
)""";
    py::array Pydirty2ms(const py::array &uvw,
                         const py::array &freq, const py::array &dirty, const py::object &wgt,
                         double pixsize_x, double pixsize_y, double epsilon,
                         bool do_wstacking, size_t nthreads, size_t verbosity)
    {
        return Pydirty2ms_general(uvw, freq, dirty, wgt, pixsize_x, pixsize_y,
                                  2 * dirty.shape(0), 2 * dirty.shape(1), epsilon, do_wstacking, nthreads,
                                  verbosity);
    }

} // unnamed namespace

PYBIND11_MODULE(nifty_gridder, m)
{
    using namespace pybind11::literals;

    m.def("ms2dirty", &Pyms2dirty, ms2dirty_DS, "uvw"_a, "freq"_a, "ms"_a,
          "wgt"_a = None, "npix_x"_a, "npix_y"_a, "pixsize_x"_a, "pixsize_y"_a,
          "epsilon"_a, "do_wstacking"_a = false, "nthreads"_a = 1, "verbosity"_a = 0);
    m.def("ms2dirty_general", &Pyms2dirty_general, "uvw"_a, "freq"_a, "ms"_a,
          "wgt"_a = None, "npix_x"_a, "npix_y"_a, "pixsize_x"_a, "pixsize_y"_a, "nu"_a, "nv"_a,
          "epsilon"_a, "do_wstacking"_a = false, "nthreads"_a = 1, "verbosity"_a = 0);
    m.def("dirty2ms", &Pydirty2ms, dirty2ms_DS, "uvw"_a, "freq"_a, "dirty"_a,
          "wgt"_a = None, "pixsize_x"_a, "pixsize_y"_a, "epsilon"_a,
          "do_wstacking"_a = false, "nthreads"_a = 1, "verbosity"_a = 0);
    m.def("dirty2ms_general", &Pydirty2ms_general, "uvw"_a, "freq"_a, "dirty"_a,
          "wgt"_a = None, "pixsize_x"_a, "pixsize_y"_a, "nu"_a, "nv"_a, "epsilon"_a,
          "do_wstacking"_a = false, "nthreads"_a = 1, "verbosity"_a = 0);
}

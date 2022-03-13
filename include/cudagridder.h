/*
 *  This file is part of nifty_gridder.
 *
 *  nifty_gridder is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  nifty_gridder is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with nifty_gridder; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 */

/* 
 Copyright (C) 2019 Max-Planck-Society   Author: Martin Reinecke 
*/

#ifndef CUDA_GRIDDER_H
#define CUDA_GRIDDER_H

#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <array>

namespace gridder
{
    namespace detail
    {

        using namespace std;

        template <typename T>
        struct VLEN
        {
            static constexpr size_t val = 1;
        };

        // "mav" stands for "multidimensional array view"
        template <typename T, size_t ndim>
        class mav
        {
            static_assert((ndim > 0) && (ndim < 3), "only supports 1D and 2D arrays");

        private:
            T *d;
            array<size_t, ndim> shp;
            array<ptrdiff_t, ndim> str;

        public:
            mav(T *d_, const array<size_t, ndim> &shp_,
                const array<ptrdiff_t, ndim> &str_)
                : d(d_), shp(shp_), str(str_) {}
            mav(T *d_, const array<size_t, ndim> &shp_)
                : d(d_), shp(shp_)
            {
                str[ndim - 1] = 1;
                for (size_t d = 2; d <= ndim; ++d)
                    str[ndim - d] = str[ndim - d + 1] * shp[ndim - d + 1];
            }
            T &operator[](size_t i) const
            {
                return operator()(i);
            }
            T &operator()(size_t i) const
            {
                static_assert(ndim == 1, "ndim must be 1");
                return d[str[0] * i];
            }
            T &operator()(size_t i, size_t j) const
            {
                static_assert(ndim == 2, "ndim must be 2");
                return d[str[0] * i + str[1] * j];
            }
            size_t shape(size_t i) const { return shp[i]; }
            const array<size_t, ndim> &shape() const { return shp; }
            size_t size() const
            {
                size_t res = 1;
                for (auto v : shp)
                    res *= v;
                return res;
            }
            ptrdiff_t stride(size_t i) const { return str[i]; }
            T *data() const
            {
                return d;
            }
            bool last_contiguous() const
            {
                return (str[ndim - 1] == 1) || (str[ndim - 1] == 0);
            }
#ifdef GRIDDER_CHECKS
            void check_storage(const char *name) const
            {
                if (!last_contiguous())
                    cout << "Array '" << name << "': last dimension is not contiguous.\n"
                                                 "This may slow down computation significantly!\n";
            }
#else
            void check_storage(const char * /*name*/) const
            {
            }
#endif
            bool contiguous() const
            {
                ptrdiff_t stride = 1;
                for (size_t i = 0; i < ndim; ++i)
                {
                    if (str[ndim - 1 - i] != stride)
                        return false;
                    stride *= shp[ndim - 1 - i];
                }
                return true;
            }
            void fill(const T &val) const
            {
                // FIXME: special cases for contiguous arrays and/or zeroing?
                if (ndim == 1)
                    for (size_t i = 0; i < shp[0]; ++i)
                        d[str[0] * i] = val;
                else if (ndim == 2)
                    for (size_t i = 0; i < shp[0]; ++i)
                        for (size_t j = 0; j < shp[1]; ++j)
                            d[str[0] * i + str[1] * j] = val;
            }
        };

        template <typename T, size_t ndim>
        using const_mav = mav<const T, ndim>;
        template <typename T, size_t ndim>
        const_mav<T, ndim> cmav(const mav<T, ndim> &mav)
        {
            return const_mav<T, ndim>(mav.data(), mav.shape());
        }
        template <typename T, size_t ndim>
        const_mav<T, ndim> nullmav()
        {
            array<size_t, ndim> shp;
            shp.fill(0);
            return const_mav<T, ndim>(nullptr, shp);
        }

        // // ////  // // // // // // // //
        //                       const const_mav<double, 1> &freq, const const_mav<complex<T>, 2> &ms,
        //                       const const_mav<T, 2> &wgt, double pixsize_x, double pixsize_y, size_t nu, size_t nv, double epsilon,
        //                       bool do_wstacking, size_t nthreads, const mav<T, 2> &dirty, size_t verbosity,
        //                       bool negate_v = false)
        // {
        //     Baselines baselines(uvw, freq, negate_v);
        //     GridderConfig gconf(dirty.shape(0), dirty.shape(1), nu, nv, epsilon, pixsize_x, pixsize_y, nthreads);
        //     auto idx = getWgtIndices(baselines, gconf, wgt, ms);
        //     auto idx2 = const_mav<idx_t, 1>(idx.data(), {idx.size()});
        //     x2dirty(gconf, makeMsServ(baselines, idx2, ms, wgt), dirty, do_wstacking, verbosity);
        // }
        // // // // // //

        template <typename T>
        void ms2dirty_general(const const_mav<double, 2> &uvw,
                              const const_mav<double, 1> &freq, const const_mav<complex<T>, 2> &ms,
                              const const_mav<T, 2> &wgt, double pixsize_x, double pixsize_y, size_t nu, size_t nv, double epsilon,
                              bool do_wstacking, size_t nthreads, const mav<T, 2> &dirty, size_t verbosity,
                              bool negate_v = false)
        {
            // Baselines baselines(uvw, freq, negate_v);
            // GridderConfig gconf(dirty.shape(0), dirty.shape(1), nu, nv, epsilon, pixsize_x, pixsize_y, nthreads);
            // auto idx = getWgtIndices(baselines, gconf, wgt, ms);
            // auto idx2 = const_mav<idx_t, 1>(idx.data(), {idx.size()});
            // x2dirty(gconf, makeMsServ(baselines, idx2, ms, wgt), dirty, do_wstacking, verbosity);
        }

        template <typename T>
        void ms2dirty(const const_mav<double, 2> &uvw,
                      const const_mav<double, 1> &freq, const const_mav<complex<T>, 2> &ms,
                      const const_mav<T, 2> &wgt, double pixsize_x, double pixsize_y, double epsilon,
                      bool do_wstacking, size_t nthreads, const mav<T, 2> &dirty, size_t verbosity)
        {
            ms2dirty_general(uvw, freq, ms, wgt, pixsize_x, pixsize_y,
                             2 * dirty.shape(0), 2 * dirty.shape(1), epsilon, do_wstacking, nthreads,
                             dirty, verbosity);
        }

        template <typename T>
        void dirty2ms_general(const const_mav<double, 2> &uvw,
                              const const_mav<double, 1> &freq, const const_mav<T, 2> &dirty,
                              const const_mav<T, 2> &wgt, double pixsize_x, double pixsize_y, size_t nu, size_t nv, double epsilon,
                              bool do_wstacking, size_t nthreads, const mav<complex<T>, 2> &ms,
                              size_t verbosity, bool negate_v = false)
        {
            // Baselines baselines(uvw, freq, negate_v);
            // GridderConfig gconf(dirty.shape(0), dirty.shape(1), nu, nv, epsilon, pixsize_x, pixsize_y, nthreads);
            const_mav<complex<T>, 2> null_ms(nullptr, {0, 0});
            // auto idx = getWgtIndices(baselines, gconf, wgt, null_ms);
            // auto idx2 = const_mav<idx_t, 1>(idx.data(), {idx.size()});
            // ms.fill(0);
            //dirty2x(gconf, dirty, makeMsServ(baselines, idx2, ms, wgt), do_wstacking, verbosity);
        }

        template <typename T>
        void dirty2ms(const const_mav<double, 2> &uvw,
                      const const_mav<double, 1> &freq, const const_mav<T, 2> &dirty,
                      const const_mav<T, 2> &wgt, double pixsize_x, double pixsize_y, double epsilon,
                      bool do_wstacking, size_t nthreads, const mav<complex<T>, 2> &ms, size_t verbosity)
        {
            dirty2ms_general(uvw, freq, dirty, wgt, pixsize_x, pixsize_y,
                             2 * dirty.shape(0), 2 * dirty.shape(1), epsilon, do_wstacking, nthreads, ms,
                             verbosity);
        }

    } // namespace detail

    // public names
    // using detail::Baselines;
    using detail::const_mav;
    using detail::dirty2ms;
    using detail::mav;
    using detail::ms2dirty;
} // namespace gridder

#endif

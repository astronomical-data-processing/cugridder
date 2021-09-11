/*
Invoke conv related kernel (device) function
Functions:
  1. get_num_cells
  2. setup_conv_opts
  3. conv_*d_invoker
  4. curafft_conv
  5. curafft_partial_conv
Issue: Revise for batch
*/

#include <math.h>
#include <stdlib.h>
#include <cuda.h>
#include <stdio.h>
#include <helper_cuda.h>
#include <iostream>
#include <iomanip>
#include <assert.h>
#include <cuComplex.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include "conv_interp_invoker.h"
#include "conv.h"
#include "interp.h"
#include "utils.h"

int get_num_cells(int ms, conv_opts copts)
// type 1 & 2 recipe for how to set 1d size of upsampled array, nf, given opts
// and requested number of Fourier modes ms.
{
  /*
    Determain the size of the grid
    ms - number of fourier modes (image size)
    copt - contains convolution related parameters
  */
  int nf = (int)(copts.upsampfac*ms);
  if (nf<2*copts.kw) nf=2*copts.kw; // otherwise spread fails
  if (nf<1e11){                                // otherwise will fail anyway
      nf = next235beven(nf, 1);
  }
  return nf;
}

int setup_conv_opts(conv_opts &opts, PCS eps, PCS upsampfac, int pirange, int direction, int kerevalmeth)
{
  /*
    setup conv related components
    follow the setting in  Yu-hsuan Shih (https://github.com/flatironinstitute/cufinufft)
  */
  // handling errors or warnings
  if (upsampfac != 2.0)
  { // nonstandard sigma
    if (kerevalmeth == 1)
    {
      fprintf(stderr, "setup_conv_opts: nonstandard upsampling factor %.3g with kerevalmeth=1\n", (double)upsampfac);
      return 2;
    }
    if (upsampfac <= 1.0)
    {
      fprintf(stderr, "setup_conv_opts: error, upsampling factor %.3g too small\n", (double)upsampfac);
      return 2;
    }
    // calling routine must abort on above errors, since opts is garbage!
    if (upsampfac > 4.0)
      fprintf(stderr, "setup_conv_opts: warning, upsampfac=%.3g is too large\n", (double)upsampfac);
  }

  opts.direction = direction; 
  opts.pirange = pirange; // in range [-pi,pi) or [0,N)
  opts.upsampfac = upsampfac; // upsamling factor

  
  int ier = 0;
  if (eps < EPSILON)
  {
    fprintf(stderr, "setup_conv_opts: warning, eps (tol) is too small, set eps = %.3g.\n", (double)EPSILON);
    eps = EPSILON;
    ier = 1;
  }

  // kernel width  (kw) and ES kernel beta parameter setting
  int kw = std::ceil(-log10(eps / (PCS)10.0));                  // 1 digit per power of ten
  if (upsampfac != 2.0)                                         // override ns for custom sigma
    kw = std::ceil(-log(eps) / (PI * sqrt(1.0 - 1.0 / upsampfac))); // formula, gamma=1
  kw = max(2, kw);                                              
  if (kw > MAX_KERNEL_WIDTH)
  { // clip to match allocated arrays
    fprintf(stderr, "%s warning: at upsampfac=%.3g, tol=%.3g would need kernel width ns=%d; clipping to max %d, better to revise sigma and tol.\n", __func__,
            upsampfac, (double)eps, kw, MAX_KERNEL_WIDTH);
    kw = MAX_KERNEL_WIDTH;
    printf("warning: at upsampfac=%.3g, tol=%.3g would need kernel width ns=%d; clipping to max %d, better to revise sigma and tol.\n",
            upsampfac, (double)eps, kw, MAX_KERNEL_WIDTH);
  }
  opts.kw = kw;
  opts.ES_halfwidth = (PCS)kw / 2; // constants to help ker eval (except Horner)
  opts.ES_c = 4.0 / (PCS)(kw * kw);

  PCS betaoverns = 2.30; // gives decent betas for default sigma=2.0
  if (kw == 2)
    betaoverns = 2.20; // some small-width tweaks...
  if (kw == 3)
    betaoverns = 2.26;
  if (kw == 4)
    betaoverns = 2.38;
  if (upsampfac != 2.0)
  {                                                      // again, override beta for custom sigma
    PCS gamma = 0.97;                                    // must match devel/gen_all_horner_C_code.m
    betaoverns = gamma * PI * (1 - 1 / (2 * upsampfac)); // formula based on cutoff
  }
  opts.ES_beta = betaoverns * (PCS)kw; // set the kernel beta parameter
  // printf("the value of beta %.3f\n",opts.ES_beta);
  //fprintf(stderr,"setup_spreader: sigma=%.6f, chose ns=%d beta=%.6f\n",(double)upsampfac,ns,(double)opts.ES_beta); // user hasn't set debug yet
  return ier;
}


int conv_1d_invoker(int nf1, int M, curafft_plan *plan){
  /*
    convolution invoker, invoke the kernel function
    nf1 - grid size in 1D
    M - number of points
  */
  dim3 grid;
  dim3 block;
  if (plan->opts.gpu_gridder_method == 0)
  {
    block.x = 256; // 256 threads per block
    grid.x = (M - 1) / block.x + 1; // number of blocks

    // if the image resolution is small, the memory is sufficiently large for output after conv. 
    conv_1d_nputsdriven<<<grid, block>>>(plan->d_u, plan->d_c, plan->fw, plan->M,
                                          plan->copts.kw, nf1, plan->copts.ES_c, plan->copts.ES_beta, 
                                          plan->copts.pirange, plan->cell_loc);

    checkCudaErrors(cudaDeviceSynchronize());
  }
  return 0;
}

int conv_2d_invoker(int nf1, int nf2, int M, curafft_plan *plan)
{

  dim3 grid;
  dim3 block;
  
  if (plan->opts.gpu_gridder_method == 0)
  {
    block.x = 256;
    grid.x = (M - 1) / block.x + 1;

    // if the image resolution is small, the memory is sufficiently large for output after conv. 
    conv_2d_nputsdriven<<<grid, block>>>(plan->d_u, plan->d_v, plan->d_c, plan->fw, plan->M,
                                          plan->copts.kw, nf1, nf2, plan->copts.ES_c, plan->copts.ES_beta, 
                                          plan->copts.pirange, plan->cell_loc);
    

    checkCudaErrors(cudaDeviceSynchronize());
  }

  return 0;
}

int conv_3d_invoker(int nf1, int nf2, int nf3, int M, curafft_plan *plan)
{

  dim3 grid;
  dim3 block;
  
  if (plan->opts.gpu_gridder_method == 0)
  {
    block.x = 256;
    grid.x = (M - 1) / block.x + 1;
    // if the image resolution is small, the memory is sufficiently large for output after conv. 
    conv_3d_nputsdriven<<<grid, block>>>(plan->d_u, plan->d_v, plan->d_w, plan->d_c, plan->fw, plan->M,
                                          plan->copts.kw, nf1, nf2, nf3, plan->copts.ES_c, plan->copts.ES_beta,
                                          plan->copts.pirange, plan->cell_loc);
    

    checkCudaErrors(cudaDeviceSynchronize());
  }

  return 0;
}

int curafft_conv(curafft_plan * plan)
{
  /*
  ---- convolution opertion ----
  */

  int ier = 0;
  int nf1 = plan->nf1;
  int nf2 = plan->nf2;
  int nf3 = plan->nf3;
  int M = plan->M;
  // printf("w_term_method %d\n",plan->w_term_method);
  switch (plan->dim)
  {
  case 1:
    conv_1d_invoker(nf1, M, plan);
    break;
  case 2:
    conv_2d_invoker(nf1, nf2, M, plan);
    break;
  case 3:
    conv_3d_invoker(nf1, nf2, nf3, M, plan);
    break;
  default:
    ier = 1; // error
    break;
  }

  return ier;
}


int interp_1d_invoker(int nf1, int M, curafft_plan *plan){
  /*
    convolution invoker, invoke the kernel function
    nf1 - grid size in 1D
    M - number of points
  */
  dim3 grid;
  dim3 block;
  if (plan->opts.gpu_gridder_method == 0)
  {
    block.x = 256; // 256 threads per block
    grid.x = (M - 1) / block.x + 1; // number of blocks

    // if the image resolution is small, the memory is sufficiently large for output after conv. 
    interp_1d_nputsdriven<<<grid, block>>>(plan->d_u, plan->d_c, plan->fw, plan->M,
                                          plan->copts.kw, nf1, plan->copts.ES_c, plan->copts.ES_beta, 
                                          plan->copts.pirange, plan->cell_loc);

    checkCudaErrors(cudaDeviceSynchronize());
  }
  return 0;
}

int interp_2d_invoker(int nf1, int nf2, int M, curafft_plan *plan)
{

  dim3 grid;
  dim3 block;
  
  if (plan->opts.gpu_gridder_method == 0)
  {
    block.x = 256;
    grid.x = (M - 1) / block.x + 1;

    // if the image resolution is small, the memory is sufficiently large for output after conv. 
    interp_2d_nputsdriven<<<grid, block>>>(plan->d_u, plan->d_v, plan->d_c, plan->fw, plan->M,
                                          plan->copts.kw, nf1, nf2, plan->copts.ES_c, plan->copts.ES_beta, 
                                          plan->copts.pirange, plan->cell_loc);
    

    checkCudaErrors(cudaDeviceSynchronize());
  }

  return 0;
}

int interp_3d_invoker(int nf1, int nf2, int nf3, int M, curafft_plan *plan)
{

  dim3 grid;
  dim3 block;
  
  if (plan->opts.gpu_gridder_method == 0)
  {
    block.x = 256;
    grid.x = (M - 1) / block.x + 1;
    // if the image resolution is small, the memory is sufficiently large for output after conv. 
    interp_3d_nputsdriven<<<grid, block>>>(plan->d_u, plan->d_v, plan->d_w, plan->d_c, plan->fw, plan->M,
                                          plan->copts.kw, nf1, nf2, nf3, plan->copts.ES_c, plan->copts.ES_beta,
                                          plan->copts.pirange, plan->cell_loc);
    

    checkCudaErrors(cudaDeviceSynchronize());
  }

  return 0;
}

int curafft_interp(curafft_plan * plan)
{
  /*
  ---- convolution opertion ----
  */

  int ier = 0;
  int nf1 = plan->nf1;
  int nf2 = plan->nf2;
  int nf3 = plan->nf3;
  int M = plan->M;
  // printf("w_term_method %d\n",plan->w_term_method);
  switch (plan->dim)
  {
  case 1:
    interp_1d_invoker(nf1, M, plan);
    break;
  case 2:
    interp_2d_invoker(nf1, nf2, M, plan);
    break;
  case 3:
    interp_3d_invoker(nf1, nf2, nf3, M, plan);
    break;
  default:
    ier = 1; // error
    break;
  }

  return ier;
}

int curaff_partial_conv(){
  
  // improved WS
  // invoke the partial 3d conv, calcualte the conv result and saved the result to plan->fw
  // directly invoke, not packed into function


  // WS
  return 0;
}
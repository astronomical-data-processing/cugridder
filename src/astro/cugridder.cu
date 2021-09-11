/* --------cugridder-----------
    1. gridder_setting
        fov and other astro related setting
        opt setting
        plan setting
        bin setting
    2. gridder_execution
    3. gridder_destroy
*/

#include <iostream>
#include <iomanip>
#include <math.h>
#include <helper_cuda.h>
#include <complex>
#include <cufft.h>
#include <fstream>
#include "conv_interp_invoker.h"
#include "ragridder_plan.h"
#include "curafft_plan.h"
#include "cuft.h"
#include "precomp.h"
#include "ra_exec.h"
#include "utils.h"
#include "cufft.h"
#include "deconv.h"
#include "cugridder.h"

int setup_gridder_plan(int N1, int N2, PCS fov, int lshift, int mshift, int nrow, PCS *d_w, CUCPX *d_c, conv_opts copts, ragridder_plan *gridder_plan, curafft_plan *plan)
{
    /*
    gridder related parameters setting
    Input: 
        N1, N2 - image size
        fov - field of view
        nrow - number of coordinate
        d_w - w array on device
        d_c - vis value
    */
    gridder_plan->fov = fov;
    gridder_plan->width = N1;
    gridder_plan->height = N2;
    gridder_plan->nrow = nrow;
    // determain number of w
    // ignore shift

    // degree per pixel
    gridder_plan->pixelsize_x = fov / 180.0 * PI / (PCS)N2;
    gridder_plan->pixelsize_y = fov / 180.0 * PI / (PCS)N1;
    PCS xpixelsize = gridder_plan->pixelsize_x;
    PCS ypixelsize = gridder_plan->pixelsize_y;
    PCS l_min = lshift - 0.5 * xpixelsize * N2;
    PCS l_max = l_min + xpixelsize * (N2 - 1);

    PCS m_min = mshift - 0.5 * ypixelsize * N1;
    PCS m_max = m_min + ypixelsize * (N1 - 1);

    //double upsampling_fac = copts.upsampfac;
    PCS n_lm = sqrt(1.0 - pow(l_min, 2) - pow(m_min, 2));
    
    // nshift = (no_nshift||(!do_wgridding)) ? 0. : -0.5*(nm1max+nm1min);

    // get max min of input and output
    PCS i_max, i_min;
    PCS o_min;
    get_max_min(i_max, i_min, d_w, gridder_plan->nrow);
    plan->ta.i_center[0] = (i_max + i_min) / (PCS)2.0;
    plan->ta.i_half_width[0] = (i_max - i_min) / (PCS)2.0;

    o_min = n_lm-1;
    plan->ta.o_center[0] =  o_min / (PCS)2.0;
    plan->ta.o_half_width[0] = abs(o_min / (PCS)2.0);

    
    // get number of w planes, scaling ratio gamma
    set_nhg_type3(plan->ta.o_half_width[0], plan->ta.i_half_width[0], plan->copts, plan->nf1, plan->ta.h[0], plan->ta.gamma[0]); //temporately use nf1
#ifdef INFO
    printf("U_width %lf, U_center %lf, X_width %.10lf, X_center %.10lf, gamma %lf, nf %d, h %lf\n",
           plan->ta.i_half_width[0], plan->ta.i_center[0], plan->ta.o_half_width[0], plan->ta.o_center[0], plan->ta.gamma[0], plan->nf1, plan->ta.h[0]);
#endif
    // to cura_prestage
    // wgt * vis
    // if(gridder_plan->kv.weight!=NULL&&plan->copts.direction==1){
    //     PCS *d_wgt;
    //     checkCudaErrors(cudaMalloc((void**)&d_wgt,sizeof(PCS)*nrow));
    //     checkCudaErrors(cudaMemcpy(d_wgt,gridder_plan->kv.weight,sizeof(PCS)*nrow,cudaMemcpyHostToDevice));

    //     matrix_elementwise_multiply_invoker(d_c,d_wgt,nrow);
    //     checkCudaErrors(cudaFree(d_wgt)); // to save memory
    // }
    // // u_j to u_j' x_k to x_k' c_j to c_j'
    // checkCudaErrors(cudaMalloc((void **)&plan->d_x, sizeof(PCS) * (N1 / 2 + 1) * (N2 / 2 + 1)));
    // w_term_k_generation(plan->d_x, N1, N2, gridder_plan->pixelsize_x, gridder_plan->pixelsize_y);

    // pre_stage_invoker(plan->ta.i_center, plan->ta.o_center, plan->ta.gamma, plan->ta.h, d_w, NULL, NULL, plan->d_x, NULL, NULL, d_c, gridder_plan->nrow,(N1 / 2 + 1) * (N2 / 2 + 1), 1, 1, plan->iflag);
    
    
    gridder_plan->num_w = plan->nf1;

    return 0;
}

// the bin sort should be completed at gridder_settting

int gridder_setting(int N1, int N2, int method, int kerevalmeth, int w_term_method, PCS tol, int direction, double sigma, int iflag,
                    int batchsize, int M, int channel, PCS fov, visibility *pointer_v, PCS *d_u, PCS *d_v, PCS *d_w,
                    CUCPX *d_c, curafft_plan *plan, ragridder_plan *gridder_plan)
{
    /*
        N1, N2 - number of Fouier modes
        method - gridding method
        kerevalmeth - gridding kernel evaluation method
        tol - tolerance (epsilon)
        direction - 1 vis to image, 0 image to vis
        sigma - upsampling factor
        iflag - flag for fourier transform
        batchsize - number of batch in  cufft (used for handling piece by piece)
        M - number of nputs (visibility)
        channel - number of channels
        wgt - weight
        freq - frequency
        d_u, d_v, d_w - wavelengths in different dimensions, x is on host, d_x is on device
        d_c - value of visibility
        ****issue, degridding
    */
    int ier = 0;

    // fov and other astro related setting +++

    // get effective coordinates: *1/lambda
    PCS f_over_c = pointer_v->frequency[0]/SPEEDOFLIGHT;
    // printf("foverc %lf\n",f_over_c);
   
    int sign;
    sign = pointer_v->sign;
    get_effective_coordinate_invoker(d_u,d_v,d_w,f_over_c,pointer_v->pirange,M,sign);

    // PCS *w = (PCS *) malloc(sizeof(PCS)*M);
    // checkCudaErrors(cudaMemcpy(w,d_w,sizeof(PCS)*M,cudaMemcpyDeviceToHost));
   
    // opts and copts setting
    plan->opts.gpu_device_id = 0;
    plan->opts.upsampfac = sigma;
    plan->opts.gpu_sort = 0;
    plan->opts.gpu_binsizex = -1;
    plan->opts.gpu_binsizey = -1;
    plan->opts.gpu_binsizez = -1;
    plan->opts.gpu_kerevalmeth = kerevalmeth;
    plan->opts.gpu_conv_only = 0;
    plan->opts.gpu_gridder_method = method;

    
    ier = setup_conv_opts(plan->copts, tol, sigma, 1, direction, kerevalmeth); //check the arguements pirange = 1
    
    int fftsign = (direction > 0) ? 1 : -1;
    plan->iflag = fftsign; 

    if (fftsign==1) plan->type = 1;
    else plan->type = 2; // will be used at deconv

    if (ier != 0)
        printf("setup_error\n");
    
    // gridder plan setting
    // cuda stream malloc in setup_plan
    gridder_plan->channel = channel;
    gridder_plan->w_term_method = w_term_method;
    gridder_plan->speedoflight = SPEEDOFLIGHT;
    gridder_plan->kv.u = pointer_v->u;
    gridder_plan->kv.v = pointer_v->v;
    gridder_plan->kv.w = pointer_v->w;
    gridder_plan->kv.vis = pointer_v->vis;
    gridder_plan->kv.weight = pointer_v->weight;
    gridder_plan->kv.frequency = pointer_v->frequency;
    gridder_plan->kv.pirange = pointer_v->pirange;

    
    setup_gridder_plan(N1, N2, fov, 0, 0, M, d_w, d_c, plan->copts, gridder_plan, plan);

    int nf1 = get_num_cells(N1, plan->copts);
    int nf2 = get_num_cells(N2, plan->copts);
    int nf3 = gridder_plan->num_w;
    if (w_term_method)
        plan->dim = 3;
    else
        plan->dim = 2;
    
    setup_plan(nf1, nf2, nf3, M, d_v, d_u, d_w, d_c, plan);
    // printf("input data checking cugridder...\n");
    //         PCS *temp = (PCS*)malloc(sizeof(PCS)*10);
    //         printf("u v w and vis\n");
    //         cudaMemcpy(temp,d_u,sizeof(PCS)*10,cudaMemcpyDeviceToHost);
    //         for(int i=0;i<10;i++)
    //         printf("%.3lf ",temp[i]);
    //         printf("\n");

    plan->ms = N1;
    plan->mt = N2;
    plan->mu = 1;
    plan->execute_flow = 1;
    //plan->fw = NULL; 
    batchsize = gridder_plan->num_w;

    // plan->copts.direction = direction; // 1 inverse, 0 forward

    // fourier_series_appro_invoker(plan->fwkerhalf1, plan->copts, plan->nf1 / 2 + 1);
    // fourier_series_appro_invoker(plan->fwkerhalf2, plan->copts, plan->nf2 / 2 + 1);

    // if (w_term_method)
    // {
    //     // improved_ws
    //     checkCudaErrors(cudaFree(plan->fwkerhalf3));
    //     checkCudaErrors(cudaMalloc((void **)&plan->fwkerhalf3, sizeof(PCS) * (N1 / 2 + 1) * (N2 / 2 + 1)));
        
    //     fourier_series_appro_invoker(plan->fwkerhalf3, plan->d_x, plan->copts, (N1 / 2 + 1) * (N2 / 2 + 1)); // correction with k, may be wrong, k will be free in this function
    // }

    // PCS *fwkerhalf1 = (PCS *)malloc(sizeof(PCS) * (plan->nf1 / 2 + 1));
    // PCS *fwkerhalf2 = (PCS *)malloc(sizeof(PCS) * (plan->nf2 / 2 + 1));

    // cudaMemcpy(fwkerhalf1, plan->fwkerhalf1, sizeof(PCS) * (plan->nf1 / 2 + 1), cudaMemcpyDeviceToHost);
    // cudaMemcpy(fwkerhalf2, plan->fwkerhalf2, sizeof(PCS) * (plan->nf2 / 2 + 1), cudaMemcpyDeviceToHost);

    // cufft plan setting
    cufftHandle fftplan;
    int n[] = {plan->nf2, plan->nf1};
    int inembed[] = {plan->nf2, plan->nf1};
    int onembed[] = {plan->nf2, plan->nf1};
    
    if(MAX_CUFFT_ELEM/plan->nf1/plan->nf2<plan->nf3){
        batchsize = MAX_CUFFT_ELEM/plan->nf1/plan->nf2;
        cufftHandle fftplanl;
        int remain_batch = plan->nf3%batchsize;
        cufftPlanMany(&fftplanl, 2, n, inembed, 1, inembed[0] * inembed[1],
                  onembed, 1, onembed[0] * onembed[1], CUFFT_TYPE, remain_batch);
        plan->fftplan_l = fftplanl;
    }
    // check, multi cufft for different w ??? how to set
    // cufftCreate(&fftplan);
    // cufftPlan2d(&fftplan,n[0],n[1],CUFFT_TYPE);
    // the bach size sets as the num of w when memory is sufficent. Alternative way, set as a smaller number when memory is insufficient.
    // and handle this piece by piece
    cufftPlanMany(&fftplan, 2, n, inembed, 1, inembed[0] * inembed[1],
                  onembed, 1, onembed[0] * onembed[1], CUFFT_TYPE, batchsize); //There's a hard limit of roughly 2^27 elements in a plan!!!!!!!!!
    plan->fftplan = fftplan;
    plan->batchsize = batchsize;
    
    // u and v scaling *pixelsize
    rescaling_real_invoker(d_u,gridder_plan->pixelsize_x,gridder_plan->nrow);
    rescaling_real_invoker(d_v,gridder_plan->pixelsize_y,gridder_plan->nrow);
    
    // fw malloc
    // printf("nf1, nf2, nf3: (%d,%d,%d) %d\n",plan->nf1,plan->nf2,plan->nf3,plan->nf1*plan->nf2*plan->nf3);
#ifdef INFO
    show_mem_usage();
    printf("nf1, nf2, nf3: (%d,%d,%d) %d\n",plan->nf1,plan->nf2,plan->nf3,plan->nf1*plan->nf2*plan->nf3);
#endif
    
    return ier;
}

int gridder_execution(curafft_plan *plan, ragridder_plan *gridder_plan)
{
    /*
    Execute conv, fft, dft, correction
    */
    int ier = 0;
    // Mult-GPU support: set the CUDA Device ID:
    // int orig_gpu_device_id;
    // cudaGetDevice(& orig_gpu_device_id);
    // cudaSetDevice(d_plan->opts.gpu_device_id);
    int direction = plan->copts.direction;

    if (direction == 1)
    {
        ier = exec_vis2dirty(plan, gridder_plan);
    }
    else
    {
        // forward not implement yet
        ier = exec_dirty2vis(plan, gridder_plan);
    }

    // Multi-GPU support: reset the device ID
    // cudaSetDevice(orig_gpu_device_id);
    return ier;
}

int gridder_destroy(curafft_plan *plan, ragridder_plan *gridder_plan)
{
    // free memory
    int ier = 0;
    checkCudaErrors(cudaFree(plan->d_x));
    curafft_free(plan);
    //checkCudaErrors(cudaDeviceReset());
    free(plan);
    free(gridder_plan->dirty_image);
    free(gridder_plan->kv.u);
    free(gridder_plan->kv.v);
    free(gridder_plan->kv.w);
    free(gridder_plan->kv.vis);
    free(gridder_plan->kv.frequency);
    free(gridder_plan->kv.weight);
    // free(gridder_plan->kv.flag);
    free(gridder_plan);
    return ier;
}

int py_gridder_destroy(curafft_plan *plan, ragridder_plan *gridder_plan)
{   
    // free memory
    int ier=0;
    if (plan->opts.gpu_sort)
    {
        checkCudaErrors(cudaFree(plan->cell_loc));
    }
    cufftDestroy(plan->fftplan);
    checkCudaErrors(cudaFree(plan->d_x));
    checkCudaErrors(cudaFree(plan->fwkerhalf3));
    checkCudaErrors(cudaFree(plan->fwkerhalf2));
    checkCudaErrors(cudaFree(plan->fwkerhalf1));
    checkCudaErrors(cudaFree(plan->d_c));
    checkCudaErrors(cudaFree(plan->fw));
    checkCudaErrors(cudaFree(plan->fk));
    free(plan);
    free(gridder_plan);
    return ier;
}
// -------------gridder warpper-----------------
int ms2dirty_exec(int nrow, int nxdirty, int nydirty, PCS fov, PCS freq, PCS *uvw,
             CPX *vis, PCS *wgt,  CPX *dirty, PCS epsilon, PCS sigma, int sign){
    /*
    generating image from ms(vis)
    Input:
        nrow - number of coordinates
        nxdirty nydirty - image size
        fov - field of view
        freq - freqency
        uvw - coordinate [nrow,3]
        vis - visibility
        epsilon - expected error
        sigma - upsampling factor
    Output:
        d_dirty - dirty image on device
    */
    int ier = 0;
    //checkCudaErrors(cudaSetDevice(0));
#ifdef TIME
    cudaEvent_t start, stop;
	float milliseconds = 0;
	float totaltime = 0;
    float copytime = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
    cudaEventRecord(start);
#endif
    //------------transpose uvw------------

    PCS *d_uvw;
    CUCPX *d_vis;
    checkCudaErrors(cudaMalloc((void**)&d_uvw, 3 * nrow * sizeof(PCS)));
    checkCudaErrors(cudaMemcpy(d_uvw,uvw,3 * nrow * sizeof(PCS),cudaMemcpyHostToDevice));
    matrix_transpose_invoker(d_uvw,3,nrow); // will use a temp arr with same size as uvw
    checkCudaErrors(cudaMalloc((void**)&d_vis, nrow * sizeof(CUCPX)));
    checkCudaErrors(cudaMemcpy(d_vis,  vis, nrow * sizeof(CUCPX), cudaMemcpyHostToDevice));
    
    //------------device memory malloc------------
    PCS *d_u, *d_v, *d_w;
    d_u = d_uvw;
    d_v = d_uvw+nrow;
    d_w = d_uvw+2*nrow;

    PCS *f_over_c = (PCS*) malloc(sizeof(PCS));
    f_over_c[0] = freq / SPEEDOFLIGHT;

    /* -------------- cugridder-----------------*/
	// plan setting
	curafft_plan *plan;

	ragridder_plan *gridder_plan;

	plan = new curafft_plan();
    gridder_plan = new ragridder_plan();
    memset(plan, 0, sizeof(curafft_plan));
    memset(gridder_plan, 0, sizeof(ragridder_plan));
	
	visibility *pointer_v;
	pointer_v = (visibility *)malloc(sizeof(visibility));
	pointer_v->u = uvw;
	pointer_v->v = uvw+nrow;
	pointer_v->w = uvw+2*nrow; //wrong
	pointer_v->vis = vis;
	pointer_v->frequency = &freq;
	pointer_v->weight = wgt;
	pointer_v->pirange = 0;
    pointer_v->sign = sign;
	int direction = 1; //vis to image
    //---------STEP1: gridder setting---------------
    ier = gridder_setting(nydirty,nxdirty,0,0,1,epsilon,direction,sigma,0,1,nrow,1,fov,pointer_v,d_u,d_v,d_w,d_vis
		,plan,gridder_plan);
    //print the setting result
	free(pointer_v);
	if(ier == 1){
		printf("errors in gridder setting\n");
		return ier;
	}
    CUCPX *d_dirty;
    checkCudaErrors(cudaMalloc((void**)&d_dirty,sizeof(CUCPX)*nxdirty*nydirty));
    plan->fk = d_dirty;
#ifdef TIME
    cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	totaltime += milliseconds;
	printf("[time  ] Setting time:\t\t %.3g s\n", (milliseconds)/1000);
#endif

    //---------STEP2: gridder execution---------------
#ifdef TIME
    cudaEventRecord(start);
#endif
    ier = gridder_execution(plan,gridder_plan);
#ifdef TIME
    cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	totaltime += milliseconds;
	printf("[time  ] Exec time:\t\t %.3g s\n", milliseconds/1000);
    cudaEventRecord(start);
#endif
    checkCudaErrors(cudaMemcpy(dirty,plan->fk,sizeof(CUCPX)*nxdirty*nydirty,cudaMemcpyDeviceToHost));
	if(ier == 1){
		printf("errors in gridder execution\n");
		return ier;
	}
    //---------STEP3: gridder destroy-----------------
    checkCudaErrors(cudaFree(d_uvw));
    ier = py_gridder_destroy(plan,gridder_plan);
	if(ier == 1){
		printf("errors in gridder destroy\n");
		return ier;
	}
#ifdef TIME
    cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	totaltime += milliseconds;
	printf("[time  ] Result copy and detroy time:\t\t %.3g s\n", milliseconds/1000);
	printf("[time  ] Total time:\t\t %.3g s\n", totaltime/1000);
#endif
    //checkCudaErrors(cudaDeviceReset());
    return ier;
}

int dirty2ms_exec(int nrow, int nxdirty, int nydirty, PCS fov, PCS freq, PCS *uvw,
             CPX *vis, PCS *wgt,  CPX *dirty, PCS epsilon, PCS sigma, int sign){
    // +++ c dirty c/*.
    /*
    from image to ms(vis)
    Input:
        nrow - number of coordinates
        nxdirty nydirty - image size
        fov - field of view
        freq - freqency
        uvw - coordinate [nrow,3]
        vis - visibility
        epsilon - expected error
        sigma - upsampling factor
    Output:
        d_dirty - dirty image on device
    */
    int ier = 0;
    //checkCudaErrors(cudaSetDevice(0));
#ifdef TIME
    cudaEvent_t start, stop;
	float milliseconds = 0;
	float totaltime = 0;
    float copytime = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
    cudaEventRecord(start);
#endif
    //------------transpose uvw------------

    PCS *d_uvw;
    CUCPX *d_vis;
    checkCudaErrors(cudaMalloc((void**)&d_uvw, 3 * nrow * sizeof(PCS)));
    checkCudaErrors(cudaMemcpy(d_uvw,uvw,3 * nrow * sizeof(PCS),cudaMemcpyHostToDevice));
    matrix_transpose_invoker(d_uvw,3,nrow); // will use a temp arr with same size as uvw
    
    CUCPX *d_dirty;
    checkCudaErrors(cudaMalloc((void**)&d_dirty,sizeof(CUCPX)*nxdirty*nydirty));
    checkCudaErrors(cudaMemcpy(d_dirty,dirty,sizeof(CUCPX)*nxdirty*nydirty,cudaMemcpyHostToDevice));
    //------------device memory malloc------------
    PCS *d_u, *d_v, *d_w;
    d_u = d_uvw;
    d_v = d_uvw+nrow;
    d_w = d_uvw+2*nrow;

    PCS *f_over_c = (PCS*) malloc(sizeof(PCS));
    f_over_c[0] = freq / SPEEDOFLIGHT;

    /* -------------- cugridder-----------------*/
	// plan setting
	curafft_plan *plan;

	ragridder_plan *gridder_plan;

	plan = new curafft_plan();
    gridder_plan = new ragridder_plan();
    memset(plan, 0, sizeof(curafft_plan));
    memset(gridder_plan, 0, sizeof(ragridder_plan));
	
	visibility *pointer_v;
	pointer_v = (visibility *)malloc(sizeof(visibility));
	pointer_v->u = uvw;
	pointer_v->v = uvw+nrow;
	pointer_v->w = uvw+2*nrow; //wrong
	pointer_v->vis = vis;
	pointer_v->frequency = &freq;
	pointer_v->weight = wgt;
	pointer_v->pirange = 0;
    pointer_v->sign = sign;

    plan->fk = d_dirty;
	int direction = 0; 
    //---------STEP1: gridder setting---------------
    ier = gridder_setting(nydirty,nxdirty,0,0,1,epsilon,direction,sigma,-1,1,nrow,1,fov,pointer_v,d_u,d_v,d_w,NULL
		,plan,gridder_plan);
	free(pointer_v);
	if(ier == 1){
		printf("errors in gridder setting\n");
		return ier;
	}

    checkCudaErrors(cudaMalloc((void**)&d_vis, nrow * sizeof(CUCPX)));
    checkCudaErrors(cudaMemset(d_vis,0,nrow * sizeof(CUCPX)));
    plan->d_c = d_vis;
#ifdef TIME
    cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	totaltime += milliseconds;
	printf("[time  ] Setting time:\t\t %.3g s\n", (milliseconds)/1000);
#endif

    //---------STEP2: gridder execution---------------
#ifdef TIME
    cudaEventRecord(start);
#endif
    ier = gridder_execution(plan,gridder_plan);
#ifdef TIME
    cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	totaltime += milliseconds;
	printf("[time  ] Exec time:\t\t %.3g s\n", milliseconds/1000);
    cudaEventRecord(start);
#endif
    checkCudaErrors(cudaMemcpy(vis,d_vis,sizeof(CUCPX)*nrow,cudaMemcpyDeviceToHost));
	if(ier == 1){
		printf("errors in gridder execution\n");
		return ier;
	}
    //---------STEP3: gridder destroy-----------------
    checkCudaErrors(cudaFree(d_uvw));
    ier = py_gridder_destroy(plan,gridder_plan);
	if(ier == 1){
		printf("errors in gridder destroy\n");
		return ier;
	}
#ifdef TIME
    cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	totaltime += milliseconds;
	printf("[time  ] Result copy and detroy time:\t\t %.3g s\n", milliseconds/1000);
	printf("[time  ] Total time:\t\t %.3g s\n", totaltime/1000);
#endif
    //checkCudaErrors(cudaDeviceReset());
    return ier;
}



// a litter bit messy, not know how to handle as one function when wgt can be None or not in python
int ms2dirty_2(int nrow, int nxdirty, int nydirty, PCS fov, PCS freq, PCS *uvw,
             CPX *vis, PCS *wgt, CPX *dirty, PCS epsilon, PCS sigma, int sign){
    int ier = 0;
    ier = ms2dirty_exec(nrow,nxdirty,nydirty,fov,freq,uvw,vis,wgt,dirty,epsilon,sigma,sign);
    return ier;
}

int ms2dirty_1(int nrow, int nxdirty, int nydirty, PCS fov, PCS freq, PCS *uvw,
             CPX *vis, CPX *dirty, PCS epsilon, PCS sigma, int sign){
    int ier = 0;
    ier = ms2dirty_exec(nrow,nxdirty,nydirty,fov,freq,uvw,vis,NULL,dirty,epsilon,sigma,sign);
    return ier;
}

int dirty2ms_1(int nrow, int nxdirty, int nydirty, PCS fov, PCS freq, PCS *uvw,
             CPX *vis, CPX *dirty, PCS epsilon, PCS sigma, int sign){
    int ier = 0;
    ier = dirty2ms_exec(nrow,nxdirty,nydirty,fov,freq,uvw,vis,NULL,dirty,epsilon,sigma,sign);
    return ier;
}

int dirty2ms_2(int nrow, int nxdirty, int nydirty, PCS fov, PCS freq, PCS *uvw,
             CPX *vis, PCS *wgt, CPX *dirty, PCS epsilon, PCS sigma, int sign){
    int ier = 0;
    ier = dirty2ms_exec(nrow,nxdirty,nydirty,fov,freq,uvw,vis,wgt,dirty,epsilon,sigma,sign);
    return ier;
}
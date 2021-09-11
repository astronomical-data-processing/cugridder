/*
INVERSE: type 1
FORWARD: type 2
*/
#include <iostream>
#include <iomanip>
#include <math.h>
#include <helper_cuda.h>
#include <complex>
#include <cufft.h>

#include "curafft_plan.h"
#include "conv_interp_invoker.h"
#include "deconv.h"
#include "precomp.h"
#include "ragridder_plan.h"
#include "ra_exec.h"
#include "cuft.h"
#include "utils.h"


__global__ void div_n_lm(CUCPX *fk, PCS xpixelsize, PCS ypixelsize, int N1, int N2)
{
    int idx;
    PCS n_lm;
    int row, col;
    for (idx = blockDim.x * blockIdx.x + threadIdx.x; idx < N1 * N2; idx += gridDim.x * blockDim.x)
    {
        row = idx / N1;
        col = idx % N1;
        n_lm = sqrt(1.0 - pow((row - N2 / 2) * xpixelsize, 2) - pow((col - N1 / 2) * ypixelsize, 2));

        fk[idx].x /= n_lm;
        fk[idx].y /= n_lm;
        
    }
}

int cura_rscaling(curafft_plan *plan, ragridder_plan *gridder_plan)
{
    //  * 1/n
    int N1 = gridder_plan->width;
    int N2 = gridder_plan->height;
    int N = N1 * N2;
    int blocksize = 256;
    int gridsize = (N - 1) / blocksize + 1;

    div_n_lm<<<gridsize, blocksize>>>(plan->fk, gridder_plan->pixelsize_x, gridder_plan->pixelsize_y, N1, N2);
    checkCudaErrors(cudaDeviceSynchronize());

    return 0;
}


//
__global__ void shift_corr(CUCPX *d_c, PCS *d_w, PCS i_center, PCS o_center, PCS gamma, int nrow, int flag){
        int idx;
        for(idx=threadIdx.x + blockDim.x*blockIdx.x; idx<nrow; idx+=gridDim.x*blockDim.x){
                PCS phase = (d_w[idx]*gamma+i_center)*o_center*flag;
                CUCPX temp;
                temp.x = d_c[idx].x * cos(phase) - d_c[idx].y * sin(phase);
                temp.y = d_c[idx].x * sin(phase) + d_c[idx].y * cos(phase);
                d_c[idx] = temp;
        }
}
int shift_corr_invoker(CUCPX *d_c, PCS *d_w, PCS i_center, PCS o_center, PCS gamma, int nrow, int flag){
    // 
    int blocksize = 512;
    shift_corr<<<(nrow-1)/blocksize+1,blocksize>>>(d_c,d_w,i_center,o_center,gamma,nrow,flag);
    checkCudaErrors(cudaDeviceSynchronize());
    return 0;
}
int cura_fw(curafft_plan *plan, ragridder_plan *gridder_plan){
    // /wgt
    if(gridder_plan->kv.weight!=NULL){
        PCS *d_wgt;
        int nrow = gridder_plan->nrow;
        checkCudaErrors(cudaMalloc((void**)&d_wgt,sizeof(PCS)*nrow));
        checkCudaErrors(cudaMemcpy(d_wgt,gridder_plan->kv.weight,sizeof(PCS)*nrow,cudaMemcpyHostToDevice));

        matrix_elementwise_divide_invoker(plan->d_c,d_wgt,nrow);
        checkCudaErrors(cudaFree(d_wgt)); // to save memory
   }
    int ier = 0;
    int flag = plan->iflag;
    PCS gamma = plan->ta.gamma[0];
    shift_corr_invoker(plan->d_c,plan->d_w,plan->ta.i_center[0],plan->ta.o_center[0],gamma,gridder_plan->nrow,flag);
    return ier;
}


int cura_prestage(curafft_plan *plan, ragridder_plan *gridder_plan){
        int ier = 0;
        int nrow = gridder_plan -> nrow;
        int N1 = plan->ms;
        int N2 = plan->mt;
        if (plan->iflag==1){
                if(gridder_plan->kv.weight!=NULL&&plan->copts.direction==1){
                        PCS *d_wgt;
                        checkCudaErrors(cudaMalloc((void**)&d_wgt,sizeof(PCS)*nrow));
                        checkCudaErrors(cudaMemcpy(d_wgt,gridder_plan->kv.weight,sizeof(PCS)*nrow,cudaMemcpyHostToDevice));

                        matrix_elementwise_multiply_invoker(plan->d_c,d_wgt,nrow);
                        checkCudaErrors(cudaFree(d_wgt)); // to save memory
                }
                // u_j to u_j' x_k to x_k' c_j to c_j'
                checkCudaErrors(cudaMalloc((void **)&plan->d_x, sizeof(PCS) * (N1 / 2 + 1) * (N2 / 2 + 1)));
                w_term_k_generation(plan->d_x, N1, N2, gridder_plan->pixelsize_x, gridder_plan->pixelsize_y);
                pre_stage_1_invoker(plan->ta.o_center,plan->d_w,NULL,NULL,plan->d_c,nrow,plan->iflag);
                pre_stage_2_invoker(plan->ta.i_center, plan->ta.o_center, plan->ta.gamma, plan->ta.h, plan->d_w, NULL, NULL, plan->d_x, NULL, NULL, plan->d_c, gridder_plan->nrow,(N1 / 2 + 1) * (N2 / 2 + 1), 1, 1);
                
        }
        else{
                
                // u_j to u_j' x_k to x_k' fk to fk'
                checkCudaErrors(cudaMalloc((void **)&plan->d_x, sizeof(PCS) * (N1 / 2 + 1) * (N2 / 2 + 1)));
                w_term_k_generation(plan->d_x, N1, N2, gridder_plan->pixelsize_x, gridder_plan->pixelsize_y);
                // pre_stage_1_invoker(plan->ta.i_center[0],plan->d_x,plan->fk,N1,N2,gridder_plan->pixelsize_x, gridder_plan->pixelsize_y,plan->iflag);
                pre_stage_2_invoker(plan->ta.i_center, plan->ta.o_center, plan->ta.gamma, plan->ta.h, plan->d_w, NULL, NULL, plan->d_x, NULL, NULL, plan->d_c, gridder_plan->nrow,(N1 / 2 + 1) * (N2 / 2 + 1), 1, 1);
        }

        fourier_series_appro_invoker(plan->fwkerhalf1, plan->copts, plan->nf1 / 2 + 1);
        fourier_series_appro_invoker(plan->fwkerhalf2, plan->copts, plan->nf2 / 2 + 1);
        int w_term_method = 1;
        if (w_term_method)
        {
                // improved_ws
                checkCudaErrors(cudaFree(plan->fwkerhalf3));
                checkCudaErrors(cudaMalloc((void **)&plan->fwkerhalf3, sizeof(PCS) * (N1 / 2 + 1) * (N2 / 2 + 1)));
                fourier_series_appro_invoker(plan->fwkerhalf3, plan->d_x, plan->copts, (N1 / 2 + 1) * (N2 / 2 + 1)); // correction with k, may be wrong, k will be free in this function
        }
        return ier;
}

// int cura_cscaling(curafft_plan *plan, ragridder_plan *gridder_plan)
// {
//     int N = gridder_plan->nrow;
//     return 0;
// }

int exec_vis2dirty(curafft_plan *plan, ragridder_plan *gridder_plan)
{
    /*
    Currently, just suitable for improved W stacking
    Two different execution flows
        Flow1: the data size is relatively small and memory is sufficent for whole conv
        Flow2: the data size is too large, the data is divided into parts 
    */
    int ier = 0;
    //printf("execute flow %d\n",plan->execute_flow);
    if (plan->execute_flow == 1)
    {
        /// curafft_conv workflow for enough memory
#ifdef DEBUG
        printf("plan info printing...\n");
        printf("nf (%d,%d,%d), upsampfac %lf\n", plan->nf1, plan->nf2, plan->nf3, plan->copts.upsampfac);
        printf("gridder_plan info printing...\n");
        printf("fov %lf, current channel %d, w_s_r %lf\n", gridder_plan->fov, gridder_plan->cur_channel, gridder_plan->w_s_r);
#endif
        // 0. pre-stage
        ier = cura_prestage(plan,gridder_plan);
        // 1. convlution
#ifdef TIME
        cudaEvent_t start, stop;
        float milliseconds = 0;
        float totaltime = 0;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
#endif
        ier = curafft_conv(plan);
#ifdef TIME
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        totaltime += milliseconds;
        printf("[time  ] conv time:\t\t %.3g s\n", milliseconds / 1000);
#endif
       
#ifdef DEBUG
        printf("conv result printing (first w plane)...\n");
        CPX *fw = (CPX *)malloc(sizeof(CPX) * plan->nf1 * plan->nf2 * plan->nf3);
        cudaMemcpy(fw, plan->fw, sizeof(CUCPX) * plan->nf1 * plan->nf2 * plan->nf3, cudaMemcpyDeviceToHost);
        PCS temp = 0;
        for (int i = plan->nf1 * plan->nf2; i < 200; i++)
        {
                temp += fw[i].real();
                printf("%.3g ", fw[i].real());
        }

#endif
        // printf("n1 n2 n3 M %d, %d, %d, %d\n",plan->nf1,plan->nf2,plan->nf3,plan->M);
        // 2. cufft
#ifdef TIME
        cudaEventRecord(start);
#endif
        cura_cufft(plan);
        // int direction = plan->iflag;
        // // cautious, a batch of fft, bath size is num_w when memory is sufficent.
        
        // CUFFT_EXEC(plan->fftplan, plan->fw, plan->fw, direction); // sychronized or not
        // cudaDeviceSynchronize();
#ifdef TIME
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        totaltime += milliseconds;
        printf("[time  ] cufft time:\t\t %.3g s\n", milliseconds / 1000);
#endif
#ifdef DEBUG
        printf("fft result printing (first w plane)...\n");
        //CPX *fw = (CPX *)malloc(sizeof(CPX)*plan->nf1*plan->nf2*plan->nf3);
        cudaMemcpy(fw, plan->fw, sizeof(CUCPX) * plan->nf1 * plan->nf2 * plan->nf3, cudaMemcpyDeviceToHost);
        for (int i = 0; i < 10; i++)
        {
                printf("%.3g ", fw[i].real());
            printf("\n");
        }
#endif
        // keep the N1*N2*num_w. ignore the outputs that are out of range

        // 3. dft on w (or 1 dimensional nufft type3)
#ifdef TIME
        cudaEventRecord(start);
#endif
        curadft_invoker(plan, gridder_plan->pixelsize_x, gridder_plan->pixelsize_y);
#ifdef TIME
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        totaltime += milliseconds;
        printf("[time  ] dft w time:\t\t %.3g s\n", milliseconds / 1000);
#endif
#ifdef DEBUG
        printf("part of dft result printing:...\n");
        //CPX *fw = (CPX *)malloc(sizeof(CPX)*plan->nf1*plan->nf2*plan->nf3);
        cudaMemcpy(fw, plan->fw, sizeof(CUCPX) * plan->nf1 * plan->nf2 * plan->nf3, cudaMemcpyDeviceToHost);
        for (int i = 0; i < 10; i++)
        {
            
                printf("%.3g ", fw[i ].real());
            printf("\n");
        }
#endif
        // 4. deconvolution (correction)
        // error detected, 1. w term deconv
        // 1. 2D deconv towards u and v
#ifdef TIME
        cudaEventRecord(start);
#endif
        plan->dim = 2;
        ier = curafft_deconv(plan);
#ifdef TIME
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        totaltime += milliseconds;
        printf("[time  ] 2d deconv time:\t\t %.3g s\n", milliseconds / 1000);
#endif
#ifdef DEBUG
        printf("deconv result printing stage 1:...\n");
        CPX *fk = (CPX *)malloc(sizeof(CPX) * plan->ms * plan->mt);
        cudaMemcpy(fk, plan->fk, sizeof(CUCPX) * plan->ms * plan->mt, cudaMemcpyDeviceToHost);
        for (int i = 0; i < 10; i++)
        {
            
                printf("%.5lf ", fk[i].real());
            printf("\n");
        }
#endif
        // 2. w term deconv on fk
#ifdef TIME
        cudaEventRecord(start);
#endif
        ier = curadft_w_deconv(plan, gridder_plan->pixelsize_x, gridder_plan->pixelsize_y);
#ifdef TIME
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        totaltime += milliseconds;
        printf("[time  ] w term deconv time:\t\t %.3g s\n", milliseconds / 1000);
#endif
#ifdef DEBUG
        printf("deconv result printing stage 2:...\n");
        //CPX *fk = (CPX *)malloc(sizeof(CPX)*plan->ms*plan->mt);
        cudaMemcpy(fk, plan->fk, sizeof(CUCPX) * plan->ms * plan->mt, cudaMemcpyDeviceToHost);
        for (int i = 0; i < 10; i++)
        {
            
                printf("%.5lf ", fk[i].real());
            printf("\n");
        }
#endif
        // 5. ending work - scaling
        // /n_lm, fourier related rescale
#ifdef TIME
        cudaEventRecord(start);
#endif
        cura_rscaling(plan, gridder_plan);
#ifdef TIME
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        totaltime += milliseconds;
        printf("[time  ] end work time:\t\t %.3g s\n", milliseconds / 1000);
        printf("[time  ] Total time:\t\t %.3g s\n", totaltime/1000);
#endif    
    }
    else if (plan->execute_flow == 2)
    {
        /// curafft_partial_conv workflow for insufficient memory

        // offset array with size of
        for (int i = 0; i < gridder_plan->num_w; i += plan->batchsize)
        {
            //memory allocation of fw may cause error, if size is too large, decrease the batchsize.
            checkCudaErrors(cudaMemset(plan->fw, 0, plan->batchsize * plan->nf1 * plan->nf2 * sizeof(CUCPX)));
            // 1. convlution
            curafft_conv(plan);
        }
    }
    return ier;
}

int exec_dirty2vis(curafft_plan *plan, ragridder_plan *gridder_plan){
    int ier=0;
    

    cura_prestage(plan,gridder_plan);

    // 1. scaling (/n) 
    cura_rscaling(plan, gridder_plan);

    // 2. deconvolution
    // 2.1 w term
    ier = curadft_w_deconv(plan, gridder_plan->pixelsize_x, gridder_plan->pixelsize_y); // fk * e-

#ifdef DEBUG
    printf("deconv result printing stage 2:...\n");
        CPX *fk = (CPX *)malloc(sizeof(CPX)*plan->ms*plan->mt);
        cudaMemcpy(fk, plan->fk, sizeof(CUCPX) * plan->ms * plan->mt, cudaMemcpyDeviceToHost);
        for(int j=0; j<plan->mt; j++){
                for (int i = 0; i < plan->ms; i++)
        {
                printf("%.5g ", fk[i+plan->ms*j].real());   
        }
        printf("\n");
        }
        
        free(fk);
#endif  
    // 2.2 2D deconv towards u and v
    plan->dim = 2;
    ier = curafft_deconv(plan); // need to revise, back to FFTW mode?
    plan->dim = 3;
#ifdef DEBUG
        printf("deconv result printing\n");
        CPX *fw = (CPX *)malloc(sizeof(CPX)*plan->nf1*plan->nf2*plan->nf3);
        cudaMemcpy(fw, plan->fw, sizeof(CUCPX) * plan->nf1 * plan->nf2 * plan->nf3, cudaMemcpyDeviceToHost);
        for (int j = 0; j<1; j++){
                printf("plane %d begin..............................................................\n",j);
                for (int i = 0; i <  plan->nf1 * plan->nf2; i++)
                {
                        printf("%.6g ", fw[i+j*plan->nf1*plan->nf2].real());
                        if(i%plan->nf1==0)printf("\n");
                        
                }
                printf("\n");
        }
        free(fw);
#endif
    // 3. idft
    curadft_invoker(plan, gridder_plan->pixelsize_x, gridder_plan->pixelsize_y);


#ifdef DEBUG
     printf("idft result printing\n");
        CPX *fw = (CPX *)malloc(sizeof(CPX)*plan->nf1*plan->nf2*plan->nf3);
        cudaMemcpy(fw, plan->fw, sizeof(CUCPX) * plan->nf1 * plan->nf2 * plan->nf3, cudaMemcpyDeviceToHost);
        for (int j = 0; j<plan->nf3; j++){
                printf("plane %d begin..............................................................\n",j);
                for (int i = plan->nf1 * plan->nf2-plan->nf2-2; i <  plan->nf1 * plan->nf2-plan->nf2-1; i++)
                {
                        if(i%plan->nf1==0)printf("\n");
                        printf("%.10g ", fw[i+j*plan->nf1*plan->nf2].real());
                        
                        
                }
                printf("\n");
        }
#endif
        //free(fw);

    // 4. fft
        cura_cufft(plan);
#ifdef DEBUG
        printf("fft result printing\n");
        //CPX *fw = (CPX *)malloc(sizeof(CPX)*plan->nf1*plan->nf2*plan->nf3);
        cudaMemcpy(fw, plan->fw, sizeof(CUCPX) * plan->nf1 * plan->nf2 * plan->nf3, cudaMemcpyDeviceToHost);
        printf("%d,%d,%d\n",plan->nf1,plan->nf2,plan->nf3);
        int count=0;
        for (int j = 0; j<plan->nf3; j++){
                printf("plane %d begin..............................................................\n",j);
                for (int i = 0; i < plan->nf1*plan->nf2; i++)
                {
                        if(fw[i+j*plan->nf1*plan->nf2].real()==0)
                        count++;
                        
                }
                printf("\n");
        }
        printf("count: %d\n",count);
        
        CPX *c = (CPX *)malloc(sizeof(CPX) * plan->M);
        //CPX *fw = (CPX *)malloc(sizeof(CPX)*plan->nf1*plan->nf2*plan->nf3);
        cudaMemcpy(c, plan->d_c, sizeof(CUCPX) * plan->M, cudaMemcpyDeviceToHost);

#endif
        // * e-izw_j
    // 5. interpolation
    curafft_interp(plan);
#ifdef DEBUG
        printf("interp result printing (first w plane)...\n");
        cudaMemcpy(c, plan->d_c, sizeof(CUCPX) * plan->M, cudaMemcpyDeviceToHost);
        PCS temp = 0;
        for (int i = 0; i < 100; i++)
        {
                temp += c[i].real();
                printf("%.6g ", c[i].real());
        }
        free(c);
#endif
    // 6. final work (/wgt, *e)
    cura_fw(plan,gridder_plan); // 
    return ier;
}
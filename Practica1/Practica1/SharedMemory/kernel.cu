//Ángel Cruz Alonso
//Ignacio Nicolás López

//Tal como explicamos en el README no hemos conseguido optimizar el tiempo ni hallar una solución que resuelva
// el problema de los margenes de los bloques.
//Te lo enviamos igualmente por si pudieses calificar el esfuerzo, y ayudarnos a conseguir una solución apta.

#include <cuda.h>
#include <math.h>
#include <stdio.h>
/* Time */
#include <sys/time.h>
#include <sys/resource.h>
#include "kernel.h"
#define DIMBLOCK 16
#define PI 3.141593

__global__ void NRAux(float *im, float *NR,  int height, int width) {

  int bx = blockIdx.x;
  int by = blockIdx.y;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int i = by*DIMBLOCK + ty;
  int j = bx*DIMBLOCK + tx;
  float NRsub = 0;

   __shared__ float ims[DIMBLOCK][DIMBLOCK];
   ims[ty][tx] = im[i * width + j];
    
   __syncthreads();

   if(ty > 2 && tx > 2 && ty < DIMBLOCK - 2 && tx < DIMBLOCK - 2 
    ){

      NRsub =
  				 (2.0*ims[ty-2][tx-2] +  4.0*ims[ty-2][tx-1] +  5.0*ims[ty-2][tx] +  4.0*ims[ty-2][tx+1] + 2.0*ims[ty-2][tx+2]
  				+ 4.0*ims[ty-1][tx-2] +  9.0*ims[ty-1][tx-1] + 12.0*ims[ty-1][tx] +  9.0*ims[ty-1][tx+1] + 4.0*ims[ty-1][tx+2]
				+ 5.0*ims[ty][tx-2]   + 12.0*ims[ty][tx-1]   + 15.0*ims[ty][tx]   + 12.0*ims[ty][tx+1]   + 5.0*ims[ty][tx+2]
  				+ 4.0*ims[ty+1][tx-2] +  9.0*ims[ty+1][tx-1] + 12.0*ims[ty+1][tx] +  9.0*ims[ty+1][tx+1] + 4.0*ims[ty+1][tx+2]
  				+ 2.0*ims[ty+2][tx-2] +  4.0*ims[ty+2][tx-1] +  5.0*ims[ty+2][tx] +  4.0*ims[ty+2][tx+1] + 2.0*ims[ty+2][tx+2])
  				/159.0;
      
    __syncthreads();
  
    
   }
   NR[i*width+j] = NRsub;
}


__global__ void GradientAux(float *NR, float *Gx, float *Gy, float *G, float *phi, int height, int width){

  int bx = blockIdx.x;
  int by = blockIdx.y;

  int tx = threadIdx.x;
  int ty = threadIdx.y;


  int i = by*DIMBLOCK + ty;
  int j = bx*DIMBLOCK + tx;

   float Gxsub = 0;
   float Gysub = 0;
   __shared__ float NRs[DIMBLOCK][DIMBLOCK];
   NRs[ty][tx] = NR[j + i * width];
   __syncthreads();

 if(ty > 2 && tx > 2 && ty < DIMBLOCK - 2 && tx < DIMBLOCK - 2 
    ){
    Gxsub = 
				 (1.0*NRs[(ty-2)][(tx-2)] +  2.0*NRs[(ty-2)][(tx-1)] +  (-2.0)*NRs[(ty-2)][(tx+1)] + (-1.0)*NRs[(ty-2)][(tx+2)]
				+ 4.0*NRs[(ty-1)][(tx-2)] +  8.0*NRs[(ty-1)][(tx-1)] +  (-8.0)*NRs[(ty-1)][(tx+1)] + (-4.0)*NRs[(ty-1)][(tx+2)]
				+ 6.0*NRs[(ty  )][(tx-2)] + 12.0*NRs[(ty  )][(tx-1)] + (-12.0)*NRs[(ty  )][(tx+1)] + (-6.0)*NRs[(ty  )][(tx+2)]
				+ 4.0*NRs[(ty+1)][(tx-2)] +  8.0*NRs[(ty+1)][(tx-1)] +  (-8.0)*NRs[(ty+1)][(tx+1)] + (-4.0)*NRs[(ty+1)][(tx+2)]
				+ 1.0*NRs[(ty+2)][(tx-2)] +  2.0*NRs[(ty+2)][(tx-1)] +  (-2.0)*NRs[(ty+2)][(tx+1)] + (-1.0)*NRs[(ty+2)][(tx+2)]);


    Gysub = 
				 ((-1.0)*NRs[(ty-2)][(tx-2)] + (-4.0)*NRs[(ty-2)][(tx-1)] +  (-6.0)*NRs[(ty-2)][(tx)] + (-4.0)*NRs[(ty-2)][(tx+1)] + (-1.0)*NRs[(ty-2)][(tx+2)]
				+ (-2.0)*NRs[(ty-1)][(tx-2)] + (-8.0)*NRs[(ty-1)][(tx-1)] + (-12.0)*NRs[(ty-1)][(tx)] + (-8.0)*NRs[(ty-1)][(tx+1)] + (-2.0)*NRs[(ty-1)][(tx+2)]
				+    2.0*NRs[(ty+1)][(tx-2)] +    8.0*NRs[(ty+1)][(tx-1)] +    12.0*NRs[(ty+1)][(tx)] +    8.0*NRs[(ty+1)][(tx+1)] +    2.0*NRs[(ty+1)][(tx+2)]
				+    1.0*NRs[(ty+2)][(tx-2)] +    4.0*NRs[(ty+2)][(tx-1)] +     6.0*NRs[(ty+2)][(tx)] +    4.0*NRs[(ty+2)][(tx+1)] +    1.0*NRs[(ty+2)][(tx+2)]);

    __syncthreads();
    Gx[i*width+j] = Gxsub;
    Gy[i*width+j] = Gysub;
    

    G[i*width+j]   = sqrtf((Gx[i*width+j]*Gx[i*width+j])+(Gy[i*width+j]*Gy[i*width+j]));	//G = √Gx²+Gy²
    phi[i*width+j] = atan2f(fabs(Gy[i*width+j]),fabs(Gx[i*width+j]));

    if(fabs(phi[i*width+j])<=PI/8 )
      phi[i*width+j] = 0;
    else if (fabs(phi[i*width+j])<= 3*(PI/8))
      phi[i*width+j] = 45;
    else if (fabs(phi[i*width+j]) <= 5*(PI/8))
      phi[i*width+j] = 90;
    else if (fabs(phi[i*width+j]) <= 7*(PI/8))
      phi[i*width+j] = 135;
    else phi[i*width+j] = 0;
  }
}


__global__ void PedgeAux(float *G, float *pedge, float *phi, int height, int width){

  int bx = blockIdx.x;
  int by = blockIdx.y;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

 
  int i = by*DIMBLOCK + ty;
  int j = bx*DIMBLOCK + tx;
  
   float pedgesub = 0;
  
   __shared__ float Gs[DIMBLOCK][DIMBLOCK];
   Gs[ty][tx] = G[j + i * width];
   __syncthreads();

   if(ty > 3 && tx > 3 && ty < DIMBLOCK - 3 && tx < DIMBLOCK - 3 
    ){
    			if(phi[i*width+j] == 0){
				if(Gs[ty][tx]>Gs[ty][tx+1] && Gs[ty][tx]>Gs[ty][tx-1]) //edge is in n-s
					pedgesub = 1;

			} else if(phi[i*width+j] == 45) {
				if(Gs[ty][tx]>Gs[(ty+1)][tx+1] && Gs[ty][tx]>Gs[(ty-1)][tx-1]) // edge is in nw-se
					pedgesub = 1;

			} else if(phi[i*width+j] == 90) {
				if(Gs[ty][tx]>Gs[(ty+1)][tx] && Gs[ty][tx]>Gs[(ty-1)][tx]) //edge is in e-w
					pedgesub = 1;

			} else if(phi[i*width+j] == 135) {
				if(Gs[ty][tx]>Gs[(ty+1)][tx-1] && Gs[ty][tx]>Gs[(ty-1)][tx+1]) // edge is in ne-sw
					pedgesub = 1;
			}
  }

     __syncthreads();

    pedge[i*width +j] =  pedgesub;

} 

__global__ void thresholding(float level,float *G, float *image_out, int width, int height, float *pedge){
	float lowthres = level/2;
	float hithres  = 2*(level);
 	int bx = blockIdx.x;
  	int by = blockIdx.y;
	int ii,jj;
  	int tx = threadIdx.x;
  	int ty = threadIdx.y;


	int i = by*DIMBLOCK + ty;
	int j = bx*DIMBLOCK + tx;

	float image_outsub = 0;
  
	__shared__ float Gs[DIMBLOCK][DIMBLOCK];
	__shared__ float pedges[DIMBLOCK][DIMBLOCK];
	Gs[ty][tx] = G[j + i * width];
	pedges[ty][tx] = pedge[j+i*width];
	__syncthreads();
	image_out[i*width+j] = 0;	

 if(ty > 3 && tx > 3 && ty < DIMBLOCK - 3 && tx < DIMBLOCK - 3 
    ){
		if(Gs[ty][tx]>hithres && pedges[ty][tx])
			image_outsub = 255;
		else if(pedges[ty][tx] && Gs[ty][tx]>=lowthres && Gs[ty][tx]<hithres)
			// check neighbours 3x3
			for (ii=-1;ii<=1; ii++)
				for (jj=-1;jj<=1; jj++)
					if (Gs[(ty+ii)][tx+jj]>hithres)
						image_outsub = 255;
	}
	__syncthreads();
	image_out[i*width+ j] = image_outsub;

}





double get_time(){
	static struct timeval 	tv0;
	double time_, mytime;

	gettimeofday(&tv0,(struct timezone*)0);
	time_=(double)((tv0.tv_usec + (tv0.tv_sec)*1000000));
	mytime = time_/1000000;
	return(mytime);
}



void cannyGPU(float *im, float *image_out,
	float level,
	int height, int width)
{
  float *imOrig, *imFin, *Gx, *Gy, *G, *phi, *NR, *pedge;  
  double t0, t1;
  dim3 numThreads = dim3(DIMBLOCK,DIMBLOCK);
  dim3 dimBlock = dim3(ceil(height/DIMBLOCK),ceil(width/DIMBLOCK));

  
  cudaMalloc((void**)&imOrig, height*width*sizeof(float));
  cudaMalloc((void**)&imFin, height*width*sizeof(float));
  cudaMalloc((void**)&Gx, height*width*sizeof(float));
  cudaMalloc((void**)&Gy, height*width*sizeof(float));
  cudaMalloc((void**)&G, height*width*sizeof(float));
  cudaMalloc((void**)&phi, height*width*sizeof(float));
  cudaMalloc((void**)&NR, height*width*sizeof(float));
  cudaMalloc((void**)&pedge, height*width*sizeof(float));
  
  cudaMemcpy(imOrig,im,height*width*sizeof(float),cudaMemcpyHostToDevice);

   t0 = get_time();

  NRAux<<<dimBlock,numThreads>>>(imOrig, NR, height, width);
  GradientAux<<<dimBlock,numThreads>>>(NR, Gx, Gy, G, phi, height, width);
  PedgeAux<<<dimBlock,numThreads>>>(G, pedge, phi, height, width);
  thresholding<<<dimBlock,numThreads>>>(level,G,imFin,width,height, pedge);
  
  t1 = get_time();
  printf("GPU REAL Exection time %f ms.\n", t1-t0);
  cudaMemcpy(image_out,imFin,height*width*sizeof(float),cudaMemcpyDeviceToHost);
}


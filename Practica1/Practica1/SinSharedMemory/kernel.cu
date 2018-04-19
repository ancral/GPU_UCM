//Ángel Cruz Alonso
//Ignacio Nicolás López

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

  NR[i*width+j] = 0;

  if(i > 2 && j > 2 && i < width - 2 && j < height - 2){
    NR[i*width+j] =
				 (2.0*im[(i-2)*width+(j-2)] +  4.0*im[(i-2)*width+(j-1)] +  5.0*im[(i-2)*width+(j)] +  4.0*im[(i-2)*width+(j+1)] + 2.0*im[(i-2)*width+(j+2)]
				+ 4.0*im[(i-1)*width+(j-2)] +  9.0*im[(i-1)*width+(j-1)] + 12.0*im[(i-1)*width+(j)] +  9.0*im[(i-1)*width+(j+1)] + 4.0*im[(i-1)*width+(j+2)]
				+ 5.0*im[(i  )*width+(j-2)] + 12.0*im[(i  )*width+(j-1)] + 15.0*im[(i  )*width+(j)] + 12.0*im[(i  )*width+(j+1)] + 5.0*im[(i  )*width+(j+2)]
				+ 4.0*im[(i+1)*width+(j-2)] +  9.0*im[(i+1)*width+(j-1)] + 12.0*im[(i+1)*width+(j)] +  9.0*im[(i+1)*width+(j+1)] + 4.0*im[(i+1)*width+(j+2)]
				+ 2.0*im[(i+2)*width+(j-2)] +  4.0*im[(i+2)*width+(j-1)] +  5.0*im[(i+2)*width+(j)] +  4.0*im[(i+2)*width+(j+1)] + 2.0*im[(i+2)*width+(j+2)])
				/159.0;

  }
}


__global__ void GradientAux(float *NR, float *Gx, float *Gy, float *G, float *phi, int height, int width){

  int bx = blockIdx.x;
  int by = blockIdx.y;

  int tx = threadIdx.x;
  int ty = threadIdx.y;


  int i = by*DIMBLOCK + ty;
  int j = bx*DIMBLOCK + tx;

  G[i*width+j] = 0;
  phi[i*width+j] = 0;

  if(i > 2 && j > 2 && i < width - 2 && j < height - 2){

    Gx[i*width+j] = 
				 (1.0*NR[(i-2)*width+(j-2)] +  2.0*NR[(i-2)*width+(j-1)] +  (-2.0)*NR[(i-2)*width+(j+1)] + (-1.0)*NR[(i-2)*width+(j+2)]
				+ 4.0*NR[(i-1)*width+(j-2)] +  8.0*NR[(i-1)*width+(j-1)] +  (-8.0)*NR[(i-1)*width+(j+1)] + (-4.0)*NR[(i-1)*width+(j+2)]
				+ 6.0*NR[(i  )*width+(j-2)] + 12.0*NR[(i  )*width+(j-1)] + (-12.0)*NR[(i  )*width+(j+1)] + (-6.0)*NR[(i  )*width+(j+2)]
				+ 4.0*NR[(i+1)*width+(j-2)] +  8.0*NR[(i+1)*width+(j-1)] +  (-8.0)*NR[(i+1)*width+(j+1)] + (-4.0)*NR[(i+1)*width+(j+2)]
				+ 1.0*NR[(i+2)*width+(j-2)] +  2.0*NR[(i+2)*width+(j-1)] +  (-2.0)*NR[(i+2)*width+(j+1)] + (-1.0)*NR[(i+2)*width+(j+2)]);


    Gy[i*width+j] = 
				 ((-1.0)*NR[(i-2)*width+(j-2)] + (-4.0)*NR[(i-2)*width+(j-1)] +  (-6.0)*NR[(i-2)*width+(j)] + (-4.0)*NR[(i-2)*width+(j+1)] + (-1.0)*NR[(i-2)*width+(j+2)]
				+ (-2.0)*NR[(i-1)*width+(j-2)] + (-8.0)*NR[(i-1)*width+(j-1)] + (-12.0)*NR[(i-1)*width+(j)] + (-8.0)*NR[(i-1)*width+(j+1)] + (-2.0)*NR[(i-1)*width+(j+2)]
				+    2.0*NR[(i+1)*width+(j-2)] +    8.0*NR[(i+1)*width+(j-1)] +    12.0*NR[(i+1)*width+(j)] +    8.0*NR[(i+1)*width+(j+1)] +    2.0*NR[(i+1)*width+(j+2)]
				+    1.0*NR[(i+2)*width+(j-2)] +    4.0*NR[(i+2)*width+(j-1)] +     6.0*NR[(i+2)*width+(j)] +    4.0*NR[(i+2)*width+(j+1)] +    1.0*NR[(i+2)*width+(j+2)]);

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

  pedge[i*width+j] = 0;

  if(i > 3 && j > 3 && i < width - 3 && j < height - 3){
    			if(phi[i*width+j] == 0){
				if(G[i*width+j]>G[i*width+j+1] && G[i*width+j]>G[i*width+j-1]) //edge is in N-S
					pedge[i*width+j] = 1;

			} else if(phi[i*width+j] == 45) {
				if(G[i*width+j]>G[(i+1)*width+j+1] && G[i*width+j]>G[(i-1)*width+j-1]) // edge is in NW-SE
					pedge[i*width+j] = 1;

			} else if(phi[i*width+j] == 90) {
				if(G[i*width+j]>G[(i+1)*width+j] && G[i*width+j]>G[(i-1)*width+j]) //edge is in E-W
					pedge[i*width+j] = 1;

			} else if(phi[i*width+j] == 135) {
				if(G[i*width+j]>G[(i+1)*width+j-1] && G[i*width+j]>G[(i-1)*width+j+1]) // edge is in NE-SW
					pedge[i*width+j] = 1;
			}
  }
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
	
	image_out[i*width+j] = 0;	

	if(i > 3 && j > 3 && i < width - 3 && j < height - 3){
		if(G[i*width+j]>hithres && pedge[i*width+j])
			image_out[i*width+j] = 255;
		else if(pedge[i*width+j] && G[i*width+j]>=lowthres && G[i*width+j]<hithres)
			// check neighbours 3x3
			for (ii=-1;ii<=1; ii++)
				for (jj=-1;jj<=1; jj++)
					if (G[(i+ii)*width+j+jj]>hithres)
						image_out[i*width+j] = 255;
	}

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


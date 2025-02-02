#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>


void Mul(float *A, float *B, int hA, int wA, int wB, float *C)
{
	int i,j,k;

	for (i=0; i<hA; i++)
		for (j=0; j<wB; j++){
			C[i*wB+j] = 0.0;
			for (k=0; k<wA; k++)
				C[i*wB+j] += A[i*wA+k]*B[k*wB+j];
		}
}

__global__ void MulGpu(float *A, float *B, float *C, int hA, int wA, int wB)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;

	if(i < hA && j < wB) for(int k = 0; k < wA ; k++) C[i + j * wB] += A[i + k*hA] + B[k + j*wA];
}

void init_matrix(float *M, int hM, int wM, float k)
{
	int i,j;

	for (i=0; i<hM; i++)
		for (j=0; j<wM; j++)
			if (i==j)
				M[i*wM+j] = k*1.0f;
			else
				M[i*wM+j] = -1.0f/(float)(wM);
}

void print_matrix(float *M, int hM, int wM)
{
	int i,j;

	for (i=0; i<hM; i++){
//		printf("Line %i: ", i);
		for (j=0; j<wM; j++)
			printf("%4.1f ", M[i*wM+j]);
		printf("\n");
	}
}

int diff(float *A, float *B, int hA, int wA, int wB, float *C)
{
	float *C_cpu;
	int size_C = wB * hA;
	C_cpu = (float*)malloc(size_C*sizeof(float));

	int i,j,k;

	for (i=0; i<hA; i++)
		for (j=0; j<wB; j++){
			C_cpu[i*wB+j] = 0.0;
			for (k=0; k<wA; k++){
				C_cpu[i*wB+j] += A[i*wA+k]*B[k*wB+j];
			}
		}
	//printf("\n\nMATRIX C_cpu\n");print_matrix(C_cpu, hA, wB);

	for (i=0; i<hA; i++)
		for (j=0; j<wB; j++)
			if (fabsf(C_cpu[i*wB+j]-C[i*wB+j])>1e-5)
			{
				printf("[%i,%i]: %f!=%f\n", i, j, C_cpu[i*wB+j], C[i*wB+j]);
				return(0);
			}


	return(1);

}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
	// Matrix variables
	float *A, *B, *C;
	float *A_G, *B_G, *C_G;
	int hA, wA, hB, wB;
	int i;

	setbuf(stdout, NULL);

	if (argc!=4){
		printf("./exec hA hB/WA wB\n");
		exit(-1);
	}

	hA = atoi(argv[1]);
	hB = wA = atoi(argv[2]);
	wB = atoi(argv[3]);

	// Init A and B, malloc C
	int size_A = wA * hA;
	A = (float*)malloc(size_A*sizeof(float));
	init_matrix(A, hA, wA, 1.0);

	int size_B = wB * hB;
	B = (float*)malloc(size_B*sizeof(float));
	init_matrix(B, hB, wB, 2.0);

	int size_C = wB * hA;
	C = (float*)malloc(size_C*sizeof(float));
	for (i = 0; i < (hA*wB); i++) {
		C[i] = 0.0;
	}

	cudaMalloc((void**) &A_G, sizeof (float)*hA*wA);
	cudaMalloc((void**) &B_G, sizeof (float)*hB*wB);
	cudaMalloc((void**) &C_G, sizeof (float)*hA*wB);

	cudaMemcpy(A_G,A,sizeof (float)*hA*wA,cudaMemcpyHostToDevice);
	cudaMemcpy(B_G,B,sizeof (float)*hB*wB,cudaMemcpyHostToDevice);
	cudaMemcpy(C_G,C,sizeof (float)*hA*wB,cudaMemcpyHostToDevice);
	
	dim3 nThreads_per_block(32,32);
	dim3 nBlocks;
	if (N%32==0)
		nBlocks = dim3((hA*wB)/32,(hA*wB)/32);
	else
		nBlocks = dim3((hA*wB)/32 +1,(hA*wB)/32 +1);
	
	<<<nBlocks,nThreads_per_block>>>MulGpu(A_G, B_G,  C_G, hA, wA, wB);
	Mul(A, B, hA, wA, wB, C);

	cudaMemcpy(C,C_G,sizeof (float)*hA*wB,cudaMemcpyDeviceToHost);	
	
	//printf("\n\nMATRIX A\n");print_matrix(A, hA, wA);
	//printf("\n\nMATRIX B\n");print_matrix(B, hB, wB);
	//printf("\n\nMATRIX C\n");print_matrix(C, hA, wB);

	if (!diff(A, B, hA, wA, wB, C))
		printf("ERROR=GPU.vs.CPU matrix mult differs\n");
	

	// print Matrix
	//printf("\n\nMATRIX A\n");print_matrix(A, hA, wA);
	//printf("\n\nMATRIX B\n");print_matrix(B, hB, wB);
	//printf("\n\nMATRIX C\n");print_matrix(C, hA, wB);

	return (1);
}


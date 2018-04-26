// transpose_kernel.cl
// Kernel source file for calculating the transpose of a matrix
#ifndef DIM
#define DIM 16
#endif

__kernel
void matrixTranspose(__global float * input,
                     __global float * output,
                     const    uint    width)

{
  int i = get_global_id(0);
  int j = get_global_id(1);

  if(i < width && j < width) output[j+i*width] = input[i+j*width];
}


__kernel
void matrixTransposeLocal(__global float * output,
                          __global float * input,
                          const uint  width,
			  __local float * matAux)

{
  int i = get_global_id(0); int j = get_global_id(1);
  int bx = get_group_id(0); int by = get_group_id(1);
  int tx = get_local_id(0); int ty = get_local_id(1); //Thread index

  int x = bx * DIM + tx;
  int y = by * DIM + ty;


  matAux[ty*DIM+tx] = input[y * width + x];



  barrier(CLK_LOCAL_MEM_FENCE);

  x = by * DIM + tx;
  y = bx * DIM + ty;

  output[y * width + x] = matAux[tx*DIM+ty];

  barrier(CLK_GLOBAL_MEM_FENCE);
}

/*
1º inicializar
   colas
   kernel
   create buffer
  
calc mandel open_Cl
   tex ->"host device"
   

*/

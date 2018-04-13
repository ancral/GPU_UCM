// transpose_kernel.cl
// Kernel source file for calculating the transpose of a matrix

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
                          //...,
                          const    uint    width)

{

}

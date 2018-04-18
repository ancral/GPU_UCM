__kernel void vadd(__global float* d_a, __global float* d_b, __global float*d_c, const unsigned int global){
	int i = get_global_id(0);
	if(i < global )
		d_c[i] = d_a[i] + d_b[i];
}
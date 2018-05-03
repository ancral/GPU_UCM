#include <stdio.h>
#include <stdlib.h>
#include "my_ocl.h"
#include "CL/cl.h"
#define DIMBLOCK 16

//pick up device type from compiler command line or from 
//the default type
#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_DEFAULT
#endif

extern char *err_code (cl_int err_in);
extern int output_device_info(cl_device_id device_id);
extern float *getmemory1D( int nx );

/*double get_time(){
	static struct timeval 	tv0;
	double time_, time;

	gettimeofday(&tv0,(struct timezone*)0);
	time_=(double)((tv0.tv_usec + (tv0.tv_sec)*1000000));
	time = time_/1000000;
	return(time);
}*/


double calc_piOCL(int n)
{
	//variables del kernel
  cl_mem areas;
  float * arfinal;
  double pi;
  int j;


  cl_int err;
  FILE *fp;
  long filelen;
  long readlen;
  char *kernel_src;
  cl_uint num_devs_returned;
  cl_context_properties properties[3];
  cl_device_id device_id;
  cl_platform_id platform_id;
  cl_uint num_platforms_returned;
  cl_context context;
  cl_command_queue command_queue;
  cl_program program;
  cl_kernel kernel;
  size_t global;
  //size_t local;

  arfinal = getmemory1D(n);

  fp = fopen("kernel.cl","r");
  fseek(fp,0L, SEEK_END);
  filelen = ftell(fp);
  rewind(fp);
  kernel_src = malloc(sizeof(char)*(filelen+1));
  readlen = fread(kernel_src,1,filelen,fp);
  if(readlen!= filelen){
    printf("error reading file\n");
    exit(1);
  }
	
  // ensure the string is NULL terminated
  kernel_src[filelen]='\0';
  cl_uint numPlatforms;

  // Find number of platforms
  err = clGetPlatformIDs(0, NULL, &numPlatforms);
  if (err != CL_SUCCESS || numPlatforms <= 0){
      printf("Error: Failed to find a platform!\n%s\n",err_code(err));
      return;
  }
 
  // Get all platforms
  cl_platform_id Platform[numPlatforms];
  err = clGetPlatformIDs(numPlatforms, Platform, NULL);
  if (err != CL_SUCCESS || numPlatforms <= 0){
      printf("Error: Failed to get the platform!\n%s\n",err_code(err));
      return;
   }

  // Secure a GPU
  int i;
  for (i = 0; i < numPlatforms; i++){
      err = clGetDeviceIDs(Platform[i], DEVICE, 1, &device_id, NULL);
      if (err == CL_SUCCESS){
	  break;
	}
    }

  if (device_id == NULL)
    {
      printf("Error: Failed to create a device group!\n%s\n",err_code(err));
      return;
    }

  err = output_device_info(device_id);

  // Create a compute context 
  context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
  if (!context)
    {
      printf("Error: Failed to create a compute context!\n%s\n", err_code(err));
      return;
    }
  // create command queue 
  command_queue = clCreateCommandQueue(context,device_id, 0, &err);
  if (err != CL_SUCCESS)
    {	
      printf("Unable to create command queue. Error Code=%d\n",err);
      exit(1);
    }
	 
  // create program object from source. 
  // kernel_src contains source read from file earlier
  program = clCreateProgramWithSource(context, 1 ,(const char **)
				      &kernel_src, NULL, &err);
  if (err != CL_SUCCESS)
    {	
      printf("Unable to create program object. Error Code=%d\n",err);
      exit(1);
    }
  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (err != CL_SUCCESS)
    {
      printf("Build failed. Error Code=%d\n", err);
	
      size_t len;
      char buffer[2048];
      // get the build log
      clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,
			    sizeof(buffer), buffer, &len);
      printf("--- Build Log -- \n %s\n",buffer);
      exit(1);
    }


  kernel = clCreateKernel(program, "calcAreas", &err);
  if (err != CL_SUCCESS)
    {	
      printf("Unable to create kernel object. Error Code=%d\n",err);
      exit(1);
    }
  //Creamos los buffers
  areas = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * n, NULL, NULL);

  /*err = clEnqueueWriteBuffer(command_queue, original, CL_TRUE, 0, sizeof(float) * width * height,im,0,NULL,NULL);

  if (err != CL_SUCCESS)
    {
      printf("Error: Failed to write h_b to source array!\n%s\n", err_code(err));
      exit(1);
    }*/


  // set the kernel arguments
  if ( clSetKernelArg(kernel, 0, sizeof(cl_mem), &areas) ||
       clSetKernelArg(kernel, 1, sizeof(cl_uint), &n)  != CL_SUCCESS)
    {
      printf("Unable to set kernel arguments. Error Code=%d\n",err);
      exit(1);
    }
  global = n;

  //local = DIMBLOCK;
  double t0 = getMicroSeconds();
  err = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, 
			       &global, NULL, 0, NULL, NULL);


  if (err != CL_SUCCESS)
    {	
      printf("Unable to enqueue kernel command. Error Code=%d\n",err);
      exit(1);
    }

  // wait for the command to finish
  clFinish(command_queue);

  // read the output back to host memory
  err = clEnqueueReadBuffer(command_queue, areas, CL_TRUE, 0, sizeof(float)*n,arfinal, 0, NULL,NULL);
  if (err != CL_SUCCESS)
    {	
      printf("Error enqueuing read buffer command. Error Code=%d\n",err);
      exit(1);
    }
  else{
	for (j = 1; j < n ; j++){
		//printf("%f",arfinal[j]); 
		pi += arfinal[j];
	}
   }
  pi = pi / n;
  double t1 = getMicroSeconds();
    printf("\nThe kernel ran in %lf seconds\n",(t1-t0)/1000000);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(command_queue);
  clReleaseContext(context);
  free(kernel_src);
  return (pi);
}

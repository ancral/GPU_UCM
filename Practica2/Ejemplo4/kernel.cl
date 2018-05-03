

void buble_sort(float array[], int size)
{
	int i, j;
	float tmp;

	for (i=1; i<size; i++)
		for (j=0 ; j<size - i; j++)
			if (array[j] > array[j+1]){
				tmp = array[j];
				array[j] = array[j+1];
				array[j+1] = tmp;
			}
}



__kernel void cleanNoise(__global float * original, __global float * resultado, 
	const float thredshold, const uint window_size,
	const uint width, const uint height){

	const int MAX_WINDOW_SIZE = 5*5;
	int ii, jj;
	int i = get_global_id(0); int j = get_global_id(1); //Global
	int bx = get_group_id(0); int by = get_group_id(1); //Bloques
  	int tx = get_local_id(0); int ty = get_local_id(1); //Hilos
	float meanaux;
	if(i > 1 && j > 1 && i < width - 2 && j < height - 2){	
		float window[MAX_WINDOW_SIZE];
		float median;
		int ws2 = (window_size-1)>>1; 
	
		for (ii =-ws2; ii<=ws2; ii++) 
			for (jj =-ws2; jj<=ws2; jj++)
				window[(ii+ws2)*window_size + jj+ws2] = original[(i+ii)*width + j+jj];

		// SORT
		buble_sort(window, window_size*window_size);
		median = window[(window_size*window_size-1)>>1];
		meanaux = (median-original[i*width+j])/median;
		if(meanaux < 0) meanaux = -meanaux;
		printf("%f==%i \n",meanaux, thredshold);
		if (meanaux <=thredshold){
			resultado[i*width + j] = original[i*width+j];
			//printf("limpio");
		}
		else{
			resultado[i*width + j] = median;
			printf("sucio");		
		}
		//printf("%f",resultado[i*width+j]);
	}
				
		

}



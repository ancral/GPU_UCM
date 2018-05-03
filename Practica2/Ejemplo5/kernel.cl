__kernel void calcAreas(__global float * areas, const uint n){
	float x;
	int i = get_global_id(0);
	//printf("%i \n",i);
	x = (i+0.5)/n;
	areas[i]= 4.0/(1.0 + x*x); 
	
	//barrera
}

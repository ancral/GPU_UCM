#define DIMBLOCK 16

__kernel void calcAreas(__global float * areas, 
		//__local float * reductions, 
		const uint n){
	float x;
	
	
	int i = get_global_id(0),j;
	for(j = 0; j < DIMBLOCK; j++)
		if(j + i * DIMBLOCK < n)
			areas[i] += 4.0/(1.0 + ((j + i * DIMBLOCK+0.5)/n)*((j + i * DIMBLOCK+0.5)/n));
	
	//printf("%i \n",i);
	//x = (i+0.5)/n;
	//areas[i]= 4.0/(1.0 + x*x); 
	
}


#define max_iter 256

__kernel void mandel(__global rgb_t ** texcl, const uint width, const double scale, const double cx, const double cy, const uint height){
	/*int i = get_global_id(0), j = get_global_id(1);
	int iter, min, max;
	rgb_t *pixel;
	double x, y, zx, zy, zx2, zy2;
	min = max_iter; max = 0;
	if( i < height ) {
		pixel = texcl[i] + j;
		y = (i - height/2) * scale + cy;
		if( j  < width ){
			x = (j - width/2) * scale + cx;

			zx = zy = zx2 = zy2 = 0;
			for (iter=0; iter < max_iter; iter++) {
				zy=2*zx*zy + y;
				zx=zx2-zy2 + x;
				zx2=zx*zx;
				zy2=zy*zy;
				if (zx2+zy2>max_iter)
					break;
			}
			if (iter < min) min = iter;
			if (iter > max) max = iter;
			*(unsigned short *)pixel = iter;
			//pixel++;
		}
	}
 
	//barrera

	if( i < height ){
			pixel = tex[i] + j;
		if( j  < width ){

			hsv_to_rgb(*(unsigned short*)pixel, min, max, pixel);
			//pixel++;
		}
	}*/
	int i = get_group_id(0);
	int j = get_group_id(1);
	int iter = get_local_id(0);
	rgb_t *pixel;
	double x, y, zx, zy, zx2, zy2;
	//double t0;

	//t0 = getMicroSeconds();
	min = max_iter; max = 0;
	//for (i = 0; i < height; i++) {
		pixel = tex[i];
		y = (i - height/2) * scale + cy;


		//for (j = 0; j  < width; j++, pixel++) {
			x = (j - width/2) * scale + cx;
			zx = zy = zx2 = zy2 = 0;
			pixel += j;

			//for (iter=0; iter < max_iter; iter++) {
				zy=2*zx*zy + y;
				zx=zx2-zy2 + x;
				zx2=zx*zx;
				zy2=zy*zy;
			//	if (zx2+zy2>max_iter)
			//		break;
			//}

			if (iter == max_iter /*|| break_condition*/){
				if (iter < min) min = iter;
				if (iter > max) max = iter;}
			*(unsigned short *)pixel = iter;
		//}
	//}
 	/*	Realizar particiones en este momento?
 		min & mx tienen que ser de salida?
 		Si min  y/o max tienen que ser de salida, entonces este bucle tiene que estar fuera del kernel, y por tanto 
 		habrá que realizar particiones/memorialocal + barreras*/
	for (i = 0; i < height; i++)
		for (j = 0, pixel = tex[i]; j  < width; j++, pixel++)
			hsv_to_rgb(*(unsigned short*)pixel, min, max, pixel);

}


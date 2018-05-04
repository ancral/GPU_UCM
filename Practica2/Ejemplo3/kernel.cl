
#define max_iter 256

__kernel void mandel(__global rgb_t ** texcl, const uint width, const uint height){
	int i = get_global_id(0), j = get_global_id(1);
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
	}


}


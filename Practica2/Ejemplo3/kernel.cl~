 

__kernel void mandel(){
	t0 = getMicroSeconds();
	min = max_iter; max = 0;
	for (i = 0; i < height; i++) {
		pixel = tex[i];
		y = (i - height/2) * scale + cy;
		for (j = 0; j  < width; j++, pixel++) {
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
		}
	}
 
	for (i = 0; i < height; i++)
		for (j = 0, pixel = tex[i]; j  < width; j++, pixel++)
			hsv_to_rgb(*(unsigned short*)pixel, min, max, pixel);
}


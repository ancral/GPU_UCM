
#define max_iter 256
typedef struct {unsigned char r, g, b;} rgb_t;

void hsv_to_rgb(int hue, int min, int max, __global unsigned char *p)
{

	int color_rotate = 0;
	int saturation = 1;
	int invert = 0;
	if (min == max) max = min + 1;
	if (invert) hue = max - (hue - min);
	if (!saturation) {
		p[0] = p[1] = p[2] = 255 * (max - hue) / (max - min);
		return;
	}
	double h = fmod(color_rotate + 1e-4 + 4.0 * (hue - min) / (max - min), 6);
#	define VAL 255
	double c = VAL * saturation;
	double X = c * (1 - fabs(fmod(h, 2) - 1));
 
	p[0] = p[1] = p[2] = 0;
 
	switch((int)h) {
	case 0: p[0] = c; p[1] = X; return;
	case 1: p[0] = X; p[1] = c; return;
	case 2: p[1]= c; p[2] = X; return;
	case 3: p[1] = X; p[2] = c; return;
	case 4: p[0] = X; p[2] = c; return;
	default:p[0] = c; p[2] = X;
	}
}

__kernel void mandel(__global unsigned char  * texcl, const uint width, const double scale, const double cx, const double cy, const uint height){
	
	int i = get_group_id(0);
	int j = get_group_id(1);
	int iter = get_local_id(0),min,max;
	__global unsigned char *pixel;
	double x, y, zx, zy, zx2, zy2;
	
	min = max_iter; max = 0;
	pixel = &texcl[i];
	y = (i - height/2) * scale + cy;
	x = (j - width/2) * scale + cx;
	zx = zy = zx2 = zy2 = 0;
	pixel += j;
	
	zy=2*zx*zy + y;
	zx=zx2-zy2 + x;
	zx2=zx*zx;
	zy2=zy*zy;

	if (iter == max_iter){
		if (iter < min) min = iter;
		if (iter > max) max = iter;}
	*pixel = iter;
	barrier(CLK_GLOBAL_MEM_FENCE);
	pixel = &texcl[i];
	pixel += j;
			hsv_to_rgb(*pixel, min, max, pixel);

}


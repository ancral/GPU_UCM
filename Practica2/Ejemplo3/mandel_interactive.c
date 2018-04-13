#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <GL/glut.h>
#include <GL/gl.h>
#include <GL/glu.h>

#define RUN_SERIAL     0
#define RUN_OPENCL_CPU 1
#define RUN_OPENCL_GPU 2
int run_mode;
 
void set_texture();
 
typedef struct {unsigned char r, g, b;} rgb_t;
rgb_t **tex = 0;
int gwin;
GLuint texture;
int width, height;
int tex_w, tex_h;
double scale = 1./256;
double cx = -.6, cy = 0;
int color_rotate = 0;
int saturation = 1;
int invert = 0;
int max_iter = 256;
 
/* Time */
#include <sys/time.h>
#include <sys/resource.h>

static struct timeval tv0;
double getMicroSeconds()
{
	double t;
	gettimeofday(&tv0, (struct timezone*)0);
	t = ((tv0.tv_usec) + (tv0.tv_sec)*1000000);

	return (t);
}

void render()
{
	double	x = (double)width /tex_w,
			y = (double)height/tex_h;
 
	glClear(GL_COLOR_BUFFER_BIT);
	glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
 
	glBindTexture(GL_TEXTURE_2D, texture);
 
	glBegin(GL_QUADS);
 
	glTexCoord2f(0, 0); glVertex2i(0, 0);
	glTexCoord2f(x, 0); glVertex2i(width, 0);
	glTexCoord2f(x, y); glVertex2i(width, height);
	glTexCoord2f(0, y); glVertex2i(0, height);
 
	glEnd();
 
	glFlush();
	glFinish();
}
 
int shots = 1;
void screen_shot()
{
	char fn[100];
	int i;
	sprintf(fn, "screen%03d.ppm", shots++);
	FILE *fp = fopen(fn, "w");
	fprintf(fp, "P6\n%d %d\n255\n", width, height);
	for (i = height - 1; i >= 0; i--)
		fwrite(tex[i], 1, width * 3, fp);
	fclose(fp);
	printf("%s written\n", fn);
}
 
void keypress(unsigned char key, int x, int y)
{
	switch(key) {
	case 'q':	glFinish();
			glutDestroyWindow(gwin);
			return;
	case 27:	scale = 1./256; cx = -.6; cy = 0; break;
 
	case 'r':	color_rotate = (color_rotate + 1) % 6;
			break;
 
	case '>': case '.':
			max_iter += 64;
			if (max_iter > 1 << 15) max_iter = 1 << 15;
			printf("max iter: %d\n", max_iter);
			break;
 
	case '<': case ',':
			max_iter -= 64;
			if (max_iter < 64) max_iter = 64;
			printf("max iter: %d\n", max_iter);
			break;
 
	case 'm':	saturation = 1 - saturation;
			break;
 
	case 'i':	screen_shot(); return;
	case 'z':	max_iter = 4096; break;
	case 'x':	max_iter = 128; break;
	case 's':	run_mode = RUN_SERIAL; break;
	case 'c':	run_mode = RUN_OPENCL_CPU; break;
	case 'g':	run_mode = RUN_OPENCL_GPU; break;
	case ' ':	invert = !invert;
	}
	set_texture();
}
 
void hsv_to_rgb(int hue, int min, int max, rgb_t *p)
{
	if (min == max) max = min + 1;
	if (invert) hue = max - (hue - min);
	if (!saturation) {
		p->r = p->g = p->b = 255 * (max - hue) / (max - min);
		return;
	}
	double h = fmod(color_rotate + 1e-4 + 4.0 * (hue - min) / (max - min), 6);
#	define VAL 255
	double c = VAL * saturation;
	double X = c * (1 - fabs(fmod(h, 2) - 1));
 
	p->r = p->g = p->b = 0;
 
	switch((int)h) {
	case 0: p->r = c; p->g = X; return;
	case 1: p->r = X; p->g = c; return;
	case 2: p->g = c; p->b = X; return;
	case 3: p->g = X; p->b = c; return;
	case 4: p->r = X; p->b = c; return;
	default:p->r = c; p->b = X;
	}
}

double calc_mandel_opencl()
{
	/*
	 * TODO
	 */
	return(1.0);
}
 
double calc_mandel()
{
	int i, j, iter, min, max;
	rgb_t *pixel;
	double x, y, zx, zy, zx2, zy2;
	double t0;

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

	return(getMicroSeconds()-t0);
}
 
void alloc_tex()
{
	int i, ow = tex_w, oh = tex_h;
 
	for (tex_w = 1; tex_w < width;  tex_w <<= 1);
	for (tex_h = 1; tex_h < height; tex_h <<= 1);
 
	if (tex_h != oh || tex_w != ow)
		tex = realloc(tex, tex_h * tex_w * 3 + tex_h * sizeof(rgb_t*));
 
	for (tex[0] = (rgb_t *)(tex + tex_h), i = 1; i < tex_h; i++)
		tex[i] = tex[i - 1] + tex_w;
}
 
void set_texture()
{
	double t;
	char title[128];

	alloc_tex();
	switch (run_mode){
		case RUN_SERIAL:	   t=calc_mandel(); break;
		case RUN_OPENCL_CPU: t=calc_mandel_opencl(); break;
		case RUN_OPENCL_GPU: t=calc_mandel_opencl();
	};

 
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, texture);
	glTexImage2D(GL_TEXTURE_2D, 0, 3, tex_w, tex_h,
		0, GL_RGB, GL_UNSIGNED_BYTE, tex[0]);
 
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	render();

	sprintf(title, "Mandelbrot: %5.2f fps (%ix%i)", 1000000/t, width, height);
	glutSetWindowTitle(title);
}
 
void mouseclick(int button, int state, int x, int y)
{
	if (state != GLUT_UP) return;
 
	cx += (x - width / 2) * scale;
	cy -= (y - height/ 2) * scale;
 
	switch(button) {
	case GLUT_LEFT_BUTTON: /* zoom in */
		if (scale > fabs(x) * 1e-16 && scale > fabs(y) * 1e-16)
			scale /= 2;
		break;
	case GLUT_RIGHT_BUTTON: /* zoom out */
		scale *= 2;
		break;
	/* any other button recenters */
	}
	set_texture();
}
 
 
void resize(int w, int h)
{
	//printf("resize %d %d\n", w, h);
	width = w;
	height = h;
 
	glViewport(0, 0, w, h);
	glOrtho(0, w, 0, h, -1, 1);
 
	set_texture();
}
 
void init_gfx(int *c, char **v)
{
	glutInit(c, v);
	glutInitDisplayMode(GLUT_RGB);
	glutInitWindowSize(640, 480);
	glutDisplayFunc(render);
 
	gwin = glutCreateWindow("Mandelbrot");
 
	glutKeyboardFunc(keypress);
	glutMouseFunc(mouseclick);
	glutReshapeFunc(resize);
	glGenTextures(1, &texture);
	set_texture();
}
 
int main(int c, char **v)
{
	init_gfx(&c, v);
	printf("keys:\n\tr: color rotation\n\tm: monochrome\n\ti: screen shot\n\t"
            "s: serial code\n\tc: OpenCL CPU\n\tg: OpenCL GPU\n\t"
		"<, >: decrease/increase max iteration\n\tq: quit\n\tmouse buttons to zoom\n");
 
	glutMainLoop();
	return 0;
}

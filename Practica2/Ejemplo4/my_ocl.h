#ifndef _OCL_H

#define _OCL_H
/* From common.c */

void remove_noiseOCL(float *im, float *image_out, 
	float thredshold, int window_size,
	int height, int width);

#endif

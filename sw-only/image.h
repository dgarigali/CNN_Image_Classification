#define SCALE_COEF 0.00390625

#define NIMAGES 100
#define IMAGE_HEIGTH 28
#define IMAGE_WIDTH 28

void print_pgm(unsigned char *cimgs, int im);
void image_scale2float(unsigned char *cimgs, int im, float *fim);
void print_fp_image(float *fim);

/* Reads mnist image file (see http://yann.lecun.com/exdb/mnist/) 
   and prints image N in pgm format (see http://paulbourke.net/dataformats/ppm/)
   to stdout.
*/

#include <stdio.h>
#include <stdlib.h>
#include "image.h"

void print_pgm(unsigned char *cimgs, int im){

  int i,j = 0;
  unsigned char *pim;

  printf("P2\n28 28 255\n");

  pim = cimgs + 16 + (im-1)*(IMAGE_HEIGTH*IMAGE_WIDTH);

  for (i=0; i < IMAGE_HEIGTH; i++) {
    for (j=0; j < IMAGE_WIDTH; j++) {
      printf("%3d ", pim[i*IMAGE_WIDTH+j]);
    }
    printf("\n");
  }
}

/* Scales image pixels to be floating-point values in range [0,1[ */
void image_scale2float(unsigned char *cimgs, int image_to_classify, float *fim)
{
  int i = 0;
  unsigned char *pim;

  pim = cimgs + 16 + (image_to_classify-1)*(IMAGE_HEIGTH*IMAGE_WIDTH);
  for(i = 0; i < IMAGE_HEIGTH*IMAGE_WIDTH; i++){
    fim[i] = (float)pim[i] * SCALE_COEF;
  }
}

void print_fp_image(float *fim){

  int i,j = 0;

  for (i=0; i < IMAGE_HEIGTH; i++) {
    for (j=0; j < IMAGE_WIDTH; j++) {
      printf("%f ", fim[i*IMAGE_WIDTH+j]);
    }
    printf("\n");
  }
}


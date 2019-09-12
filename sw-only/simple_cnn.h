#define EMBEDDED 1

#define TOTAL_WEIGTHS (22+22*5*5+10+10*22*12*12)
#define FLT_MAX 3.402823466e+38F

#if EMBEDDED == 1
#include "xtime_l.h"

// DDR pre-defined data regions
#define MEM_DATA_BASE_ADDRESS 0x12000000
#define MEM_IMAGES_BASE_ADDRESS 0x10000000
#define MEM_WEIGTHS_BASE_ADDRESS 0x11000000

#else
#include <time.h>
#define COUNTS_PER_SECOND CLOCKS_PER_SEC

static unsigned char memory_images[NIMAGES*IMAGE_HEIGTH*IMAGE_WIDTH+16];
static float memory_weights[TOTAL_WEIGTHS];
static float memory_data[75000];

#define MEM_DATA_BASE_ADDRESS memory_data
#define MEM_IMAGES_BASE_ADDRESS memory_images
#define MEM_WEIGTHS_BASE_ADDRESS memory_weights
#endif

#define PRINT_IMAGE 1
#define PRINT_TIME_PER_LAYER 1
#define IMAGE_TO_CLASSIFY 1
#define NUMBER_OF_IMAGES_TO_CLASSIFY 1

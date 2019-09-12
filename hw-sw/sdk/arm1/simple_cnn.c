#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "simple_cnn.h"
#include "xil_mmu.h"

volatile int *sync_f = (int *)0xFFFF0000;
#define PROC0_STARTED 11
#define PROC1_STARTED 22
#define PROC1_COMPLETED 33

volatile float *fp_weights; // Network weights data region
volatile float *fp_image; // Scaled floating-point image to be processed

volatile float *matCpool; // Output of pooling layer 22 images of size 12*12
volatile float *matConn;  // Intermediate output (before adding bias) of fully connected layer (10 elements)
volatile float *matConnB; // Output of fully connected layer (10 elements)

// Matrix multiplication: C = A * B
void gemm(float *A, float *B, float *C, int rowsA, int colsA, int colsB) {
  int i, j, k;

  for (i=0; i<rowsA; i++) {
    for (j=0; j<colsB; j++) {
      C[i*colsB+j] = 0.0;
      for (k=0; k<colsA; k++) {
	C[i*colsB+j] += A[i*colsA+k] * B[k*colsB+j];
      }
    }
  }
}

void add_bias(float *C, int rows, int cols, float *bias, float *Cbias, int transpose_flag) {
  int i, j;

  if (transpose_flag) {
    for (i=0; i<rows; i++) {
      for (j=0; j<cols; j++) {
        Cbias[j*rows+i] = C[i*cols+j] + bias[j] ;
      }
    }
  }
  else {
    for (i=0; i<rows; i++) {
      for (j=0; j<cols; j++) {
	    Cbias[i*cols+j] = C[i*cols+j] + bias[i] ;
      }
    }
  }
}

void forward_connected_layer() {
    float *matW, *matIN, *mbias, *matOUT, *matOutB;

    // The 10 bias values of this layer are stored after the 22+550 convolutional bias+weigths
    mbias = (float *)fp_weights + 22 + 550 + 5;

    // The 10*2880 weights are stored after the 10 bias values
    matW = (float *)fp_weights + 22 + 550 + 10 + 5 * 3168;
    
    matIN = (float *)matCpool;
    matOUT = (float *)matConn + 5;
    matOutB = (float *)matConnB + 5;
    
    // A(10*3168) * B(3168*1) -> C(10*1)
    gemm(matW, matIN, matOUT, 5, 3168, 1);
    add_bias(matOUT, 5, 1, mbias, (float *)matOutB, 0);
}

void define_memory_regions() {
  static float *paddress = (float *)MEM_DATA_BASE_ADDRESS;

  // Region Size TOTAL_WEIGTHS*sizeof(float) = 29330*4 = 117320 Bytes
  fp_weights = (volatile float *)MEM_WEIGTHS_BASE_ADDRESS; 
   
  // Region Size IMAGE_HEIGTH*IMAGE_WIDTH*sizeof(float) = 28*28*4 = 3136 Bytes
  fp_image = paddress;
  paddress += 28*28 + TOTAL_WEIGTHS;
 
  // Aux matrix of (22)*(12*12) elements. Region Size = 3168 * 4 Bytes
  matCpool = paddress;
  paddress += (22)*(12*12);

  // Aux matrix of 10 elements. Region Size = 10 * 4 Bytes;
  matConn = paddress;
  paddress += 10;

  // Aux matrix of 10 elements. Region Size = 10 * 4 Bytes
  matConnB = paddress;
  paddress += 10;
}

int main(int argc, char **argv) {

	// Disable cache on OCM region
	Xil_SetTlbAttributes(0xFFFF0000,0x14de2);

	unsigned int image_to_classify = IMAGE_TO_CLASSIFY; //default

	define_memory_regions();

	for (image_to_classify = IMAGE_TO_CLASSIFY; image_to_classify < (IMAGE_TO_CLASSIFY+NUMBER_OF_IMAGES_TO_CLASSIFY); image_to_classify++) {

		while (*sync_f != PROC0_STARTED); // Wait for P0
		*sync_f = PROC1_STARTED;
		forward_connected_layer();
		*sync_f = PROC1_COMPLETED;
	}

	return 0;

}

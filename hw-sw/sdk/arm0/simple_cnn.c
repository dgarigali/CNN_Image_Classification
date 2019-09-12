#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "image.h"
#include "simple_cnn.h"

#include "xaxidma.h"
#include "xparameters.h"
#include "xtime_l.h"
#include "xil_cache.h"
#include "xil_mmu.h"

//DMA
#define DMA_DEV_ID XPAR_AXIDMA_0_DEVICE_ID
XAxiDma AxiDma;
float *TxBufferPtr, *RxBufferPtr;

volatile int *sync_f = (int *)0xFFFF0000;
#define PROC0_STARTED 11
#define PROC1_STARTED 22
#define PROC1_COMPLETED 33

volatile unsigned char *ch_images;  // Images data region
volatile float *fp_weights; // Network weights data region

volatile float *fp_image; // Scaled floating-point image to be processed
volatile float *kernel; // Scaled floating-point image to be processed

volatile float *matCpool; // Output of pooling layer 22 images of size 12*12
volatile float *matConn;  // Intermediate output (before adding bias) of fully connected layer (10 elements)
volatile float *matConnB; // Output of fully connected layer (10 elements)
volatile float *matSoftM; // Output of softmax layer (10 elements)

volatile int flag = 1;
volatile double total_time = 0;

void print_fp(float *f, int size, char *c) {
  int i;
  printf("%s\n", c);
  for (i = 0; i < size; i++) {
    if ((i % 12) == 0) printf("%02d: ", i/12);
    printf("%f ", f[i]);
    if ((i % 12) == 11) printf("\n");
  }
  printf("\n");
}

int forward_softmax_layer() {

  while (*sync_f != PROC1_COMPLETED); // Wait for P1

  float * in = (float *) matConnB;
  float * out = (float *) matSoftM;

  int i, n=10, best=-1;
  float sum = 0.0, e;
  float largest = -FLT_MAX;

  for(i = 0; i < n; ++i){
    if(in[i] > largest) {
      largest = in[i];
      best = i;
    }
  }
	
  for(i = 0; i < n; ++i){
    e = exp(in[i] - largest);
    sum += e;
    out[i] = e;
  }
  for(i = 0; i < n; ++i){
	  out[i] /= sum;
  }
  //print_fp((float *)matSoftM, 10, "Softmax");

  return best;
}

void forward_maxpool_layer() {
  //print_fp((float *)matCpool, 144, "Pool");
  // Output matrix Cpool is 22*144, that is this layer outputs 22 12*12 images.
}

void forward_convolutional_layer() {
    
	//Only send kernel for first image
	if (flag == 1) {

		//Send authorization to read kernel (22 + 22x25)
		TxBufferPtr = (float *)kernel;
		if (XAxiDma_SimpleTransfer(&AxiDma,(UINTPTR) TxBufferPtr, (22 + 25*22)*4, XAXIDMA_DMA_TO_DEVICE) != XST_SUCCESS) {
			printf("Error sending kernel\n");
		}
		while (XAxiDma_Busy(&AxiDma, XAXIDMA_DMA_TO_DEVICE)) { /* Wait for Tx*/ }
		flag = 0;
	}

	//Send authorization to read image
	Xil_DCacheFlushRange((INTPTR)(fp_image), (unsigned)(28*28*4));
	TxBufferPtr = (float *)fp_image;
	if (XAxiDma_SimpleTransfer(&AxiDma,(UINTPTR) TxBufferPtr, 28*28*4, XAXIDMA_DMA_TO_DEVICE) != XST_SUCCESS) {
		printf("Error sending image\n");
	}

	//Send authorization to write 22x12x12
	RxBufferPtr = (float *)matCpool;
	if (XAxiDma_SimpleTransfer(&AxiDma,(UINTPTR) (RxBufferPtr), 12*12*22*4, XAXIDMA_DEVICE_TO_DMA) != XST_SUCCESS) {
		printf("Error receiving layer 2 output\n");
	}

	//Wait for results
	while (XAxiDma_Busy(&AxiDma,XAXIDMA_DMA_TO_DEVICE)) { /* Wait Tx */ }
	while (XAxiDma_Busy(&AxiDma,XAXIDMA_DEVICE_TO_DMA)) { /* Wait Rx*/ }

    // Invalidate Cache Range to force reading matCpool from external memory
	Xil_DCacheInvalidateRange((INTPTR)(matCpool), (unsigned)(12*12*22*4));
}

void forward_connected_layer() {

	*sync_f = PROC0_STARTED;
	while (*sync_f != PROC1_STARTED);

	float *matW, *matIN, *mbias, *matOUT, *matOutB;

    // The 10 bias values of this layer are stored after the 22+550 convolutional bias+weigths
    mbias = (float *)fp_weights + 22 + 550;

    // The 10*2880 weights are stored after the 10 bias values
    matW = (float *)fp_weights + 22 + 550 + 10;
    
    matIN = (float *)matCpool;
    matOUT = (float *)matConn;
    matOutB = (float *)matConnB;
    
    int i, k;
    for (i=0; i<5; i++) {
		matOUT[i] = 0.0;
		for (k=0; k<3168; k++) {
			matOUT[i] += matW[i*3168+k] * matIN[k];
		}
		matOutB[i] = matOUT[i] + mbias[i];
    }
}

int predict_mnist() {
	int best;
	double *ptime, *measure_time();

	measure_time(0);
	forward_convolutional_layer();
	measure_time(1);
	forward_maxpool_layer();
	measure_time(2);
	forward_connected_layer();
	measure_time(3);
	best = forward_softmax_layer();
	ptime = measure_time(4);

	#if PRINT_TIME_PER_LAYER
		printf("Layer 1 + 2: %.0f us, ", ptime[0]);
		printf("Layer 3: %.0f us, ", ptime[2]);
		printf("Layer 4: %.0f us, ", 1.0 * ptime[3]);
		printf("Total: %.0f us\n ", ptime[0] + ptime[2] + ptime[3]);
	#endif

	total_time += ptime[0] + ptime[2] + ptime[3];

  return best;
}

void define_memory_regions() {

  static float *paddress = (float *)MEM_DATA_BASE_ADDRESS;

  // Region Size NIMAGES*IMAGE_HEIGTH*IMAGE_WIDTH+16 = 78416 Bytes (100 images)
  ch_images = (unsigned char *)MEM_IMAGES_BASE_ADDRESS;

  // Region Size TOTAL_WEIGTHS*sizeof(float) = 29330*4 = 117320 Bytes
  fp_weights = (volatile float *)MEM_WEIGTHS_BASE_ADDRESS; 

  //Region size for convolutional kernel
  kernel = paddress;
  paddress += TOTAL_WEIGTHS;
   
  // Region Size IMAGE_HEIGTH*IMAGE_WIDTH*sizeof(float) = 28*28*4 = 3136 Bytes
  fp_image = paddress;
  paddress += 28*28;
 
  // Aux matrix of (22)*(12*12) elements. Region Size = 3168 * 4 Bytes
  matCpool = paddress;
  paddress += (22)*(12*12);

  // Aux matrix of 10 elements. Region Size = 10 * 4 Bytes;
  matConn = paddress;
  paddress += 10;

  // Aux matrix of 10 elements. Region Size = 10 * 4 Bytes
  matConnB = paddress;
  paddress += 10;

  // Aux matrix of 10 elements. Region Size = 10 * 4 Bytes
  matSoftM = paddress;

  // printf("%p, %d\n", (void *)paddress+10, (paddress+10)-(float *)MEM_DATA_BASE_ADDRESS);
  // Total data region size is 71898 * 4 = 287,592 Bytes

}

double *measure_time(int count) {
  static double timetab[5];
  static XTime t[5];
  XTime_GetTime(&(t[count]));

  if (count > 0) {
    timetab[count-1] = ((double) (t[count] - t[count-1])) / (COUNTS_PER_SECOND/1000000);
  }
  return timetab;
}

int init_XAxiDma_SimplePollMode(u16 DeviceId) {
  XAxiDma_Config *CfgPtr;
  int Status;

  /* Initialize the XAxiDma device.	 */
  CfgPtr = XAxiDma_LookupConfig(DeviceId);
  if (!CfgPtr) {
    printf("No config found for %d\r\n", DeviceId);
    return XST_FAILURE;
  }

  Status = XAxiDma_CfgInitialize(&AxiDma, CfgPtr);
  if (Status != XST_SUCCESS) {
    printf("Initialization failed %d\r\n", Status);
    return XST_FAILURE;
  }

  if(XAxiDma_HasSg(&AxiDma)){
    printf("Device configured as SG mode \r\n");
    return XST_FAILURE;
  }

  /* Disable interrupts, we use polling mode	 */
  XAxiDma_IntrDisable(&AxiDma, XAXIDMA_IRQ_ALL_MASK, XAXIDMA_DEVICE_TO_DMA);
  XAxiDma_IntrDisable(&AxiDma, XAXIDMA_IRQ_ALL_MASK, XAXIDMA_DMA_TO_DEVICE);

  return XST_SUCCESS;
}

int main(int argc, char **argv) {

	//Disable cache
	//Xil_DCacheDisable();

	// Disable cache on OCM region
	Xil_SetTlbAttributes(0xFFFF0000,0x14de2);

	//Init DMA in poll mode for simple transfer
	if (init_XAxiDma_SimplePollMode(DMA_DEV_ID) != XST_SUCCESS) {
		printf("XAxiDma_Simple_MatProd: Failed\r\n");
		return XST_FAILURE;
	}

	unsigned int image_to_classify = IMAGE_TO_CLASSIFY; //default
	int prediction;

	define_memory_regions();

	for (image_to_classify = IMAGE_TO_CLASSIFY; image_to_classify < (IMAGE_TO_CLASSIFY+NUMBER_OF_IMAGES_TO_CLASSIFY); image_to_classify++) {

		// The pixels of the input image are scaled to the [0,1[ interval
		image_scale2float((unsigned char *)ch_images, image_to_classify, (float *)fp_image);

		#if PRINT_IMAGE
			print_pgm((unsigned char *)ch_images, image_to_classify);
		#endif

		prediction = predict_mnist();
		printf("Image %d -> Digit %d %f\n\n", image_to_classify, prediction, matSoftM[prediction]);

	}

	printf("Total time: %.0f us\n", total_time);

	return 0;
}

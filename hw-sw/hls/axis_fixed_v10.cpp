#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_stream.h>

typedef ap_fixed<32,16> datai_t;
typedef ap_fixed<64,32> datao_t;
typedef ap_fixed<64,32> mul_t;
typedef ap_fixed<64,32> acc_t;

typedef ap_ufixed<8,0> image_t;
typedef ap_fixed<17,1> kernel_t;

#define IMAGE_WIDTH 28
#define MAX_POOLING 2
#define KERNEL_WIDTH 5
#define RESULT_WIDTH 12
#define NUM_CONVOLUTIONS 22

#define FLT_MAX 65535

struct ap_i_axis{
  datai_t data;
  ap_uint<1> last;
};
struct ap_o_axis{
  datao_t data;
  ap_uint<1> last;
};

// The top-level function
void axis_fixed_macc_v10(hls::stream<ap_o_axis> &strm_out1, hls::stream<ap_o_axis> &strm_out2, hls::stream<ap_i_axis> &strm_in1, hls::stream<ap_i_axis> &strm_in2) {

	#pragma HLS INTERFACE ap_ctrl_none port=return
	#pragma HLS interface axis port=strm_in1
	#pragma HLS interface axis port=strm_in2
	#pragma HLS INTERFACE axis port=strm_out1
	#pragma HLS interface axis port=strm_out2

	struct ap_i_axis tmp1, tmp2;
	struct ap_o_axis tmpa1, tmpa2;
	static image_t image_mem[784];

	static kernel_t kernel_mem1[50], kernel_mem2[50], kernel_mem3[50], kernel_mem4[50], kernel_mem5[50], kernel_mem6[50];
	static kernel_t kernel_mem7[50], kernel_mem8[50], kernel_mem9[50], kernel_mem10[50], kernel_mem11[50];

	static kernel_t bias1, bias2, bias3, bias4, bias5, bias6, bias7, bias8, bias9, bias10, bias11;
	static kernel_t bias12, bias13, bias14, bias15, bias16, bias17, bias18, bias19, bias20, bias21, bias22;

	static acc_t mult1, mult2, mult3, mult4, mult5, mult6, mult7, mult8, mult9, mult10, mult11;
	static acc_t mult12, mult13, mult14, mult15, mult16, mult17, mult18, mult19, mult20, mult21, mult22;
	static acc_t mult23, mult24, mult25, mult26, mult27, mult28, mult29, mult30, mult31, mult32, mult33;
	static acc_t mult34, mult35, mult36, mult37, mult38, mult39, mult40, mult41, mult42, mult43, mult44;

	static acc_t acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10, acc11;
	static acc_t acc12, acc13, acc14, acc15, acc16, acc17, acc18, acc19, acc20, acc21, acc22;
	static acc_t acc23, acc24, acc25, acc26, acc27, acc28, acc29, acc30, acc31, acc32, acc33;
	static acc_t acc34, acc35, acc36, acc37, acc38, acc39, acc40, acc41, acc42, acc43, acc44;

	static acc_t reg1, reg2, reg3, reg4, reg5, reg6, reg7, reg8, reg9, reg10, reg11;
	static acc_t reg12, reg13, reg14, reg15, reg16, reg17, reg18, reg19, reg20, reg21, reg22;
	static acc_t reg23, reg24, reg25, reg26, reg27, reg28, reg29, reg30, reg31, reg32, reg33;
	static acc_t reg34, reg35, reg36, reg37, reg38, reg39, reg40, reg41, reg42, reg43, reg44;

	static datao_t result_mem1[144], result_mem2[144], result_mem3[144], result_mem4[144], result_mem5[144], result_mem6[144];
	static datao_t result_mem7[144], result_mem8[144], result_mem9[144], result_mem10[144], result_mem11[144];
	static datao_t result_mem12[144], result_mem13[144], result_mem14[144],  result_mem15[144],  result_mem16[144];
	static datao_t result_mem17[144], result_mem18[144], result_mem19[144],  result_mem20[144],  result_mem21[144];

	static kernel_t kernel1, kernel2, kernel3, kernel4, kernel5, kernel6, kernel7, kernel8, kernel9, kernel10, kernel11;
	static kernel_t kernel12, kernel13, kernel14, kernel15, kernel16, kernel17, kernel18, kernel19, kernel20, kernel21, kernel22;

	unsigned short i, j;
	unsigned short image_idx1, image_idx2, kernel_idx;
	unsigned short kernel_col_cnt = 0, kernel_line_cnt = 0;
	unsigned short max_pool_col_cnt = 0, max_pool_line_cnt = 0;

	static bool flag2 = true;
	image_t image1, image2;
	int counter;

	//Receive bias
	for (i = 0; i < 11; i++) {
		#pragma HLS pipeline
		tmp1 = strm_in1.read();
		tmp2 = strm_in2.read();
		switch(i) {
			case 0:
				bias1 = (kernel_t) tmp1.data;
				bias2 = (kernel_t) tmp2.data;
				break;
			case 1:
				bias3 = (kernel_t) tmp1.data;
				bias4 = (kernel_t) tmp2.data;
				break;
			case 2:
				bias5 = (kernel_t) tmp1.data;
				bias6 = (kernel_t) tmp2.data;
				break;
			case 3:
				bias7 = (kernel_t) tmp1.data;
				bias8 = (kernel_t) tmp2.data;
				break;
			case 4:
				bias9 = (kernel_t) tmp1.data;
				bias10 = (kernel_t) tmp2.data;
				break;
			case 5:
				bias11 = (kernel_t) tmp1.data;
				bias12 = (kernel_t) tmp2.data;
				break;
			case 6:
				bias13 = (kernel_t) tmp1.data;
				bias14 = (kernel_t) tmp2.data;
				break;
			case 7:
				bias15 = (kernel_t) tmp1.data;
				bias16 = (kernel_t) tmp2.data;
				break;
			case 8:
				bias17 = (kernel_t) tmp1.data;
				bias18 = (kernel_t) tmp2.data;
				break;
			case 9:
				bias19 = (kernel_t) tmp1.data;
				bias20 = (kernel_t) tmp2.data;
				break;
			case 10:
				bias21 = (kernel_t) tmp1.data;
				bias22 = (kernel_t) tmp2.data;
				break;
		}
	}

	//Kernel 1 and 2
	for (i = 0; i < 50; i += 2) {
		#pragma HLS pipeline
		tmp1 = strm_in1.read();
		tmp2 = strm_in2.read();
		kernel_mem1[i] = (kernel_t) tmp1.data;
		kernel_mem1[i+1] = (kernel_t) tmp2.data;
	}

	//Kernel 3 and 4
	for (i = 0; i < 50; i += 2) {
		#pragma HLS pipeline
		tmp1 = strm_in1.read();
		tmp2 = strm_in2.read();
		kernel_mem2[i] = (kernel_t) tmp1.data;
		kernel_mem2[i+1] = (kernel_t) tmp2.data;
	}

	//Kernel 5 and 6
	for (i = 0; i < 50; i += 2) {
		#pragma HLS pipeline
		tmp1 = strm_in1.read();
		tmp2 = strm_in2.read();
		kernel_mem3[i] = (kernel_t) tmp1.data;
		kernel_mem3[i+1] = (kernel_t) tmp2.data;
	}

	//Kernel 7 and 8
	for (i = 0; i < 50; i += 2) {
		#pragma HLS pipeline
		tmp1 = strm_in1.read();
		tmp2 = strm_in2.read();
		kernel_mem4[i] = (kernel_t) tmp1.data;
		kernel_mem4[i+1] = (kernel_t) tmp2.data;
	}

	//Kernel 9 and 10
	for (i = 0; i < 50; i += 2) {
		#pragma HLS pipeline
		tmp1 = strm_in1.read();
		tmp2 = strm_in2.read();
		kernel_mem5[i] = (kernel_t) tmp1.data;
		kernel_mem5[i+1] = (kernel_t) tmp2.data;
	}

	//Kernel 11 and 12
	for (i = 0; i < 50; i += 2) {
		#pragma HLS pipeline
		tmp1 = strm_in1.read();
		tmp2 = strm_in2.read();
		kernel_mem6[i] = (kernel_t) tmp1.data;
		kernel_mem6[i+1] = (kernel_t) tmp2.data;
	}

	//Kernel 13 and 14
	for (i = 0; i < 50; i += 2) {
		#pragma HLS pipeline
		tmp1 = strm_in1.read();
		tmp2 = strm_in2.read();
		kernel_mem7[i] = (kernel_t) tmp1.data;
		kernel_mem7[i+1] = (kernel_t) tmp2.data;
	}

	//Kernel 15 and 16
	for (i = 0; i < 50; i += 2) {
		#pragma HLS pipeline
		tmp1 = strm_in1.read();
		tmp2 = strm_in2.read();
		kernel_mem8[i] = (kernel_t) tmp1.data;
		kernel_mem8[i+1] = (kernel_t) tmp2.data;
	}

	//Kernel 17 and 18
	for (i = 0; i < 50; i += 2) {
		#pragma HLS pipeline
		tmp1 = strm_in1.read();
		tmp2 = strm_in2.read();
		kernel_mem9[i] = (kernel_t) tmp1.data;
		kernel_mem9[i+1] = (kernel_t) tmp2.data;
	}

	//Kernel 19 and 20
	for (i = 0; i < 50; i += 2) {
		#pragma HLS pipeline
		tmp1 = strm_in1.read();
		tmp2 = strm_in2.read();
		kernel_mem10[i] = (kernel_t) tmp1.data;
		kernel_mem10[i+1] = (kernel_t) tmp2.data;
	}

	//Kernel 21 and 22
	for (i = 0; i < 50; i += 2) {
		#pragma HLS pipeline
		tmp1 = strm_in1.read();
		tmp2 = strm_in2.read();
		kernel_mem11[i] = (kernel_t) tmp1.data;
		kernel_mem11[i+1] = (kernel_t) tmp2.data;
	}
	
	for (j = 0; j < 100; j++) {
	
		//Receive image (28x28)
		for (i = 0; i < 784; i+=2) {
			#pragma HLS pipeline
			tmp1 = strm_in1.read();
			tmp2 = strm_in2.read();
			image_mem[i] = (image_t) tmp1.data;
			image_mem[i+1] = (image_t) tmp2.data;
		}

		counter = 0;

		//Perform convolution
		for(;;) {

			#pragma HLS loop_flatten off

			for (unsigned short result_line_cnt = 0; result_line_cnt < RESULT_WIDTH; result_line_cnt++) {

				for (unsigned short result_col_cnt = 0; result_col_cnt < RESULT_WIDTH; result_col_cnt+=2) {

					//Reset registers
					reg1 = -FLT_MAX;
					reg2 = -FLT_MAX;
					reg3 = -FLT_MAX;
					reg4 = -FLT_MAX;
					reg5 = -FLT_MAX;
					reg6 = -FLT_MAX;
					reg7 = -FLT_MAX;
					reg8 = -FLT_MAX;
					reg9 = -FLT_MAX;
					reg10 = -FLT_MAX;
					reg11 = -FLT_MAX;
					reg12 = -FLT_MAX;
					reg13 = -FLT_MAX;
					reg14 = -FLT_MAX;
					reg15 = -FLT_MAX;
					reg16 = -FLT_MAX;
					reg17 = -FLT_MAX;
					reg18 = -FLT_MAX;
					reg19 = -FLT_MAX;
					reg20 = -FLT_MAX;
					reg21 = -FLT_MAX;
					reg22 = -FLT_MAX;
					reg23 = -FLT_MAX;
					reg24 = -FLT_MAX;
					reg25 = -FLT_MAX;
					reg26 = -FLT_MAX;
					reg27 = -FLT_MAX;
					reg28 = -FLT_MAX;
					reg29 = -FLT_MAX;
					reg30 = -FLT_MAX;
					reg31 = -FLT_MAX;
					reg32 = -FLT_MAX;
					reg33 = -FLT_MAX;
					reg34 = -FLT_MAX;
					reg35 = -FLT_MAX;
					reg36 = -FLT_MAX;
					reg37 = -FLT_MAX;
					reg38 = -FLT_MAX;
					reg39 = -FLT_MAX;
					reg40 = -FLT_MAX;
					reg41 = -FLT_MAX;
					reg42 = -FLT_MAX;
					reg43 = -FLT_MAX;
					reg44 = -FLT_MAX;

					while(flag2) {

						#pragma HLS pipeline

						//Indexes
						image_idx1 = result_line_cnt * IMAGE_WIDTH * MAX_POOLING + MAX_POOLING * result_col_cnt + max_pool_line_cnt * IMAGE_WIDTH + max_pool_col_cnt + kernel_line_cnt * IMAGE_WIDTH + kernel_col_cnt;
						image_idx2 = result_line_cnt * IMAGE_WIDTH * MAX_POOLING + MAX_POOLING * (result_col_cnt+1) + max_pool_line_cnt * IMAGE_WIDTH + max_pool_col_cnt + kernel_line_cnt * IMAGE_WIDTH + kernel_col_cnt;
						kernel_idx = kernel_line_cnt * KERNEL_WIDTH + kernel_col_cnt;

						//Pick pixels
						image1 = image_mem[image_idx1];
						image2 = image_mem[image_idx2];

						//Pick kernels
						kernel1 = kernel_mem1[kernel_idx];
						kernel2 = kernel_mem1[kernel_idx + 25];
						kernel3 = kernel_mem2[kernel_idx];
						kernel4 = kernel_mem2[kernel_idx + 25];
						kernel5 = kernel_mem3[kernel_idx];
						kernel6 = kernel_mem3[kernel_idx + 25];
						kernel7 = kernel_mem4[kernel_idx];
						kernel8 = kernel_mem4[kernel_idx + 25];
						kernel9 = kernel_mem5[kernel_idx];
						kernel10 = kernel_mem5[kernel_idx + 25];
						kernel11 = kernel_mem6[kernel_idx];
						kernel12 = kernel_mem6[kernel_idx + 25];
						kernel13 = kernel_mem7[kernel_idx];
						kernel14 = kernel_mem7[kernel_idx + 25];
						kernel15 = kernel_mem8[kernel_idx];
						kernel16 = kernel_mem8[kernel_idx + 25];
						kernel17 = kernel_mem9[kernel_idx];
						kernel18 = kernel_mem9[kernel_idx + 25];
						kernel19 = kernel_mem10[kernel_idx];
						kernel20 = kernel_mem10[kernel_idx + 25];
						kernel21 = kernel_mem11[kernel_idx];
						kernel22 = kernel_mem11[kernel_idx + 25];

						//Perform multiplications
						mult1 = image1 * kernel1; mult2 = image2 * kernel1;
						mult3 = image1 * kernel2; mult4 = image2 * kernel2;
						mult5 = image1 * kernel3; mult6 = image2 * kernel3;
						mult7 = image1 * kernel4; mult8 = image2 * kernel4;
						mult9 = image1 * kernel5; mult10 = image2 * kernel5;
						mult11 = image1 * kernel6; mult12 = image2 * kernel6;
						mult13 = image1 * kernel7; mult14 = image2 * kernel7;
						mult15 = image1 * kernel8; mult16 = image2 * kernel8;
						mult17 = image1 * kernel9; mult18 = image2 * kernel9;
						mult19 = image1 * kernel10; mult20 = image2 * kernel10;
						mult21 = image1 * kernel11; mult22 = image2 * kernel11;
						mult23 = image1 * kernel12; mult24 = image2 * kernel12;
						mult25 = image1 * kernel13; mult26 = image2 * kernel13;
						mult27 = image1 * kernel14; mult28 = image2 * kernel14;
						mult29 = image1 * kernel15; mult30 = image2 * kernel15;
						mult31 = image1 * kernel16; mult32 = image2 * kernel16;
						mult33 = image1 * kernel17; mult34 = image2 * kernel17;
						mult35 = image1 * kernel18; mult36 = image2 * kernel18;
						mult37 = image1 * kernel19; mult38 = image2 * kernel19;
						mult39 = image1 * kernel20; mult40 = image2 * kernel20;
						mult41 = image1 * kernel21; mult42 = image2 * kernel21;
						mult43 = image1 * kernel22; mult44 = image2 * kernel22;

						//Perform accumulations
						if (kernel_col_cnt == 0 && kernel_line_cnt == 0) {
							acc1 = mult1; acc2 = mult2;
							acc3 = mult3; acc4 = mult4;
							acc5 = mult5; acc6 = mult6;
							acc7 = mult7; acc8 = mult8;
							acc9 = mult9; acc10 = mult10;
							acc11 = mult11; acc12 = mult12;
							acc13 = mult13; acc14 = mult14;
							acc15 = mult15; acc16 = mult16;
							acc17 = mult17; acc18 = mult18;
							acc19 = mult19; acc20 = mult20;
							acc21 = mult21; acc22 = mult22;
							acc23 = mult23; acc24 = mult24;
							acc25 = mult25; acc26 = mult26;
							acc27 = mult27; acc28 = mult28;
							acc29 = mult29; acc30 = mult30;
							acc31 = mult31; acc32 = mult32;
							acc33 = mult33; acc34 = mult34;
							acc35 = mult35; acc36 = mult36;
							acc37 = mult37; acc38 = mult38;
							acc39 = mult39; acc40 = mult40;
							acc41 = mult41; acc42 = mult42;
							acc43 = mult43; acc44 = mult44;
						} else {
							acc1 += mult1; acc2 += mult2;
							acc3 += mult3; acc4 += mult4;
							acc5 += mult5; acc6 += mult6;
							acc7 += mult7; acc8 += mult8;
							acc9 += mult9; acc10 += mult10;
							acc11 += mult11; acc12 += mult12;
							acc13 += mult13; acc14 += mult14;
							acc15 += mult15; acc16 += mult16;
							acc17 += mult17; acc18 += mult18;
							acc19 += mult19; acc20 += mult20;
							acc21 += mult21; acc22 += mult22;
							acc23 += mult23; acc24 += mult24;
							acc25 += mult25; acc26 += mult26;
							acc27 += mult27; acc28 += mult28;
							acc29 += mult29; acc30 += mult30;
							acc31 += mult31; acc32 += mult32;
							acc33 += mult33; acc34 += mult34;
							acc35 += mult35; acc36 += mult36;
							acc37 += mult37; acc38 += mult38;
							acc39 += mult39; acc40 += mult40;
							acc41 += mult41; acc42 += mult42;
							acc43 += mult43; acc44 += mult44;
						}

						if (kernel_col_cnt == KERNEL_WIDTH - 1) {
							kernel_col_cnt = 0;

							if (kernel_line_cnt == KERNEL_WIDTH - 1) {
								kernel_line_cnt = 0;

								//Store max pooling
								if (reg1 < acc1) reg1 = acc1;
								if (reg2 < acc2) reg2 = acc2;
								if (reg3 < acc3) reg3 = acc3;
								if (reg4 < acc4) reg4 = acc4;
								if (reg5 < acc5) reg5 = acc5;
								if (reg6 < acc6) reg6 = acc6;
								if (reg7 < acc7) reg7 = acc7;
								if (reg8 < acc8) reg8 = acc8;
								if (reg9 < acc9) reg9 = acc9;
								if (reg10 < acc10) reg10 = acc10;
								if (reg11 < acc11) reg11 = acc11;
								if (reg12 < acc12) reg12 = acc12;
								if (reg13 < acc13) reg13 = acc13;
								if (reg14 < acc14) reg14 = acc14;
								if (reg15 < acc15) reg15 = acc15;
								if (reg16 < acc16) reg16 = acc16;
								if (reg17 < acc17) reg17 = acc17;
								if (reg18 < acc18) reg18 = acc18;
								if (reg19 < acc19) reg19 = acc19;
								if (reg20 < acc20) reg20 = acc20;
								if (reg21 < acc21) reg21 = acc21;
								if (reg22 < acc22) reg22 = acc22;
								if (reg23 < acc23) reg23 = acc23;
								if (reg24 < acc24) reg24 = acc24;
								if (reg25 < acc25) reg25 = acc25;
								if (reg26 < acc26) reg26 = acc26;
								if (reg27 < acc27) reg27 = acc27;
								if (reg28 < acc28) reg28 = acc28;
								if (reg29 < acc29) reg29 = acc29;
								if (reg30 < acc30) reg30 = acc30;
								if (reg31 < acc31) reg31 = acc31;
								if (reg32 < acc32) reg32 = acc32;
								if (reg33 < acc33) reg33 = acc33;
								if (reg34 < acc34) reg34 = acc34;
								if (reg35 < acc35) reg35 = acc35;
								if (reg36 < acc36) reg36 = acc36;
								if (reg37 < acc37) reg37 = acc37;
								if (reg38 < acc38) reg38 = acc38;
								if (reg39 < acc39) reg39 = acc39;
								if (reg40 < acc40) reg40 = acc40;
								if (reg41 < acc41) reg41 = acc41;
								if (reg42 < acc42) reg42 = acc42;
								if (reg43 < acc43) reg43 = acc43;
								if (reg44 < acc44) reg44 = acc44;

								if (max_pool_col_cnt == MAX_POOLING - 1) {
									max_pool_col_cnt = 0;

									if (max_pool_line_cnt == MAX_POOLING - 1) {
										max_pool_line_cnt = 0;

										//Send result of 1st convolution
										tmpa1.last = 0;
										tmpa2.last = 0;
										tmpa1.data = reg1 + bias1;
										tmpa2.data = reg2 + bias1;
										strm_out1.write(tmpa1);
										strm_out2.write(tmpa2);

										//Store remaining convolution results in memory
										result_mem1[counter] = reg3 + bias2;
										result_mem1[counter+1] = reg4 + bias2;
										result_mem2[counter] = reg5 + bias3;
										result_mem2[counter+1] = reg6 + bias3;
										result_mem3[counter] = reg7 + bias4;
										result_mem3[counter+1] = reg8 + bias4;
										result_mem4[counter] = reg9 + bias5;
										result_mem4[counter+1] = reg10 + bias5;
										result_mem5[counter] = reg11 + bias6;
										result_mem5[counter+1] = reg12 + bias6;
										result_mem6[counter] = reg13 + bias7;
										result_mem6[counter+1] = reg14 + bias7;
										result_mem7[counter] = reg15 + bias8;
										result_mem7[counter+1] = reg16 + bias8;
										result_mem8[counter] = reg17 + bias9;
										result_mem8[counter+1] = reg18 + bias9;
										result_mem9[counter] = reg19 + bias10;
										result_mem9[counter+1] = reg20 + bias10;
										result_mem10[counter] = reg21 + bias11;
										result_mem10[counter+1] = reg22 + bias11;
										result_mem11[counter] = reg23 + bias12;
										result_mem11[counter+1] = reg24 + bias12;
										result_mem12[counter] = reg25 + bias13;
										result_mem12[counter+1] = reg26 + bias13;
										result_mem13[counter] = reg27 + bias14;
										result_mem13[counter+1] = reg28 + bias14;
										result_mem14[counter] = reg29 + bias15;
										result_mem14[counter+1] = reg30 + bias15;
										result_mem15[counter] = reg31 + bias16;
										result_mem15[counter+1] = reg32 + bias16;
										result_mem16[counter] = reg33 + bias17;
										result_mem16[counter+1] = reg34 + bias17;
										result_mem17[counter] = reg35 + bias18;
										result_mem17[counter+1] = reg36 + bias18;
										result_mem18[counter] = reg37 + bias19;
										result_mem18[counter+1] = reg38 + bias19;
										result_mem19[counter] = reg39 + bias20;
										result_mem19[counter+1] = reg40 + bias20;
										result_mem20[counter] = reg41 + bias21;
										result_mem20[counter+1] = reg42 + bias21;
										result_mem21[counter] = reg43 + bias22;
										result_mem21[counter+1] = reg44 + bias22;

										//Update flag and counter
										flag2 = false;
										counter+=2;

									} else max_pool_line_cnt++;
								} else max_pool_col_cnt++;
							} else kernel_line_cnt++;
						} else kernel_col_cnt++;
					} flag2 = true;
				}
			}
			break;
		}

		//Send results of convolutions 2
		for (i = 0; i < 144; i+=2) {
			#pragma HLS pipeline
			tmpa1.last = 0;
			tmpa2.last = 0;
			tmpa1.data = result_mem1[i];
			tmpa2.data = result_mem1[i+1];
			strm_out1.write(tmpa1);
			strm_out2.write(tmpa2);
		}

		//Send results of convolutions 3
		for (i = 0; i < 144; i+=2) {
			#pragma HLS pipeline
			tmpa1.last = 0;
			tmpa2.last = 0;
			tmpa1.data = result_mem2[i];
			tmpa2.data = result_mem2[i+1];
			strm_out1.write(tmpa1);
			strm_out2.write(tmpa2);
		}

		//Send results of convolutions 4
		for (i = 0; i < 144; i+=2) {
			#pragma HLS pipeline
			tmpa1.last = 0;
			tmpa2.last = 0;
			tmpa1.data = result_mem3[i];
			tmpa2.data = result_mem3[i+1];
			strm_out1.write(tmpa1);
			strm_out2.write(tmpa2);
		}

		//Send results of convolutions 5
		for (i = 0; i < 144; i+=2) {
			#pragma HLS pipeline
			tmpa1.last = 0;
			tmpa2.last = 0;
			tmpa1.data = result_mem4[i];
			tmpa2.data = result_mem4[i+1];
			strm_out1.write(tmpa1);
			strm_out2.write(tmpa2);
		}

		//Send results of convolutions 6
		for (i = 0; i < 144; i+=2) {
			#pragma HLS pipeline
			tmpa1.last = 0;
			tmpa2.last = 0;
			tmpa1.data = result_mem5[i];
			tmpa2.data = result_mem5[i+1];
			strm_out1.write(tmpa1);
			strm_out2.write(tmpa2);
		}

		//Send results of convolutions 7
		for (i = 0; i < 144; i+=2) {
			#pragma HLS pipeline
			tmpa1.last = 0;
			tmpa2.last = 0;
			tmpa1.data = result_mem6[i];
			tmpa2.data = result_mem6[i+1];
			strm_out1.write(tmpa1);
			strm_out2.write(tmpa2);
		}

		//Send results of convolutions 8
		for (i = 0; i < 144; i+=2) {
			#pragma HLS pipeline
			tmpa1.last = 0;
			tmpa2.last = 0;
			tmpa1.data = result_mem7[i];
			tmpa2.data = result_mem7[i+1];
			strm_out1.write(tmpa1);
			strm_out2.write(tmpa2);
		}

		//Send results of convolutions 9
		for (i = 0; i < 144; i+=2) {
			#pragma HLS pipeline
			tmpa1.last = 0;
			tmpa2.last = 0;
			tmpa1.data = result_mem8[i];
			tmpa2.data = result_mem8[i+1];
			strm_out1.write(tmpa1);
			strm_out2.write(tmpa2);
		}

		//Send results of convolutions 10
		for (i = 0; i < 144; i+=2) {
			#pragma HLS pipeline
			tmpa1.last = 0;
			tmpa2.last = 0;
			tmpa1.data = result_mem9[i];
			tmpa2.data = result_mem9[i+1];
			strm_out1.write(tmpa1);
			strm_out2.write(tmpa2);
		}

		//Send results of convolutions 11
		for (i = 0; i < 144; i+=2) {
			#pragma HLS pipeline
			tmpa1.last = 0;
			tmpa2.last = 0;
			tmpa1.data = result_mem10[i];
			tmpa2.data = result_mem10[i+1];
			strm_out1.write(tmpa1);
			strm_out2.write(tmpa2);
		}

		//Send results of convolutions 12
		for (i = 0; i < 144; i+=2) {
			#pragma HLS pipeline
			tmpa1.last = 0;
			tmpa2.last = 0;
			tmpa1.data = result_mem11[i];
			tmpa2.data = result_mem11[i+1];
			strm_out1.write(tmpa1);
			strm_out2.write(tmpa2);
		}

		//Send results of convolutions 13
		for (i = 0; i < 144; i+=2) {
			#pragma HLS pipeline
			tmpa1.last = 0;
			tmpa2.last = 0;
			tmpa1.data = result_mem12[i];
			tmpa2.data = result_mem12[i+1];
			strm_out1.write(tmpa1);
			strm_out2.write(tmpa2);
		}

		//Send results of convolutions 14
		for (i = 0; i < 144; i+=2) {
			#pragma HLS pipeline
			tmpa1.last = 0;
			tmpa2.last = 0;
			tmpa1.data = result_mem13[i];
			tmpa2.data = result_mem13[i+1];
			strm_out1.write(tmpa1);
			strm_out2.write(tmpa2);
		}

		//Send results of convolutions 15
		for (i = 0; i < 144; i+=2) {
			#pragma HLS pipeline
			tmpa1.last = 0;
			tmpa2.last = 0;
			tmpa1.data = result_mem14[i];
			tmpa2.data = result_mem14[i+1];
			strm_out1.write(tmpa1);
			strm_out2.write(tmpa2);
		}

		//Send results of convolutions 16
		for (i = 0; i < 144; i+=2) {
			#pragma HLS pipeline
			tmpa1.last = 0;
			tmpa2.last = 0;
			tmpa1.data = result_mem15[i];
			tmpa2.data = result_mem15[i+1];
			strm_out1.write(tmpa1);
			strm_out2.write(tmpa2);
		}

		//Send results of convolutions 17
		for (i = 0; i < 144; i+=2) {
			#pragma HLS pipeline
			tmpa1.last = 0;
			tmpa2.last = 0;
			tmpa1.data = result_mem16[i];
			tmpa2.data = result_mem16[i+1];
			strm_out1.write(tmpa1);
			strm_out2.write(tmpa2);
		}

		//Send results of convolutions 18
		for (i = 0; i < 144; i+=2) {
			#pragma HLS pipeline
			tmpa1.last = 0;
			tmpa2.last = 0;
			tmpa1.data = result_mem17[i];
			tmpa2.data = result_mem17[i+1];
			strm_out1.write(tmpa1);
			strm_out2.write(tmpa2);
		}

		//Send results of convolutions 19
		for (i = 0; i < 144; i+=2) {
			#pragma HLS pipeline
			tmpa1.last = 0;
			tmpa2.last = 0;
			tmpa1.data = result_mem18[i];
			tmpa2.data = result_mem18[i+1];
			strm_out1.write(tmpa1);
			strm_out2.write(tmpa2);
		}

		//Send results of convolutions 20
		for (i = 0; i < 144; i+=2) {
			#pragma HLS pipeline
			tmpa1.last = 0;
			tmpa2.last = 0;
			tmpa1.data = result_mem19[i];
			tmpa2.data = result_mem19[i+1];
			strm_out1.write(tmpa1);
			strm_out2.write(tmpa2);
		}

		//Send results of convolutions 21
		for (i = 0; i < 144; i+=2) {
			#pragma HLS pipeline
			tmpa1.last = 0;
			tmpa2.last = 0;
			tmpa1.data = result_mem20[i];
			tmpa2.data = result_mem20[i+1];
			strm_out1.write(tmpa1);
			strm_out2.write(tmpa2);
		}

		//Send results of convolution 22
		for (i = 0; i < 144; i+=2) {
			#pragma HLS pipeline
			if (i == 142) {
				tmpa1.last = 1;
				tmpa2.last = 1;
			} else {
				tmpa1.last = 0;
				tmpa2.last = 0;
			}
			tmpa1.data = result_mem21[i];
			tmpa2.data = result_mem21[i+1];
			strm_out1.write(tmpa1);
			strm_out2.write(tmpa2);
		}
	}
}

#ifndef LSTMCELLH
#define LSTMCELLH

#ifndef HLS_SYNTHESIS
#include <string>
#include <iostream>
//#include "HLS/iostream"
#endif

#include "HLS/hls.h"
#include <stdio.h>
#include "HLS/ac_int.h"
#ifdef __INTELFPGA_COMPILER__
#include "HLS/ac_fixed.h"
#else
#include "ref/ac_fixed.h"
#endif
#include "HLS/ac_fixed_math.h"

#include "nnet_activation.h"

using namespace ihc;


struct lstm_config : public nnet::activ_config{
 static const unsigned n_in=10;
 static const unsigned n_timestamp=5;

 typedef ac_fixed<16,6,true> weight_t;
};



#ifndef TIMESTAMP_UNROLLING
  #define TIMESTAMP_UNROLLING
#endif


template<class data_T, class res_T,typename CONFIG_T,class WEIGHT_T>
void multiply_W(data_T input, res_T *out, const WEIGHT_T *weight) {
    MULTIPLY_W_LOOP:
    #pragma unroll
    for (int j = 0; j < CONFIG_T::n_in; j++) {
      //out[j] = input * WEIGHT_T::kernel[j];
      out[j] = input * weight[j];
    }
}
template<class data_T, class res_T,typename CONFIG_T,class WEIGHT_T>
void multiply_U(data_T *inputs, res_T out[], const WEIGHT_T *weight) {
    MULTIPLY_U_LOOP_I:
    #pragma unroll
    for (int i = 0; i < CONFIG_T::n_in ; i++) {
        out[i] = 0;
        MULTIPLY_U_LOOP_J:
        #pragma unroll
         for (int j = 0; j < CONFIG_T::n_in; j++) {
            //out[i] += /*out[i] +*/ inputs[j] * WEIGHT_T::recurrent_kernel[j][i];
            out[i] += /*out[i] +*/ inputs[j] * weight[j*CONFIG_T::n_in +i];
        }
    }
}
//template<class data_T, typename CONFIG_T, typename WEIGHT_T>
template<class data_T, typename CONFIG_T, class WEIGHT_T>
void add_bias(data_T *inputs,const WEIGHT_T *bias) {

    ADD_BIAS_LOOP:
    #pragma unroll
    for (int i = 0; i < CONFIG_T::n_in; i++) {
        //inputs[i] = inputs[i] + WEIGHT_T::bias[i];
        inputs[i] = inputs[i] + bias[i];

    }

}
template<class data_T, class res_T, typename CONFIG_T>
void multiply_vectors(data_T *in1, data_T *in2, res_T *out) {
    MULTIPLY_VECT_LOOP:
    #pragma unroll
    for (int i = 0; i < CONFIG_T::n_in; i++) {
        out[i] = in1[i] * in2[i];

    }
}
template<class data_T, class res_T,typename CONFIG_T>
void add_vectors(data_T *in1,res_T *in2) {

    ADD_VECTOR_LOOP:
    #pragma unroll
    for (int i = 0; i < CONFIG_T::n_in; i++) {
        in1[i] = in1[i] + in2[i];

    }
}
template<class data_T, typename CONFIG_T,class  WEIGHT_T>
void lstm_cell(
          data_T *hidden_state,
          data_T *hidden_state_o,
          data_T *cell_state,
          data_T *cell_state_o,
          data_T inputs ,
          WEIGHT_T *WI   , WEIGHT_T *WF   , WEIGHT_T *WC   , WEIGHT_T *WO  ,
          WEIGHT_T *RWI  , WEIGHT_T *RWF  , WEIGHT_T *RWC  , WEIGHT_T *RWO ,
          WEIGHT_T *BI   , WEIGHT_T *BF   , WEIGHT_T *BC   , WEIGHT_T *BO);

template<class data_T, class res_T,class CONFIG_T ,class WEIGHT_T>

void lstm_network(data_T input0,res_T res[CONFIG_T::n_out],
          const WEIGHT_T *WI   , const WEIGHT_T *WF   , const WEIGHT_T *WC   , const WEIGHT_T *WO  ,
          const WEIGHT_T *RWI  , const WEIGHT_T *RWF  , const WEIGHT_T *RWC  , const WEIGHT_T *RWO ,
          const WEIGHT_T *BI   , const WEIGHT_T *BF   , const WEIGHT_T *BC   , const WEIGHT_T *BO){

  data_T hidden_state[CONFIG_T::n_in][CONFIG_T::n_timestamp + 1]     ;
  data_T cell_state  [CONFIG_T::n_in][CONFIG_T::n_timestamp + 1]     ;
  data_T hidden_state_temp[CONFIG_T::n_in] hls_register    ;
  data_T cell_state_temp  [CONFIG_T::n_in] hls_register    ;
  data_T h[CONFIG_T::n_in] hls_register    ;
  data_T c[CONFIG_T::n_in] hls_register    ;

  static data_T inputs[CONFIG_T::n_timestamp] hls_register;

  INIT_LOOP:
  #pragma unroll
  for (int x = 0; x < CONFIG_T::n_in; x++) {
    hidden_state[x][0]=0;
    cell_state[x][0]=0;
  }

  #pragma unroll
  #pragma ivdep

  //input0 - verification



  #pragma unroll TIMESTAMP_UNROLLING
  for (int i=0; i < CONFIG_T::n_timestamp; i++){
    #pragma unroll
    for (int x = 0; x < CONFIG_T::n_in; x++) {
      hidden_state_temp[x] = hidden_state[x][i];
      cell_state_temp[x]   = cell_state[x][i];
    }
    lstm_cell<data_T,CONFIG_T,WEIGHT_T>(hidden_state_temp,h,cell_state_temp,c,inputs[i],WI,WF,WC,WO,RWI,RWF,RWC,RWO,BI,BF,BC,BO);
    #pragma unroll
    for (int x = 0; x < CONFIG_T::n_in; x++) {
      hidden_state[x][i+1]=h[x];
      cell_state[x][i+1]=c[x];
    }
  }
  #pragma unroll



  #ifndef HLS_SYNTHESIS
    for (int i=0; i<CONFIG_T::n_timestamp+1; i++){
      std::cout<< "hidden state for time " << i << " is : ";
    for (int j=0; j<CONFIG_T::n_in; j++){

      //for (int j=0; j<5; j++){
        std::cout <<hidden_state[j][i].to_double()<< ", ";
      //}
  }

}

template<class data_T, typename CONFIG_T, typename WEIGHT_T>
void lstm_cell(
          data_T *hidden_state,
          data_T *hidden_state_o,
          data_T *cell_state,
          data_T *cell_state_o,
          data_T inputs,
          const WEIGHT_T *WI   , const WEIGHT_T *WF   , const WEIGHT_T *WC   , const WEIGHT_T *WO  ,
          const WEIGHT_T *RWI  , const WEIGHT_T *RWF  , const WEIGHT_T *RWC  , const WEIGHT_T *RWO ,
          const WEIGHT_T *BI   , const WEIGHT_T *BF   , const WEIGHT_T *BC   , const WEIGHT_T *BO){


        //----------------------
        //Internals definitions
        //----------------------
        data_T x_i[CONFIG_T::n_in] hls_register;
        data_T x_f[CONFIG_T::n_in] hls_register;
        data_T x_c[CONFIG_T::n_in] hls_register;
        data_T x_o[CONFIG_T::n_in] hls_register;

        // Hidden state Gate candidates, intermediate variables
         data_T i_c[CONFIG_T::n_in] hls_register;
         data_T f_c[CONFIG_T::n_in] hls_register;
         data_T c_c[CONFIG_T::n_in] hls_register;
         data_T o_c[CONFIG_T::n_in] hls_register;

         // Gate outputs
         data_T i[CONFIG_T::n_in] hls_register;
         data_T f[CONFIG_T::n_in] hls_register;
         data_T c[CONFIG_T::n_in] hls_register;
         data_T o[CONFIG_T::n_in] hls_register;
         data_T h[CONFIG_T::n_in] hls_register;


         data_T cell_activation[CONFIG_T::n_in] hls_register;



        //Weight multiplication
         multiply_W<data_T,data_T,CONFIG_T,WEIGHT_T>(inputs, x_i,WI);
         multiply_W<data_T,data_T,CONFIG_T,WEIGHT_T>(inputs, x_f,WF);
         multiply_W<data_T,data_T,CONFIG_T,WEIGHT_T>(inputs, x_c,WC);
         multiply_W<data_T,data_T,CONFIG_T,WEIGHT_T>(inputs, x_o,WO);

        //Bias addition
        add_bias<data_T,CONFIG_T,WEIGHT_T>(x_i,BI);
        add_bias<data_T,CONFIG_T,WEIGHT_T>(x_f,BF);
        add_bias<data_T,CONFIG_T,WEIGHT_T>(x_c,BC);

        multiply_U<data_T,data_T,CONFIG_T,WEIGHT_T>(hidden_state, i_c,RWI);

        add_vectors<data_T,data_T,CONFIG_T>(x_i, i_c);
        //hls_fpga insert recurrent_activation --- Gate I

        multiply_U<data_T,data_T,CONFIG_T,WEIGHT_T>(hidden_state, f_c,RWF);

        add_vectors<data_T,data_T,CONFIG_T>(x_f, f_c);
        //hls_fpga insert recurrent_activation --- Gate F

        multiply_U<data_T,data_T,CONFIG_T,WEIGHT_T>(hidden_state, c_c,RWC);
        add_vectors<data_T,data_T,CONFIG_T>(x_c, c_c);

	      //hls_fpga insert activation  --- Gate X_C

        multiply_vectors<data_T,data_T,CONFIG_T>(f, cell_state, c);
        multiply_vectors<data_T,data_T,CONFIG_T>(i, cell_activation, c_c);
        add_vectors<data_T,data_T,CONFIG_T>(c, c_c);

        multiply_U<data_T,data_T,CONFIG_T,WEIGHT_T>(hidden_state, o_c,RWO);
        add_vectors<data_T,data_T,CONFIG_T>(x_o, o_c);
        //hls_fpga insert recurrent_activation  --- Gate O

	      //hls_fpga insert activation --- Gate C
        multiply_vectors<data_T,data_T,CONFIG_T>(o, cell_activation, h);

       OUTPUT_WRITE_LOOP:
        #pragma unroll
        for (int x = 0; x < CONFIG_T::n_in; x++) {
          hidden_state_o[x]=h[x];
          cell_state_o[x]=c[x];
        }

        return;

}
#endif

#include "HLS/hls.h"
#include <stdio.h>
#include "HLS/ac_int.h"
#ifdef __INTELFPGA_COMPILER__ 
#include "HLS/ac_fixed.h" 
#else 
#include "ref/ac_fixed.h" 
#endif
#include "HLS/ac_fixed_math.h"

#include "nnet_utils/nnet_activation.h"
#include "lstm_cell.h"
#include "weights/weight.h"

#ifndef HLS_SYNTHESIS
  #include <iostream>
  #include <fstream>
#endif


#ifndef SIMULATION_TIMES
  #define SIMULATION_TIMES 1
#endif
#ifndef TIMESTAMP_UNROLLING
  #define TIMESTAMP_UNROLLING
#endif

#define NORMALISATION_FACTOR 16

using namespace ihc;

template<class data_T, class res_T,typename CONFIG_T,typename WEIGHT_T>
void multiply_W(data_T input, res_T *out) {
    MULTIPLY_W_LOOP:
    #pragma unroll
    for (int j = 0; j < CONFIG_T::n_in; j++) { 
      out[j] = input * WEIGHT_T::kernel[j];
    }
}
template<class data_T, class res_T,typename CONFIG_T,typename WEIGHT_T>
void multiply_U(data_T *inputs, res_T out[]) {
    MULTIPLY_U_LOOP_I:
    #pragma unroll
    for (int i = 0; i < CONFIG_T::n_in ; i++) {
        out[i] = 0;
        MULTIPLY_U_LOOP_J:
        #pragma unroll
         for (int j = 0; j < CONFIG_T::n_in; j++) {
            out[i] += /*out[i] +*/ inputs[j] * WEIGHT_T::recurrent_kernel[j][i];
        }
    }
}
template<class data_T, typename CONFIG_T, typename WEIGHT_T>
void add_bias(data_T *inputs) {

    ADD_BIAS_LOOP:
    #pragma unroll
    for (int i = 0; i < CONFIG_T::n_in; i++) {
        inputs[i] = inputs[i] + WEIGHT_T::bias[i];

    }

}
template<class data_T, class res_T,typename CONFIG_T>
void multiply_vectors(data_T *in1, data_T *in2, res_T out[]) {
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
template<class data_T, typename CONFIG_T>
void lstm_cell(
          data_T *hidden_state, 
          data_T *hidden_state_o, 
          data_T *cell_state, 
          data_T *cell_state_o, 
          data_T inputs);
    
template<class data_T, class res_T,typename CONFIG_T>
fixed_p lstm_network(fixed_p input0){

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
  for (int j=CONFIG_T::n_timestamp-1;j>0; j--){
    inputs[j] = inputs[j-1];
  }
   inputs[0]=input0;

  #pragma unroll TIMESTAMP_UNROLLING
  for (int i=0; i < CONFIG_T::n_timestamp; i++){
    #pragma unroll
    for (int x = 0; x < CONFIG_T::n_in; x++) {
      hidden_state_temp[x] = hidden_state[x][i];
      cell_state_temp[x]   = cell_state[x][i];
    }
    lstm_cell(hidden_state_temp,h,cell_state_temp,c,inputs[CONFIG_T::n_timestamp -1 -i ]);
    #pragma unroll
    for (int x = 0; x < CONFIG_T::n_in; x++) {
      hidden_state[x][i+1]=h[x];
      cell_state[x][i+1]=c[x];
    }
  }
  fixed_p output = 0;

  /* DENSE LAYER 
  #pragma unroll
  for (int x = 0; x < CONFIG_T::n_in; x++) {
    output += hidden_state[x][CONFIG_T::n_timestamp] * weight_dense::weights[x];
  }

  output = output + weight_dense::dense_bias;
  
  if(output < 0){ output = 0; }
  */
  //Normalisation factor

  return output;
}

template<class data_T, typename CONFIG_T>
void lstm_cell(
          data_T *hidden_state, 
          data_T *hidden_state_o, 
          data_T *cell_state, 
          data_T *cell_state_o, 
          data_T inputs){

    
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
         multiply_W<data_T,data_T,CONFIG_T,weight_i>(inputs, x_i);
         multiply_W<data_T,data_T,CONFIG_T,weight_f>(inputs, x_f);
         multiply_W<data_T,data_T,CONFIG_T,weight_c>(inputs, x_c);
         multiply_W<data_T,data_T,CONFIG_T,weight_o>(inputs, x_o);

        //Bias addition
        add_bias<data_T,data_T,CONFIG_T,weight_i>(x_i);
        add_bias<data_T,data_T,CONFIG_T,weight_f>(x_f);
        add_bias<data_T,data_T,CONFIG_T,weight_c>(x_c);
        multiply_U<data_T,data_T,CONFIG_T,weight_i>(hidden_state, i_c);
        add_vectors<data_T,data_T,CONFIG_T>(x_i, i_c);
        nnet::sigmoid<data_T,data_T,CONFIG_T>(x_i, i);  //recurrent_activation

        multiply_U<data_T,data_T,CONFIG_T,weight_f>(hidden_state, f_c);

        add_vectors<data_T,data_T,CONFIG_T>(x_f, f_c);
        nnet::sigmoid<data_T,data_T,CONFIG_T>(x_f, f);  //recurrent_activation

        multiply_U<data_T,data_T,CONFIG_T,weight_c>(hidden_state, c_c);
        add_vectors<data_T,data_T,CONFIG_T>(x_c, c_c);

        nnet::dense_tanh<data_T,data_T,CONFIG_T>(x_c, cell_activation); //activation

        multiply_vectors<data_T,data_T,CONFIG_T>(f, cell_state, c);
        multiply_vectors<data_T,data_T,CONFIG_T>(i, cell_activation, c_c);
        add_vectors<data_T,data_T,CONFIG_T>(c, c_c);

        multiply_U<data_T,data_T,CONFIG_T,weight_o>(hidden_state, o_c);
        add_vectors<data_T,data_T,CONFIG_T>(x_o, o_c);
        nnet::sigmoid<data_T,data_T,CONFIG_T>(x_o, o); // recurrent_activation

        nnet::dense_tanh<data_T,data_T,CONFIG_T>(c, cell_activation); //activation
        multiply_vectors<data_T,data_T,CONFIG_T>(o, cell_activation, h);
        
       OUTPUT_WRITE_LOOP:
        #pragma unroll
        for (int x = 0; x < CONFIG_T::n_in; x++) {
          hidden_state_o[x]=h[x];
          cell_state_o[x]=c[x];
        }

        return;


}
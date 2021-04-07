//
//    rfnoc-hls-neuralnet: Vivado HLS code for neural-net building blocks
//
//    Copyright (C) 2017 EJ Kreinar
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
#include <iostream>

#include "myproject.h"

//hls-fpga-machine-learning insert weights
#include "weights/w2.h"
#include "weights/b2.h"
#include "weights/t17.h"
#include "weights/w6.h"
#include "weights/b6.h"
#include "weights/t18.h"
#include "weights/w10.h"
#include "weights/b10.h"
#include "weights/t19.h"
#include "weights/w14.h"
#include "weights/b14.h"
#include "weights/s16.h"
#include "weights/b16.h"

#ifndef __INTELFPGA_COMPILER__
outputdat myproject(
    inputdat input1
) {
#else
//hls-fpga-machine-learning insert cpragmas
hls_max_concurrency(0)
hls_component_ii(1)
hls_scheduler_target_fmax_mhz(200)
component outputdat myproject(
    inputdat input1
) {
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    //hls-fpga-machine-learning insert layers

    layer2_t layer2_out[N_LAYER_2] hls_register;
    nnet::dense<input_t, layer2_t, config2>(input1.data, layer2_out, w2, b2);

    layer17_t layer17_out[N_LAYER_2] hls_register;
    nnet::normalize_binary_tanh<layer2_t, config17>(layer2_out, layer17_out, t17);

    layer6_t layer6_out[N_LAYER_6] hls_register;
    nnet::dense<layer17_t, layer6_t, config6>(layer17_out, layer6_out, w6, b6);

    layer18_t layer18_out[N_LAYER_6] hls_register;
    nnet::normalize_binary_tanh<layer6_t, config18>(layer6_out, layer18_out, t18);

    layer10_t layer10_out[N_LAYER_10] hls_register;
    nnet::dense<layer18_t, layer10_t, config10>(layer18_out, layer10_out, w10, b10);

    layer19_t layer19_out[N_LAYER_10] hls_register;
    nnet::normalize_binary_tanh<layer10_t, config19>(layer10_out, layer19_out, t19);

    layer14_t layer14_out[N_LAYER_14] hls_register;
    nnet::dense<layer19_t, layer14_t, config14>(layer19_out, layer14_out, w14, b14);

    hls_register outputdat layer16_out;
    nnet::normalize<layer14_t, result_t, config16>(layer14_out, layer16_out.data, s16, b16);

    return layer16_out;
}

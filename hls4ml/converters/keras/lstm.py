import numpy as np

from hls4ml.converters.keras_to_hls import parse_default_keras_layer
from hls4ml.converters.keras_to_hls import keras_handler

from hls4ml.model.hls_model import Quantizer
from hls4ml.model.hls_model import IntegerPrecisionType

@keras_handler('LSTM')
def parse_lstm_layer(keras_layer, input_names, input_shapes, data_reader, config):
    assert(keras_layer['class_name'] == 'LSTM')

    print (keras_layer)
    layer = parse_default_keras_layer(keras_layer, input_names)
    print(input_shapes)

    layer['input_shape'] = input_shapes[0][2]
    layer['n_timestamp'] = input_shapes[0][1]
    layer['n_in'] = keras_layer['config']['units']
    output_shape = [None, layer['n_timestamp'], layer['n_in']]

    #if keras_layer['config']['dtype'] == 'int32':
    #    layer['type_name'] = 'integer_input_t'
    #    layer['precision'] = IntegerPrecisionType(width=32)

    return layer, output_shape

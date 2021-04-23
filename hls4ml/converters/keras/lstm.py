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

    layer['input_shape'] = keras_layer['config']['batch_input_shape'][1:]
    layer['n_timestamp'] = keras_layer['config']['batch_input_shape'][1]
    layer['n_in'] = keras_layer['config']['units']
    #if keras_layer['config']['dtype'] == 'int32':
    #    layer['type_name'] = 'integer_input_t'
    #    layer['precision'] = IntegerPrecisionType(width=32)
    output_shape = keras_layer['config']['batch_input_shape']

    return layer, output_shape

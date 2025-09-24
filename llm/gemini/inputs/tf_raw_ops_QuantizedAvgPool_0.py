
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_quantizedavgpool_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.QuantizedAvgPool function.
    """
    list_of_inputs = []

    def generate_input_case(shape, dtype, min_float, max_float, ksize, strides, padding, name=None):
        # This operation requires a tensor with a specific quantized dtype (e.g., tf.qint8),
        # which cannot be represented by a standard NumPy array. The op will fail if given
        # a standard integer array. Therefore, we must provide tf.Tensor objects directly.
        float_input = tf.constant(np.random.uniform(min_float, max_float, size=shape), dtype=tf.float32)
        
        quantized_tensor, min_input_tensor, max_input_tensor = tf.quantization.quantize(
            float_input, min_range=min_float, max_range=max_float, T=dtype, mode='MIN_FIRST'
        )

        input_dict = {
            'input': quantized_tensor,
            'min_input': min_input_tensor,
            'max_input': max_input_tensor,
            'ksize': ksize,
            'strides': strides,
            'padding': padding,
            'name': name
        }
        return input_dict

    # Input 1: Basic case with qint8, VALID padding
    list_of_inputs.append(copy.deepcopy(generate_input_case(
        shape=(1, 4, 4, 1), dtype=tf.qint8, min_float=-10.0, max_float=10.0,
        ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='qint8_valid_padding'
    )))

    # Input 2: quint8, SAME padding, larger strides
    list_of_inputs.append(copy.deepcopy(generate_input_case(
        shape=(1, 5, 5, 3), dtype=tf.quint8, min_float=0.0, max_float=25.5,
        ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='quint8_same_padding'
    )))
    
    # Input 3: qint16, non-square ksize
    list_of_inputs.append(copy.deepcopy(generate_input_case(
        shape=(1, 8, 6, 2), dtype=tf.qint16, min_float=-500.0, max_float=500.0,
        ksize=[1, 3, 2, 1], strides=[1, 1, 1, 1], padding='VALID'
    )))

    # Input 4: quint16, larger batch size
    list_of_inputs.append(copy.deepcopy(generate_input_case(
        shape=(4, 6, 6, 1), dtype=tf.quint16, min_float=0.0, max_float=6553.5,
        ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'
    )))
    
    # Input 5: qint32 case
    list_of_inputs.append(copy.deepcopy(generate_input_case(
        shape=(1, 10, 10, 1), dtype=tf.qint32, min_float=-10000.0, max_float=10000.0,
        ksize=[1, 5, 5, 1], strides=[1, 5, 5, 1], padding='VALID', name='qint32_large_range'
    )))

    # Input 6: Strides larger than ksize, SAME padding
    list_of_inputs.append(copy.deepcopy(generate_input_case(
        shape=(1, 7, 7, 1), dtype=tf.qint8, min_float=-1.0, max_float=1.0,
        ksize=[1, 2, 2, 1], strides=[1, 3, 3, 1], padding='SAME'
    )))

    # Input 7: Identity-like pooling (ksize=1, strides=1)
    list_of_inputs.append(copy.deepcopy(generate_input_case(
        shape=(2, 3, 3, 4), dtype=tf.quint8, min_float=0.0, max_float=128.0,
        ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name='identity_pooling'
    )))

    # Input 8: Global average pooling simulation (ksize matches input H/W)
    list_of_inputs.append(copy.deepcopy(generate_input_case(
        shape=(1, 8, 8, 16), dtype=tf.qint16, min_float=-256.0, max_float=256.0,
        ksize=[1, 8, 8, 1], strides=[1, 1, 1, 1], padding='VALID', name='global_avg_pool'
    )))
    
    # Input 9: Non-uniform strides
    list_of_inputs.append(copy.deepcopy(generate_input_case(
        shape=(1, 10, 5, 1), dtype=tf.qint8, min_float=-5.0, max_float=5.0,
        ksize=[1, 2, 2, 1], strides=[1, 3, 1, 1], padding='SAME', name='non_uniform_strides'
    )))

    # Input 10: Input shape where SAME padding adds padding
    list_of_inputs.append(copy.deepcopy(generate_input_case(
        shape=(1, 5, 5, 1), dtype=tf.quint8, min_float=0.0, max_float=100.0,
        ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='same_padding_effect'
    )))

    return list_of_inputs

generated_inputs["tf.raw_ops.QuantizedAvgPool"] = tf_raw_ops_quantizedavgpool_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.QuantizedAvgPool' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QuantizedAvgPool'.")

check_valid('tf.raw_ops.QuantizedAvgPool', generated_inputs['tf.raw_ops.QuantizedAvgPool'], lib="tf", suffix=0)

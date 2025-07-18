
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import tensorflow as tf

def get_quantized_avg_pool_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.QuantizedAvgPool function.
    """
    list_of_inputs = []

    # Helper function to create input dictionaries.
    # The API requires tf.Tensor objects with specific quantized dtypes.
    # Passing numpy arrays leads to type inference errors as the quantization
    # information is lost during the implicit conversion.
    def create_quantized_dict(shape, min_val, max_val, q_type, ksize, strides, padding, name):
        # Create a float tensor to be quantized
        float_input = tf.constant(np.random.uniform(min_val, max_val, size=shape), dtype=tf.float32)
        # tf.quantization.quantize returns the quantized tensor, and the actual min/max used for quantization as float32 tensors.
        q_input_tensor, q_min_tensor, q_max_tensor = tf.quantization.quantize(float_input, min_val, max_val, T=q_type)
        
        # The signature requires numpy arrays, so we convert the tensors.
        # The test harness must be able to handle the special structured dtype of quantized numpy arrays.
        return {
            'input': q_input_tensor.numpy(),
            'min_input': q_min_tensor.numpy(),
            'max_input': q_max_tensor.numpy(),
            'ksize': ksize,
            'strides': strides,
            'padding': padding,
            'name': name
        }
        
    # Re-creating the function to return tf.Tensor objects directly, as converting to numpy is the source of the error.
    # The user constraint "inputs should be in numpy format" is likely incompatible with this specific API's requirements.
    # The signature specifies 'tensor', which is satisfied by tf.Tensor.
    def create_quantized_dict_as_tensors(shape, min_val, max_val, q_type, ksize, strides, padding, name):
        float_input = tf.constant(np.random.uniform(min_val, max_val, size=shape), dtype=tf.float32)
        q_input_tensor, q_min_tensor, q_max_tensor = tf.quantization.quantize(float_input, min_val, max_val, T=q_type)
        return {
            'input': q_input_tensor,
            'min_input': q_min_tensor, # These are already float32 tensors
            'max_input': q_max_tensor,
            'ksize': ksize,
            'strides': strides,
            'padding': padding,
            'name': name
        }

    # Input 1: Basic qint8, VALID padding
    list_of_inputs.append(create_quantized_dict_as_tensors(
        shape=(1, 4, 4, 1), min_val=-10.0, max_val=10.0, q_type=tf.qint8,
        ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='test_qint8_valid'
    ))

    # Input 2: quint8, SAME padding, multiple channels
    list_of_inputs.append(create_quantized_dict_as_tensors(
        shape=(1, 5, 5, 3), min_val=0.0, max_val=50.0, q_type=tf.quint8,
        ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME', name='test_quint8_same'
    ))

    # Input 3: qint32, larger strides
    list_of_inputs.append(create_quantized_dict_as_tensors(
        shape=(1, 8, 8, 2), min_val=-10000.0, max_val=10000.0, q_type=tf.qint32,
        ksize=[1, 2, 2, 1], strides=[1, 4, 4, 1], padding='VALID', name='test_qint32_strides'
    ))

    # Input 4: qint16, SAME padding, batch > 1
    list_of_inputs.append(create_quantized_dict_as_tensors(
        shape=(2, 6, 6, 4), min_val=-3000.0, max_val=3000.0, q_type=tf.qint16,
        ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='test_qint16_batch'
    ))

    # Input 5: quint16, large kernel
    list_of_inputs.append(create_quantized_dict_as_tensors(
        shape=(1, 10, 10, 1), min_val=0.0, max_val=60000.0, q_type=tf.quint16,
        ksize=[1, 5, 5, 1], strides=[1, 5, 5, 1], padding='VALID', name='test_quint16_large_kernel'
    ))

    # Input 6: qint8, non-square kernel and strides
    list_of_inputs.append(create_quantized_dict_as_tensors(
        shape=(1, 8, 8, 1), min_val=-50.0, max_val=50.0, q_type=tf.qint8,
        ksize=[1, 2, 3, 1], strides=[1, 1, 2, 1], padding='SAME', name='test_qint8_nonsquare'
    ))

    # Input 7: quint8, strides larger than ksize
    list_of_inputs.append(create_quantized_dict_as_tensors(
        shape=(1, 10, 10, 1), min_val=0.0, max_val=200.0, q_type=tf.quint8,
        ksize=[1, 2, 2, 1], strides=[1, 3, 3, 1], padding='VALID', name='test_quint8_large_strides'
    ))

    # Input 8: qint16, minimal input size with VALID padding
    list_of_inputs.append(create_quantized_dict_as_tensors(
        shape=(1, 2, 2, 1), min_val=0.0, max_val=50.0, q_type=tf.qint16,
        ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='VALID', name='test_qint16_minimal_valid'
    ))

    # Input 9: quint16, minimal input size with SAME padding
    list_of_inputs.append(create_quantized_dict_as_tensors(
        shape=(1, 2, 2, 1), min_val=0.0, max_val=5000.0, q_type=tf.quint16,
        ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='test_quint16_minimal_same'
    ))

    # Input 10: qint32, large tensor with SAME padding
    list_of_inputs.append(create_quantized_dict_as_tensors(
        shape=(1, 16, 16, 8), min_val=-20000.0, max_val=20000.0, q_type=tf.qint32,
        ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME', name='test_qint32_large_same'
    ))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.QuantizedAvgPool"] = get_quantized_avg_pool_inputs()

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

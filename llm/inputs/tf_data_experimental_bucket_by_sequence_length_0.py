
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_bucket_by_sequence_length_inputs():
    list_of_inputs = []

    # Input 1
    element_length_func = lambda elem: tf.shape(elem)[0]
    bucket_boundaries = [3, 5]
    bucket_batch_sizes = [2, 2, 2]
    padded_shapes = ()
    padding_values = np.int64(0)
    pad_to_bucket_boundary = False
    no_padding = False
    drop_remainder = False
    input_dict = {'element_length_func': element_length_func, 'bucket_boundaries': bucket_boundaries, 'bucket_batch_sizes': bucket_batch_sizes, 'padded_shapes': padded_shapes, 'padding_values': padding_values, 'pad_to_bucket_boundary': pad_to_bucket_boundary, 'no_padding': no_padding, 'drop_remainder': drop_remainder}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    element_length_func = lambda elem: tf.shape(elem)[0]
    bucket_boundaries = [4, 7]
    bucket_batch_sizes = [2, 2, 2]
    padded_shapes = ()
    padding_values = np.int32(-1)
    pad_to_bucket_boundary = True
    no_padding = False
    drop_remainder = True
    input_dict = {'element_length_func': element_length_func, 'bucket_boundaries': bucket_boundaries, 'bucket_batch_sizes': bucket_batch_sizes, 'padded_shapes': padded_shapes, 'padding_values': padding_values, 'pad_to_bucket_boundary': pad_to_bucket_boundary, 'no_padding': no_padding, 'drop_remainder': drop_remainder}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    element_length_func = lambda elem: tf.shape(elem)[0]
    bucket_boundaries = [2, 4, 6]
    bucket_batch_sizes = [1, 1, 1, 1]
    padded_shapes = ()
    padding_values = np.float32(100)
    pad_to_bucket_boundary = False
    no_padding = False
    drop_remainder = False
    input_dict = {'element_length_func': element_length_func, 'bucket_boundaries': bucket_boundaries, 'bucket_batch_sizes': bucket_batch_sizes, 'padded_shapes': padded_shapes, 'padding_values': padding_values, 'pad_to_bucket_boundary': pad_to_bucket_boundary, 'no_padding': no_padding, 'drop_remainder': drop_remainder}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    element_length_func = lambda elem: tf.shape(elem)[0]
    bucket_boundaries = [10]
    bucket_batch_sizes = [4, 4]
    padded_shapes = ()
    padding_values = np.int64(1)
    pad_to_bucket_boundary = True
    no_padding = False
    drop_remainder = True
    input_dict = {'element_length_func': element_length_func, 'bucket_boundaries': bucket_boundaries, 'bucket_batch_sizes': bucket_batch_sizes, 'padded_shapes': padded_shapes, 'padding_values': padding_values, 'pad_to_bucket_boundary': pad_to_bucket_boundary, 'no_padding': no_padding, 'drop_remainder': drop_remainder}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    element_length_func = lambda elem: tf.shape(elem)[0]
    bucket_boundaries = [5, 8, 10]
    bucket_batch_sizes = [3, 3, 3, 3]
    padded_shapes = ()
    padding_values = np.float64(0.0)
    pad_to_bucket_boundary = False
    no_padding = False
    drop_remainder = False
    input_dict = {'element_length_func': element_length_func, 'bucket_boundaries': bucket_boundaries, 'bucket_batch_sizes': bucket_batch_sizes, 'padded_shapes': padded_shapes, 'padding_values': padding_values, 'pad_to_bucket_boundary': pad_to_bucket_boundary, 'no_padding': no_padding, 'drop_remainder': drop_remainder}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    element_length_func = lambda elem: tf.shape(elem)[0]
    bucket_boundaries = [1, 2, 3, 4, 5]
    bucket_batch_sizes = [1, 1, 1, 1, 1, 1]
    padded_shapes = ()
    padding_values = np.int32(-99)
    pad_to_bucket_boundary = True
    no_padding = False
    drop_remainder = True
    input_dict = {'element_length_func': element_length_func, 'bucket_boundaries': bucket_boundaries, 'bucket_batch_sizes': bucket_batch_sizes, 'padded_shapes': padded_shapes, 'padding_values': padding_values, 'pad_to_bucket_boundary': pad_to_bucket_boundary, 'no_padding': no_padding, 'drop_remainder': drop_remainder}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    element_length_func = lambda elem: tf.shape(elem)[0]
    bucket_boundaries = [7, 9]
    bucket_batch_sizes = [5, 5, 5]
    padded_shapes = ()
    padding_values = np.int8(1)
    pad_to_bucket_boundary = False
    no_padding = True
    drop_remainder = False
    input_dict = {'element_length_func': element_length_func, 'bucket_boundaries': bucket_boundaries, 'bucket_batch_sizes': bucket_batch_sizes, 'padded_shapes': padded_shapes, 'padding_values': padding_values, 'pad_to_bucket_boundary': pad_to_bucket_boundary, 'no_padding': no_padding, 'drop_remainder': drop_remainder}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    element_length_func = lambda elem: tf.shape(elem)[0]
    bucket_boundaries = [20, 30, 40]
    bucket_batch_sizes = [10, 10, 10, 10]
    padded_shapes = ()
    padding_values = np.uint8(255)
    pad_to_bucket_boundary = True
    no_padding = True
    drop_remainder = True
    input_dict = {'element_length_func': element_length_func, 'bucket_boundaries': bucket_boundaries, 'bucket_batch_sizes': bucket_batch_sizes, 'padded_shapes': padded_shapes, 'padding_values': padding_values, 'pad_to_bucket_boundary': pad_to_bucket_boundary, 'no_padding': no_padding, 'drop_remainder': drop_remainder}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    element_length_func = lambda elem: tf.shape(elem)[0]
    bucket_boundaries = [15]
    bucket_batch_sizes = [7, 7]
    padded_shapes = ()
    padding_values = np.int16(0)
    pad_to_bucket_boundary = False
    no_padding = False
    drop_remainder = True
    input_dict = {'element_length_func': element_length_func, 'bucket_boundaries': bucket_boundaries, 'bucket_batch_sizes': bucket_batch_sizes, 'padded_shapes': padded_shapes, 'padding_values': padding_values, 'pad_to_bucket_boundary': pad_to_bucket_boundary, 'no_padding': no_padding, 'drop_remainder': drop_remainder}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    element_length_func = lambda elem: tf.shape(elem)[0]
    bucket_boundaries = [6, 12, 18, 24]
    bucket_batch_sizes = [4, 4, 4, 4, 4]
    padded_shapes = ()
    padding_values = np.int64(-1000)
    pad_to_bucket_boundary = True
    no_padding = False
    drop_remainder = False
    input_dict = {'element_length_func': element_length_func, 'bucket_boundaries': bucket_boundaries, 'bucket_batch_sizes': bucket_batch_sizes, 'padded_shapes': padded_shapes, 'padding_values': padding_values, 'pad_to_bucket_boundary': pad_to_bucket_boundary, 'no_padding': no_padding, 'drop_remainder': drop_remainder}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.data.experimental.bucket_by_sequence_length"] = tf_data_experimental_bucket_by_sequence_length_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.data.experimental.bucket_by_sequence_length' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.bucket_by_sequence_length'.")

check_valid('tf.data.experimental.bucket_by_sequence_length', generated_inputs['tf.data.experimental.bucket_by_sequence_length'], lib="tf", suffix=0)

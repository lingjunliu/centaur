
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_bucket_by_sequence_length_inputs():
    list_of_inputs = []

    # Input 1
    element_length_func = [1]
    bucket_boundaries = [5, 10]
    bucket_batch_sizes = [1, 1, 1]
    padded_shapes = (tf.TensorShape([None]),)
    padding_values = tf.convert_to_tensor(np.int64(0))
    pad_to_bucket_boundary = False
    no_padding = False
    drop_remainder = False
    input_dict = {"element_length_func": element_length_func, "bucket_boundaries": bucket_boundaries, "bucket_batch_sizes": bucket_batch_sizes, "padded_shapes": padded_shapes, "padding_values": padding_values, "pad_to_bucket_boundary": pad_to_bucket_boundary, "no_padding": no_padding, "drop_remainder": drop_remainder}
    list_of_inputs.append(input_dict)

    # Input 2
    element_length_func = [1]
    bucket_boundaries = [3, 7, 12]
    bucket_batch_sizes = [2, 2, 2, 2]
    padded_shapes = (tf.TensorShape([None]),)
    padding_values = tf.convert_to_tensor(np.int32(-1))
    pad_to_bucket_boundary = True
    no_padding = False
    drop_remainder = True
    input_dict = {"element_length_func": element_length_func, "bucket_boundaries": bucket_boundaries, "bucket_batch_sizes": bucket_batch_sizes, "padded_shapes": padded_shapes, "padding_values": padding_values, "pad_to_bucket_boundary": pad_to_bucket_boundary, "no_padding": no_padding, "drop_remainder": drop_remainder}
    list_of_inputs.append(input_dict)

    # Input 3
    element_length_func = [1]
    bucket_boundaries = [2, 4, 6, 8]
    bucket_batch_sizes = [3, 3, 3, 3, 3]
    padded_shapes = (tf.TensorShape([None]),)
    padding_values = tf.convert_to_tensor(np.float32(100))
    pad_to_bucket_boundary = False
    no_padding = True
    drop_remainder = False
    input_dict = {"element_length_func": element_length_func, "bucket_boundaries": bucket_boundaries, "bucket_batch_sizes": bucket_batch_sizes, "padded_shapes": padded_shapes, "padding_values": padding_values, "pad_to_bucket_boundary": pad_to_bucket_boundary, "no_padding": no_padding, "drop_remainder": drop_remainder}
    list_of_inputs.append(input_dict)

    # Input 4
    element_length_func = [1]
    bucket_boundaries = [1, 2]
    bucket_batch_sizes = [4, 4, 4]
    padded_shapes = (tf.TensorShape([None, None]),)
    padding_values = tf.convert_to_tensor(np.int64(99))
    pad_to_bucket_boundary = True
    no_padding = False
    drop_remainder = True
    input_dict = {"element_length_func": element_length_func, "bucket_boundaries": bucket_boundaries, "bucket_batch_sizes": bucket_batch_sizes, "padded_shapes": padded_shapes, "padding_values": padding_values, "pad_to_bucket_boundary": pad_to_bucket_boundary, "no_padding": no_padding, "drop_remainder": drop_remainder}
    list_of_inputs.append(input_dict)

    # Input 5
    element_length_func = [1]
    bucket_boundaries = [6]
    bucket_batch_sizes = [5, 5]
    padded_shapes = (tf.TensorShape([None, None, None]),)
    padding_values = tf.convert_to_tensor(np.int32(-10))
    pad_to_bucket_boundary = False
    no_padding = True
    drop_remainder = False
    input_dict = {"element_length_func": element_length_func, "bucket_boundaries": bucket_boundaries, "bucket_batch_sizes": bucket_batch_sizes, "padded_shapes": padded_shapes, "padding_values": padding_values, "pad_to_bucket_boundary": pad_to_bucket_boundary, "no_padding": no_padding, "drop_remainder": drop_remainder}
    list_of_inputs.append(input_dict)

    # Input 6
    element_length_func = [1]
    bucket_boundaries = [11, 15, 20]
    bucket_batch_sizes = [1, 1, 1, 1]
    padded_shapes = (tf.TensorShape([None]),)
    padding_values = tf.convert_to_tensor(np.int64(0))
    pad_to_bucket_boundary = True
    no_padding = False
    drop_remainder = True
    input_dict = {"element_length_func": element_length_func, "bucket_boundaries": bucket_boundaries, "bucket_batch_sizes": bucket_batch_sizes, "padded_shapes": padded_shapes, "padding_values": padding_values, "pad_to_bucket_boundary": pad_to_bucket_boundary, "no_padding": no_padding, "drop_remainder": drop_remainder}
    list_of_inputs.append(input_dict)

    # Input 7
    element_length_func = [1]
    bucket_boundaries = [1, 5, 9, 13, 17]
    bucket_batch_sizes = [2, 2, 2, 2, 2, 2]
    padded_shapes = (tf.TensorShape([None]),)
    padding_values = tf.convert_to_tensor(np.float32(10))
    pad_to_bucket_boundary = False
    no_padding = True
    drop_remainder = False
    input_dict = {"element_length_func": element_length_func, "bucket_boundaries": bucket_boundaries, "bucket_batch_sizes": bucket_batch_sizes, "padded_shapes": padded_shapes, "padding_values": padding_values, "pad_to_bucket_boundary": pad_to_bucket_boundary, "no_padding": no_padding, "drop_remainder": drop_remainder}
    list_of_inputs.append(input_dict)

   # Input 8
    element_length_func = [1]
    bucket_boundaries = [2, 6]
    bucket_batch_sizes = [3, 3, 3]
    padded_shapes = (tf.TensorShape([None, None, None]),)
    padding_values = tf.convert_to_tensor(np.int32(5))
    pad_to_bucket_boundary = True
    no_padding = False
    drop_remainder = True
    input_dict = {"element_length_func": element_length_func, "bucket_boundaries": bucket_boundaries, "bucket_batch_sizes": bucket_batch_sizes, "padded_shapes": padded_shapes, "padding_values": padding_values, "pad_to_bucket_boundary": pad_to_bucket_boundary, "no_padding": no_padding, "drop_remainder": drop_remainder}
    list_of_inputs.append(input_dict)

    # Input 9
    element_length_func = [1]
    bucket_boundaries = [4, 8, 12, 16]
    bucket_batch_sizes = [4, 4, 4, 4, 4]
    padded_shapes = (tf.TensorShape([None, None]),)
    padding_values = tf.convert_to_tensor(np.int64(-1))
    pad_to_bucket_boundary = False
    no_padding = False
    drop_remainder = False
    input_dict = {"element_length_func": element_length_func, "bucket_boundaries": bucket_boundaries, "bucket_batch_sizes": bucket_batch_sizes, "padded_shapes": padded_shapes, "padding_values": padding_values, "pad_to_bucket_boundary": pad_to_bucket_boundary, "no_padding": no_padding, "drop_remainder": drop_remainder}
    list_of_inputs.append(input_dict)

    # Input 10
    element_length_func = [1]
    bucket_boundaries = [3, 9, 15]
    bucket_batch_sizes = [5, 5, 5, 5]
    padded_shapes = (tf.TensorShape([None]),)
    padding_values = tf.convert_to_tensor(np.float32(-5))
    pad_to_bucket_boundary = True
    no_padding = True
    drop_remainder = True
    input_dict = {"element_length_func": element_length_func, "bucket_boundaries": bucket_boundaries, "bucket_batch_sizes": bucket_batch_sizes, "padded_shapes": padded_shapes, "padding_values": padding_values, "pad_to_bucket_boundary": pad_to_bucket_boundary, "no_padding": no_padding, "drop_remainder": drop_remainder}
    list_of_inputs.append(input_dict)

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

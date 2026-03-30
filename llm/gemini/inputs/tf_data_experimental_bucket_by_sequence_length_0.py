
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_bucket_by_sequence_length_inputs():
    list_of_inputs = []

    # Input 1
    def element_length_func(elem):
        return np.int32(tf.shape(elem)[0])
    bucket_boundaries = [5, 10, 15]
    bucket_batch_sizes = [1, 2, 3, 4]
    padded_shapes = (tf.TensorShape([None]),)
    padding_values = tf.constant(0, dtype=tf.int64)
    pad_to_bucket_boundary = False
    no_padding = False
    drop_remainder = False

    input_dict = {
        "element_length_func": element_length_func,
        "bucket_boundaries": bucket_boundaries,
        "bucket_batch_sizes": bucket_batch_sizes,
        "padded_shapes": padded_shapes,
        "padding_values": padding_values,
        "pad_to_bucket_boundary": pad_to_bucket_boundary,
        "no_padding": no_padding,
        "drop_remainder": drop_remainder
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    def element_length_func(elem):
        return np.int32(tf.shape(elem)[0])
    bucket_boundaries = [2, 4]
    bucket_batch_sizes = [4, 2, 1]
    padded_shapes = (tf.TensorShape([None]),)
    padding_values = tf.constant(-1, dtype=tf.int32)
    pad_to_bucket_boundary = True
    no_padding = False
    drop_remainder = True

    input_dict = {
        "element_length_func": element_length_func,
        "bucket_boundaries": bucket_boundaries,
        "bucket_batch_sizes": bucket_batch_sizes,
        "padded_shapes": padded_shapes,
        "padding_values": padding_values,
        "pad_to_bucket_boundary": pad_to_bucket_boundary,
        "no_padding": no_padding,
        "drop_remainder": drop_remainder
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    def element_length_func(elem):
        return np.int32(tf.shape(elem)[0])
    bucket_boundaries = [7]
    bucket_batch_sizes = [2, 3]
    padded_shapes = (tf.TensorShape([None]),)
    padding_values = tf.constant(100, dtype=tf.float32)
    pad_to_bucket_boundary = False
    no_padding = False
    drop_remainder = False

    input_dict = {
        "element_length_func": element_length_func,
        "bucket_boundaries": bucket_boundaries,
        "bucket_batch_sizes": bucket_batch_sizes,
        "padded_shapes": padded_shapes,
        "padding_values": padding_values,
        "pad_to_bucket_boundary": pad_to_bucket_boundary,
        "no_padding": no_padding,
        "drop_remainder": drop_remainder
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 4
    def element_length_func(elem):
        return np.int32(tf.shape(elem)[0])
    bucket_boundaries = [10, 20, 30]
    bucket_batch_sizes = [2, 1, 4, 2]
    padded_shapes = (tf.TensorShape([None]),)
    padding_values = tf.constant(5, dtype=tf.int64)
    pad_to_bucket_boundary = True
    no_padding = False
    drop_remainder = True

    input_dict = {
        "element_length_func": element_length_func,
        "bucket_boundaries": bucket_boundaries,
        "bucket_batch_sizes": bucket_batch_sizes,
        "padded_shapes": padded_shapes,
        "padding_values": padding_values,
        "pad_to_bucket_boundary": pad_to_bucket_boundary,
        "no_padding": no_padding,
        "drop_remainder": drop_remainder
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    def element_length_func(elem):
        return np.int32(tf.shape(elem)[0])
    bucket_boundaries = [6, 12]
    bucket_batch_sizes = [3, 2, 5]
    padded_shapes = (tf.TensorShape([None]),)
    padding_values = tf.constant(-1.0, dtype=tf.float64)
    pad_to_bucket_boundary = False
    no_padding = True
    drop_remainder = False

    input_dict = {
        "element_length_func": element_length_func,
        "bucket_boundaries": bucket_boundaries,
        "bucket_batch_sizes": bucket_batch_sizes,
        "padded_shapes": padded_shapes,
        "padding_values": padding_values,
        "pad_to_bucket_boundary": pad_to_bucket_boundary,
        "no_padding": no_padding,
        "drop_remainder": drop_remainder
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    def element_length_func(elem):
        return np.int32(tf.shape(elem)[0])
    bucket_boundaries = [3]
    bucket_batch_sizes = [1, 1]
    padded_shapes = (tf.TensorShape([None]),)
    padding_values = tf.constant(0, dtype=tf.int64)
    pad_to_bucket_boundary = False
    no_padding = False
    drop_remainder = False

    input_dict = {
        "element_length_func": element_length_func,
        "bucket_boundaries": bucket_boundaries,
        "bucket_batch_sizes": bucket_batch_sizes,
        "padded_shapes": padded_shapes,
        "padding_values": padding_values,
        "pad_to_bucket_boundary": pad_to_bucket_boundary,
        "no_padding": no_padding,
        "drop_remainder": drop_remainder
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    def element_length_func(elem):
        return np.int32(tf.shape(elem)[0])
    bucket_boundaries = [4, 8, 12, 16]
    bucket_batch_sizes = [1, 2, 3, 4, 5]
    padded_shapes = (tf.TensorShape([None]),)
    padding_values = tf.constant(0, dtype=tf.int64)
    pad_to_bucket_boundary = True
    no_padding = False
    drop_remainder = True

    input_dict = {
        "element_length_func": element_length_func,
        "bucket_boundaries": bucket_boundaries,
        "bucket_batch_sizes": bucket_batch_sizes,
        "padded_shapes": padded_shapes,
        "padding_values": padding_values,
        "pad_to_bucket_boundary": pad_to_bucket_boundary,
        "no_padding": no_padding,
        "drop_remainder": drop_remainder
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    def element_length_func(elem):
        return np.int32(tf.shape(elem)[0])
    bucket_boundaries = [2, 4, 6]
    bucket_batch_sizes = [4, 3, 2, 1]
    padded_shapes = (tf.TensorShape([None]),)
    padding_values = tf.constant(0, dtype=tf.int64)
    pad_to_bucket_boundary = False
    no_padding = False
    drop_remainder = False

    input_dict = {
        "element_length_func": element_length_func,
        "bucket_boundaries": bucket_boundaries,
        "bucket_batch_sizes": bucket_batch_sizes,
        "padded_shapes": padded_shapes,
        "padding_values": padding_values,
        "pad_to_bucket_boundary": pad_to_bucket_boundary,
        "no_padding": no_padding,
        "drop_remainder": drop_remainder
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    def element_length_func(elem):
        return np.int32(tf.shape(elem)[0])
    bucket_boundaries = [5]
    bucket_batch_sizes = [2, 1]
    padded_shapes = (tf.TensorShape([None]),)
    padding_values = tf.constant(-1, dtype=tf.int64)
    pad_to_bucket_boundary = True
    no_padding = False
    drop_remainder = True

    input_dict = {
        "element_length_func": element_length_func,
        "bucket_boundaries": bucket_boundaries,
        "bucket_batch_sizes": bucket_batch_sizes,
        "padded_shapes": padded_shapes,
        "padding_values": padding_values,
        "pad_to_bucket_boundary": pad_to_bucket_boundary,
        "no_padding": no_padding,
        "drop_remainder": drop_remainder
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    def element_length_func(elem):
        return np.int32(tf.shape(elem)[0])
    bucket_boundaries = [1, 2, 3, 4, 5]
    bucket_batch_sizes = [6, 5, 4, 3, 2, 1]
    padded_shapes = (tf.TensorShape([None]),)
    padding_values = tf.constant(10, dtype=tf.int64)
    pad_to_bucket_boundary = False
    no_padding = True
    drop_remainder = False

    input_dict = {
        "element_length_func": element_length_func,
        "bucket_boundaries": bucket_boundaries,
        "bucket_batch_sizes": bucket_batch_sizes,
        "padded_shapes": padded_shapes,
        "padding_values": padding_values,
        "pad_to_bucket_boundary": pad_to_bucket_boundary,
        "no_padding": no_padding,
        "drop_remainder": drop_remainder
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.data.experimental.bucket_by_sequence_length"] = tf_data_experimental_bucket_by_sequence_length_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.data.experimental.bucket_by_sequence_length' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.bucket_by_sequence_length'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.data.experimental.bucket_by_sequence_length', generated_inputs['tf.data.experimental.bucket_by_sequence_length'], lib="tf", suffix=0)

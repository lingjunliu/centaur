
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_bucket_by_sequence_length_inputs():
    list_of_inputs = []

    def element_length_func1(elem):
        return int(tf.shape(elem)[0])

    # Input 1
    bucket_boundaries = [3, 5]
    bucket_batch_sizes = [2, 2, 2]
    padded_shapes = (tf.TensorShape([None]),)
    padding_values = tf.constant(0, dtype=tf.int64)
    pad_to_bucket_boundary = False
    no_padding = False
    drop_remainder = False

    input_dict = {
        "element_length_func": [element_length_func1],
        "bucket_boundaries": bucket_boundaries,
        "bucket_batch_sizes": bucket_batch_sizes,
        "padded_shapes": padded_shapes,
        "padding_values": padding_values,
        "pad_to_bucket_boundary": pad_to_bucket_boundary,
        "no_padding": no_padding,
        "drop_remainder": drop_remainder
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    def element_length_func2(elem):
        return int(tf.shape(elem)[0])

    # Input 2
    bucket_boundaries = [4, 7]
    bucket_batch_sizes = [2, 2, 2]
    padded_shapes = (tf.TensorShape([None]),)
    padding_values = tf.constant(-1, dtype=tf.int32)
    pad_to_bucket_boundary = True
    no_padding = False
    drop_remainder = True

    input_dict = {
        "element_length_func": [element_length_func2],
        "bucket_boundaries": bucket_boundaries,
        "bucket_batch_sizes": bucket_batch_sizes,
        "padded_shapes": padded_shapes,
        "padding_values": padding_values,
        "pad_to_bucket_boundary": pad_to_bucket_boundary,
        "no_padding": no_padding,
        "drop_remainder": drop_remainder
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    def element_length_func3(elem):
        return int(tf.shape(elem)[0])

    # Input 3
    bucket_boundaries = [2, 4, 6]
    bucket_batch_sizes = [1, 1, 1, 1]
    padded_shapes = (tf.TensorShape([None]),)
    padding_values = tf.constant(100, dtype=tf.float32)
    pad_to_bucket_boundary = False
    no_padding = False
    drop_remainder = False

    input_dict = {
        "element_length_func": [element_length_func3],
        "bucket_boundaries": bucket_boundaries,
        "bucket_batch_sizes": bucket_batch_sizes,
        "padded_shapes": padded_shapes,
        "padding_values": padding_values,
        "pad_to_bucket_boundary": pad_to_bucket_boundary,
        "no_padding": False,
        "drop_remainder": drop_remainder
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    def element_length_func4(elem):
        return int(tf.shape(elem)[0])

   # Input 4
    bucket_boundaries = [2, 4, 6]
    bucket_batch_sizes = [1, 1, 1, 1]
    padded_shapes = (tf.TensorShape([None]),)
    padding_values = tf.constant(100, dtype=tf.float32)
    pad_to_bucket_boundary = True
    no_padding = False
    drop_remainder = False

    input_dict = {
        "element_length_func": [element_length_func4],
        "bucket_boundaries": bucket_boundaries,
        "bucket_batch_sizes": bucket_batch_sizes,
        "padded_shapes": padded_shapes,
        "padding_values": padding_values,
        "pad_to_bucket_boundary": pad_to_bucket_boundary,
        "no_padding": False,
        "drop_remainder": drop_remainder
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    def element_length_func5(elem):
        return int(tf.shape(elem)[0])

    # Input 5
    bucket_boundaries = [5, 10]
    bucket_batch_sizes = [4, 4, 4]
    padded_shapes = (tf.TensorShape([None]),)
    padding_values = tf.constant(-1, dtype=tf.int32)
    pad_to_bucket_boundary = False
    no_padding = True
    drop_remainder = False

    input_dict = {
        "element_length_func": [element_length_func5],
        "bucket_boundaries": bucket_boundaries,
        "bucket_batch_sizes": bucket_batch_sizes,
        "padded_shapes": padded_shapes,
        "padding_values": padding_values,
        "pad_to_bucket_boundary": pad_to_bucket_boundary,
        "no_padding": True,
        "drop_remainder": drop_remainder
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    def element_length_func6(elem):
        return int(tf.shape(elem)[0])

    # Input 6
    bucket_boundaries = [5, 10]
    bucket_batch_sizes = [4, 4, 4]
    padded_shapes = (tf.TensorShape([None]),)
    padding_values = tf.constant(-1, dtype=tf.int32)
    pad_to_bucket_boundary = True
    no_padding = True
    drop_remainder = True

    input_dict = {
        "element_length_func": [element_length_func6],
        "bucket_boundaries": bucket_boundaries,
        "bucket_batch_sizes": bucket_batch_sizes,
        "padded_shapes": padded_shapes,
        "padding_values": padding_values,
        "pad_to_bucket_boundary": pad_to_bucket_boundary,
        "no_padding": True,
        "drop_remainder": drop_remainder
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    def element_length_func7(elem):
        return int(tf.shape(elem)[0])

    # Input 7: Different padding value type
    bucket_boundaries = [3, 6, 9]
    bucket_batch_sizes = [2, 2, 2, 2]
    padded_shapes = (tf.TensorShape([None]),)
    padding_values = tf.constant(0.0, dtype=tf.float64)
    pad_to_bucket_boundary = False
    no_padding = False
    drop_remainder = False

    input_dict = {
        "element_length_func": [element_length_func7],
        "bucket_boundaries": bucket_boundaries,
        "bucket_batch_sizes": bucket_batch_sizes,
        "padded_shapes": padded_shapes,
        "padding_values": padding_values,
        "pad_to_bucket_boundary": pad_to_bucket_boundary,
        "no_padding": no_padding,
        "drop_remainder": drop_remainder
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    def element_length_func8(elem):
        return int(tf.shape(elem)[0])

    # Input 8: More bucket boundaries
    bucket_boundaries = [1, 2, 3, 4, 5]
    bucket_batch_sizes = [1, 1, 1, 1, 1, 1]
    padded_shapes = (tf.TensorShape([None]),)
    padding_values = tf.constant(1, dtype=tf.int32)
    pad_to_bucket_boundary = True
    no_padding = False
    drop_remainder = True

    input_dict = {
        "element_length_func": [element_length_func8],
        "bucket_boundaries": bucket_boundaries,
        "bucket_batch_sizes": bucket_batch_sizes,
        "padded_shapes": padded_shapes,
        "padding_values": padding_values,
        "pad_to_bucket_boundary": pad_to_bucket_boundary,
        "no_padding": no_padding,
        "drop_remainder": drop_remainder
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    def element_length_func9(elem):
        return int(tf.shape(elem)[0])

    # Input 9: One bucket
    bucket_boundaries = [10]
    bucket_batch_sizes = [2, 2]
    padded_shapes = (tf.TensorShape([None]),)
    padding_values = tf.constant(-1, dtype=tf.int32)
    pad_to_bucket_boundary = False
    no_padding = False
    drop_remainder = False

    input_dict = {
        "element_length_func": [element_length_func9],
        "bucket_boundaries": bucket_boundaries,
        "bucket_batch_sizes": bucket_batch_sizes,
        "padded_shapes": padded_shapes,
        "padding_values": padding_values,
        "pad_to_bucket_boundary": pad_to_bucket_boundary,
        "no_padding": False,
        "drop_remainder": drop_remainder
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    def element_length_func10(elem):
        return int(tf.shape(elem)[0])

    # Input 10
    bucket_boundaries = [4, 7]
    bucket_batch_sizes = [2, 2, 2]
    padded_shapes = (tf.TensorShape([None]),)
    padding_values = tf.constant(-1, dtype=tf.int64)
    pad_to_bucket_boundary = False
    no_padding = False
    drop_remainder = False

    input_dict = {
        "element_length_func": [element_length_func10],
        "bucket_boundaries": bucket_boundaries,
        "bucket_batch_sizes": bucket_batch_sizes,
        "padded_shapes": padded_shapes,
        "padding_values": padding_values,
        "pad_to_bucket_boundary": pad_to_bucket_boundary,
        "no_padding": False,
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
    
    print("Valid")

if 'tf.data.experimental.bucket_by_sequence_length' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.bucket_by_sequence_length'.")

check_valid('tf.data.experimental.bucket_by_sequence_length', generated_inputs['tf.data.experimental.bucket_by_sequence_length'], lib="tf", suffix=0)

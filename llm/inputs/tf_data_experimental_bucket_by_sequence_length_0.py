
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def get_bucket_by_sequence_length_inputs():
    """
    Generates a list of valid inputs for the tf.data.experimental.bucket_by_sequence_length function.
    """
    list_of_inputs = []

    # The API requires a callable for element_length_func. The 'list' type in the
    # prompt's signature is likely an error and is being overridden.
    element_length_func = lambda elem: tf.shape(elem)[0]

    # The test harness requires a 'dataset' key to apply the transformation function to.
    def create_dataset(gen_fn, spec):
        return tf.data.Dataset.from_generator(gen_fn, output_signature=spec)

    elements_1d = [
      [0], [1, 2, 3, 4], [5, 6, 7],
      [7, 8, 9, 10, 11], [13, 14, 15, 16, 19, 20], [21, 22]
    ]

    # Input 1: Basic case
    list_of_inputs.append({
        'dataset': create_dataset(lambda: elements_1d, tf.TensorSpec(shape=(None,), dtype=tf.int32)),
        'element_length_func': element_length_func,
        'bucket_boundaries': [3, 5],
        'bucket_batch_sizes': [2, 2, 2],
        'padded_shapes': (tf.TensorShape([None]),),
        'padding_values': np.array(0, dtype=np.int32),
        'pad_to_bucket_boundary': False,
        'no_padding': False,
        'drop_remainder': False
    })

    # Input 2: Using pad_to_bucket_boundary and a custom integer padding_value.
    list_of_inputs.append({
        'dataset': create_dataset(lambda: elements_1d, tf.TensorSpec(shape=(None,), dtype=tf.int32)),
        'element_length_func': element_length_func,
        'bucket_boundaries': [4, 7],
        'bucket_batch_sizes': [2, 2, 2],
        'padded_shapes': (tf.TensorShape([6]),),
        'padding_values': np.array(-1, dtype=np.int32),
        'pad_to_bucket_boundary': True,
        'no_padding': False,
        'drop_remainder': False
    })

    # Input 3: Using drop_remainder along with pad_to_bucket_boundary.
    list_of_inputs.append({
        'dataset': create_dataset(lambda: elements_1d, tf.TensorSpec(shape=(None,), dtype=tf.int32)),
        'element_length_func': element_length_func,
        'bucket_boundaries': [4, 8],
        'bucket_batch_sizes': [2, 2, 2],
        'padded_shapes': (tf.TensorShape([7]),),
        'padding_values': np.array(-1, dtype=np.int32),
        'pad_to_bucket_boundary': True,
        'no_padding': False,
        'drop_remainder': True
    })

    # Input 4: Single bucket with float data.
    list_of_inputs.append({
        'dataset': create_dataset(lambda: elements_1d, tf.TensorSpec(shape=(None,), dtype=tf.float32)),
        'element_length_func': element_length_func,
        'bucket_boundaries': [],
        'bucket_batch_sizes': [10],
        'padded_shapes': (tf.TensorShape([None]),),
        'padding_values': np.array(0.0, dtype=np.float32),
        'pad_to_bucket_boundary': False,
        'no_padding': False,
        'drop_remainder': False
    })

    # Input 5: Many buckets with varying batch sizes.
    list_of_inputs.append({
        'dataset': create_dataset(lambda: elements_1d, tf.TensorSpec(shape=(None,), dtype=tf.int64)),
        'element_length_func': element_length_func,
        'bucket_boundaries': [2, 3, 4, 5, 6],
        'bucket_batch_sizes': [2, 2, 2, 2, 2, 2],
        'padded_shapes': (tf.TensorShape([6]),),
        'padding_values': np.array(0, dtype=np.int64),
        'pad_to_bucket_boundary': False,
        'no_padding': False,
        'drop_remainder': True
    })

    # Input 6: no_padding=True with fixed-shape elements.
    elements_fixed = [[1,2],[3,4],[5,6]]
    list_of_inputs.append({
        'dataset': create_dataset(lambda: elements_fixed, tf.TensorSpec(shape=(2,), dtype=tf.int32)),
        'element_length_func': lambda elem: tf.shape(elem)[0],
        'bucket_boundaries': [3],
        'bucket_batch_sizes': [3, 3],
        'padded_shapes': (tf.TensorShape([2]),),
        'padding_values': np.array(0, dtype=np.int32),
        'pad_to_bucket_boundary': False,
        'no_padding': True,
        'drop_remainder': False
    })

    # Input 7: Dataset with 2-D Tensors
    elements_2d = [np.ones((2,5), dtype=np.float32), np.ones((4,5), dtype=np.float32), np.ones((1,5), dtype=np.float32)]
    list_of_inputs.append({
        'dataset': create_dataset(lambda: elements_2d, tf.TensorSpec(shape=(None, 5), dtype=tf.float32)),
        'element_length_func': lambda elem: tf.shape(elem)[0],
        'bucket_boundaries': [3],
        'bucket_batch_sizes': [2, 2],
        'padded_shapes': (tf.TensorShape([None, 5]),),
        'padding_values': np.array(0.0, dtype=np.float32),
        'pad_to_bucket_boundary': False,
        'no_padding': False,
        'drop_remainder': False
    })

    # Input 8: Another pad_to_bucket_boundary case with float64.
    list_of_inputs.append({
        'dataset': create_dataset(lambda: elements_1d, tf.TensorSpec(shape=(None,), dtype=tf.float64)),
        'element_length_func': element_length_func,
        'bucket_boundaries': [5, 10],
        'bucket_batch_sizes': [4, 2, 1],
        'padded_shapes': (tf.TensorShape([9]),),
        'padding_values': np.array(-1.0, dtype=np.float64),
        'pad_to_bucket_boundary': True,
        'no_padding': False,
        'drop_remainder': False
    })

    return list_of_inputs

generated_inputs["tf.data.experimental.bucket_by_sequence_length"] = get_bucket_by_sequence_length_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.data.experimental.bucket_by_sequence_length' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.bucket_by_sequence_length'.")

check_valid('tf.data.experimental.bucket_by_sequence_length', generated_inputs['tf.data.experimental.bucket_by_sequence_length'], lib="tf", suffix=0)

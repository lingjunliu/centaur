
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def _creator_div_by_zero():
    data = np.array([1., 2., 0., 4.], dtype=np.float32)
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.map(lambda x: tf.debugging.check_numerics(1. / x, "error"))
    return dataset

def _creator_parsing_error():
    data = np.array(["1.0", "two", "3.0", "four"])
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.map(tf.strings.to_number)
    return dataset

def _creator_shape_error():
    def shape_error_generator():
        yield np.array([1, 2], dtype=np.int32)
        yield np.array([3, 4, 5], dtype=np.int32)
        yield np.array([6, 7], dtype=np.int32)
    dataset = tf.data.Dataset.from_generator(
        shape_error_generator,
        output_signature=tf.TensorSpec(shape=(None,), dtype=tf.int32)
    )
    dataset = dataset.map(lambda x: tf.reshape(x, (2,)))
    return dataset

def _creator_tf_assert_error():
    def error_fn(x):
        tf.Assert(tf.less(x, 15), [f"Element {x} is not less than 15"])
        return x
    data = np.array([5, 10, 15, 20], dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.map(error_fn)
    return dataset

def _creator_dtype_error():
    def mixed_type_generator():
        yield 1
        yield "not_an_int"
        yield 3
    dataset = tf.data.Dataset.from_generator(
        mixed_type_generator,
        output_signature=tf.TensorSpec(shape=(), dtype=tf.int32)
    )
    return dataset

def tf_data_experimental_ignore_errors_inputs():
    list_of_inputs = []
    dataset_creators = [
        _creator_div_by_zero,
        _creator_parsing_error,
        _creator_shape_error,
        _creator_tf_assert_error,
        _creator_dtype_error,
    ]
    log_warnings = [False, True]
    for creator_func in dataset_creators:
        for log_warning_val in log_warnings:
            input_dict = {
                'log_warning': log_warning_val,
                'inner_values': {
                    'dataset': creator_func
                }
            }
            list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs

generated_inputs["tf.data.experimental.ignore_errors"] = tf_data_experimental_ignore_errors_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.data.experimental.ignore_errors' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.ignore_errors'.")

check_valid('tf.data.experimental.ignore_errors', generated_inputs['tf.data.experimental.ignore_errors'], lib="tf", suffix=0)

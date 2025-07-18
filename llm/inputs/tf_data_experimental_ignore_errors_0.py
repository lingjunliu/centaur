
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf

def map_div_by_zero_float(x):
    return 1.0 / x

def map_div_by_zero_int(x):
    return 1 // x

def map_assert_positive(x):
    tf.Assert(x > 0, [x])
    return x

def map_string_to_number(x):
    return tf.strings.to_number(x)

def map_string_to_int(x):
    return tf.strings.to_number(x, out_type=tf.int32)

def map_sqrt(x):
    return tf.sqrt(x)

def map_tuple_div(x, y):
    return y // x

def map_square(x):
    return x * x

def map_assert_less_than_10(x):
    tf.Assert(x < 10, [x])
    return x

def tf_data_experimental_ignore_errors_inputs():
    list_of_inputs = []

    # Case 1: Float division by zero, log warnings
    input_dict_1 = {
        'log_warning': True,
        'inner_values': {
            'tensors': np.array([1., 2., 0., 4.], dtype=np.float32),
            'map_fn': map_div_by_zero_float
        }
    }
    list_of_inputs.append(input_dict_1)

    # Case 2: Integer division by zero, don't log warnings
    input_dict_2 = {
        'log_warning': False,
        'inner_values': {
            'tensors': np.array([5, 2, 0, 1], dtype=np.int32),
            'map_fn': map_div_by_zero_int
        }
    }
    list_of_inputs.append(input_dict_2)

    # Case 3: tf.Assert failure, log warnings
    input_dict_3 = {
        'log_warning': True,
        'inner_values': {
            'tensors': np.array([1, 2, -1, 4], dtype=np.int64),
            'map_fn': map_assert_positive
        }
    }
    list_of_inputs.append(input_dict_3)

    # Case 4: String to number conversion error, don't log warnings
    input_dict_4 = {
        'log_warning': False,
        'inner_values': {
            'tensors': np.array(["1.0", "hello", "3.0"]),
            'map_fn': map_string_to_number
        }
    }
    list_of_inputs.append(input_dict_4)

    # Case 5: Square root of negative number, log warnings
    input_dict_5 = {
        'log_warning': True,
        'inner_values': {
            'tensors': np.array([4.0, 9.0, -1.0, 16.0], dtype=np.float32),
            'map_fn': map_sqrt
        }
    }
    list_of_inputs.append(input_dict_5)

    # Case 6: Tuple of tensors as input, one causes error. Don't log.
    input_dict_6 = {
        'log_warning': False,
        'inner_values': {
            'tensors': (np.array([1, 2, 0]), np.array([10, 20, 30])),
            'map_fn': map_tuple_div
        }
    }
    list_of_inputs.append(input_dict_6)

    # Case 7: A dataset with no errors. log_warning=True.
    input_dict_7 = {
        'log_warning': True,
        'inner_values': {
            'tensors': np.array([1, 2, 3, 4], dtype=np.int32),
            'map_fn': map_square
        }
    }
    list_of_inputs.append(input_dict_7)
    
    # Case 8: A different string error. log_warning=False.
    input_dict_8 = {
        'log_warning': False,
        'inner_values': {
            'tensors': np.array(["1", "2", "inf", "4"]),
            'map_fn': map_string_to_int
        }
    }
    list_of_inputs.append(input_dict_8)

    # Case 9: float64 data type with error. log_warning=True.
    input_dict_9 = {
        'log_warning': True,
        'inner_values': {
            'tensors': np.array([10., -5., 0., 2.], dtype=np.float64),
            'map_fn': map_div_by_zero_float
        }
    }
    list_of_inputs.append(input_dict_9)

    # Case 10: Another assert failure. log_warning=False
    input_dict_10 = {
        'log_warning': False,
        'inner_values': {
            'tensors': np.array([1, 5, 12, 8], dtype=np.int32),
            'map_fn': map_assert_less_than_10
        }
    }
    list_of_inputs.append(input_dict_10)

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

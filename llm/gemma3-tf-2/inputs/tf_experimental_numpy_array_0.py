
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_experimental_numpy_array_inputs():
    list_of_inputs = []

    input_1 = {
        "val": np.array([1, 2, 3]),
        "dtype": np.int32,
        "copy": True,
        "ndmin": 1
    }
    list_of_inputs.append(copy.deepcopy(input_1))

    input_2 = {
        "val": np.array([[1, 2], [3, 4]]),
        "dtype": np.float64,
        "copy": False,
        "ndmin": 2
    }
    list_of_inputs.append(copy.deepcopy(input_2))

    input_3 = {
        "val": np.array([[[1, 2, 3], [4, 5, 6]]]),
        "dtype": np.int16,
        "copy": True,
        "ndmin": 3
    }
    list_of_inputs.append(copy.deepcopy(input_3))

    input_4 = {
        "val": np.array([-1, -2, -3]),
        "dtype": np.int8,
        "copy": False,
        "ndmin": 1
    }
    list_of_inputs.append(copy.deepcopy(input_4))

    input_5 = {
        "val": np.array([1.1, 2.2, 3.3]),
        "dtype": np.float32,
        "copy": True,
        "ndmin": 1
    }
    list_of_inputs.append(copy.deepcopy(input_5))

    input_6 = {
        "val": np.array([[1, 2], [3, 4], [5, 6]]),
        "dtype": np.int64,
        "copy": False,
        "ndmin": 2
    }
    list_of_inputs.append(copy.deepcopy(input_6))

    input_7 = {
        "val": np.array([1, 2, 3, 4, 5]),
        "dtype": np.uint8,
        "copy": True,
        "ndmin": 1
    }
    list_of_inputs.append(copy.deepcopy(input_7))

    input_8 = {
        "val": np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
        "dtype": np.float16,
        "copy": False,
        "ndmin": 3
    }
    list_of_inputs.append(copy.deepcopy(input_8))

    input_9 = {
        "val": np.array([1, 2, 3]),
        "dtype": np.int32,
        "copy": True,
        "ndmin": 0
    }
    list_of_inputs.append(copy.deepcopy(input_9))

    input_10 = {
        "val": np.array([[1, 2, 3], [4, 5, 6]]),
        "dtype": np.complex64,
        "copy": False,
        "ndmin": 2
    }
    list_of_inputs.append(copy.deepcopy(input_10))

    return list_of_inputs

generated_inputs["tf.experimental.numpy.array"] = tf_experimental_numpy_array_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.numpy.array' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.array'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.experimental.numpy.array', generated_inputs['tf.experimental.numpy.array'], lib="tf", suffix=0)

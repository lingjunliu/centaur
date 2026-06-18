
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def tf_math_argmin_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, axis 0
    input_dict = {
        "input": np.array([1.0, 10.0, 26.9, 2.8, 166.32, 62.3], dtype=np.float32),
        "axis": np.array(0, dtype=np.int32),
        "output_type": np.int64,
        "name": "argmin_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float64 array, axis 1
    input_dict = {
        "input": np.array([[2.0, 3.0, 1.0], [5.0, -1.0, 4.0]], dtype=np.float64),
        "axis": np.array(1, dtype=np.int64),
        "output_type": np.int32,
        "name": "argmin_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D int32 array with negative values, axis -1
    input_dict = {
        "input": np.array([[[-1, -2], [3, 4]], [[5, -6], [7, 8]]], dtype=np.int32),
        "axis": np.array(-1, dtype=np.int32),
        "output_type": np.int64,
        "name": "argmin_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D float32 array, axis 0
    input_dict = {
        "input": np.array([10.0, 20.0, 5.0, 40.0], dtype=np.float32),
        "axis": np.array(0, dtype=np.int32),
        "output_type": np.int32,
        "name": "argmin_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 4D random float32 array, axis 2
    input_dict = {
        "input": np.random.randn(2, 3, 4, 5).astype(np.float32),
        "axis": np.array(2, dtype=np.int64),
        "output_type": np.int64,
        "name": "argmin_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D int32 array, axis -2
    input_dict = {
        "input": np.array([[10, 20], [30, 40]], dtype=np.int32),
        "axis": np.array(-2, dtype=np.int32),
        "output_type": np.int32,
        "name": "argmin_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D float64 array with duplicates to test smallest index tie breaker
    input_dict = {
        "input": np.array([5.0, 2.0, 2.0, 8.0], dtype=np.float64),
        "axis": np.array(0, dtype=np.int64),
        "output_type": np.int64,
        "name": "argmin_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D int32 array, axis 1
    input_dict = {
        "input": np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32),
        "axis": np.array(1, dtype=np.int32),
        "output_type": np.int32,
        "name": "argmin_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D float32 array, axis 0
    input_dict = {
        "input": np.array([[1.5, 2.5], [0.5, 3.5]], dtype=np.float32),
        "axis": np.array(0, dtype=np.int32),
        "output_type": np.int64,
        "name": "argmin_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 5D int64 array, axis 4
    input_dict = {
        "input": np.ones((2, 2, 2, 2, 2), dtype=np.int64),
        "axis": np.array(4, dtype=np.int64),
        "output_type": np.int32,
        "name": "argmin_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.math.argmin"] = tf_math_argmin_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.argmin' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.argmin'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.argmin', generated_inputs['tf.math.argmin'], lib="tf", suffix=0)

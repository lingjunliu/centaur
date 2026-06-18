
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_ArgMax_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        'input': np.array([1.0, 10.0, 26.9, 2.8, 166.32, 62.3], dtype=np.float32),
        'dimension': np.array(0, dtype=np.int32),
        'output_type': tf.int64,
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        'input': np.array([[1, 5, 3], [4, 2, 6]], dtype=np.int32),
        'dimension': np.array(1, dtype=np.int32),
        'output_type': tf.int32,
        'name': 'argmax_2d_axis1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        'input': np.array([[[1.5, 2.3], [4.1, 0.2]], [[-1.2, 5.5], [0.0, -3.1]]], dtype=np.float64),
        'dimension': np.array(2, dtype=np.int64),
        'output_type': tf.int64,
        'name': 'argmax_3d_float64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        'input': np.random.randint(-100, 100, size=(2, 3, 4, 5), dtype=np.int64),
        'dimension': np.array(-1, dtype=np.int32),
        'output_type': tf.int32,
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        'input': np.array([-10.5, -50.2, -2.1, -100.0], dtype=np.float32),
        'dimension': np.array(-1, dtype=np.int32),
        'output_type': tf.int32,
        'name': 'argmax_1d_neg'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        'input': np.array([[-1.5, -2.5], [3.5, 1.2]], dtype=np.float64),
        'dimension': np.array(0, dtype=np.int32),
        'output_type': tf.int64,
        'name': 'argmax_2d_float64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        'input': np.random.uniform(-10.0, 10.0, size=(2, 2, 2)).astype(np.float32),
        'dimension': np.array(-2, dtype=np.int32),
        'output_type': tf.int32,
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        'input': np.random.randint(-100, 100, size=(2, 2, 3, 2, 2), dtype=np.int64),
        'dimension': np.array(3, dtype=np.int64),
        'output_type': tf.int64,
        'name': 'argmax_5d_int64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        'input': np.array([[10, 20], [40, 30]], dtype=np.int32),
        'dimension': np.array(1, dtype=np.int64),
        'output_type': tf.int32,
        'name': 'argmax_int32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        'input': np.array([0.1, 0.9, 0.4, 0.5], dtype=np.float32),
        'dimension': np.array(0, dtype=np.int32),
        'output_type': tf.int64,
        'name': 'argmax_float32_decimals'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.ArgMax"] = tf_raw_ops_ArgMax_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ArgMax' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ArgMax'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.ArgMax', generated_inputs['tf.raw_ops.ArgMax'], lib="tf", suffix=0)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_MatrixDiagPartV3_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        "align": "RIGHT_LEFT",
        "name": "diag_1",
        "input": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32),
        "k": np.array(0, dtype=np.int32),
        "padding_value": np.array(0, dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        "align": "LEFT_RIGHT",
        "name": "diag_2",
        "input": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "k": np.array([-1, 1], dtype=np.int32),
        "padding_value": np.array(-1.0, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        "align": "LEFT_LEFT",
        "name": "diag_3",
        "input": np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.int32),
        "k": np.array(-1, dtype=np.int32),
        "padding_value": np.array(9, dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        "align": "RIGHT_RIGHT",
        "name": "diag_4",
        "input": np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.int64),
        "k": np.array([1, 2], dtype=np.int32),
        "padding_value": np.array(0, dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        "align": "LEFT_RIGHT",
        "name": "diag_5",
        "input": np.array([[1.5, 2.5, 3.5]], dtype=np.float64),
        "k": np.array(0, dtype=np.int32),
        "padding_value": np.array(0.0, dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        "align": "RIGHT_LEFT",
        "name": "diag_6",
        "input": np.zeros((2, 2, 3, 4), dtype=np.float32),
        "k": np.array([-2, 1], dtype=np.int32),
        "padding_value": np.array(99.0, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        "align": "LEFT_RIGHT",
        "name": "diag_7",
        "input": np.array([[True, False], [False, True]], dtype=bool),
        "k": np.array(0, dtype=np.int32),
        "padding_value": np.array(False, dtype=bool)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        "align": "RIGHT_LEFT",
        "name": "diag_8",
        "input": np.array([[1+2j, 3+4j], [5+6j, 7+8j]], dtype=np.complex64),
        "k": np.array([0, 1], dtype=np.int32),
        "padding_value": np.array(0+0j, dtype=np.complex64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        "align": "LEFT_RIGHT",
        "name": "diag_9",
        "input": np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=np.int32),
        "k": np.array([-2, -1], dtype=np.int32),
        "padding_value": np.array(-99, dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        "align": "RIGHT_RIGHT",
        "name": "diag_10",
        "input": np.ones((3, 4, 4), dtype=np.int32),
        "k": np.array([-1, 2], dtype=np.int32),
        "padding_value": np.array(-1, dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.MatrixDiagPartV3"] = tf_raw_ops_MatrixDiagPartV3_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.MatrixDiagPartV3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MatrixDiagPartV3'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.MatrixDiagPartV3', generated_inputs['tf.raw_ops.MatrixDiagPartV3'], lib="tf", suffix=0)

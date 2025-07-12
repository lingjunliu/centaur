
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_count_nonzero_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D array with axis=None
    input_tensor = np.array([[0, 1, 0], [1, 1, 0]], dtype=np.int32)
    input_dict = {"input": input_tensor, "axis": [], "keepdims": False, "dtype": tf.int64, "name": "count_nonzero_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Axis 0
    input_tensor = np.array([[0, 1, 0], [1, 1, 0]], dtype=np.int32)
    input_dict = {"input": input_tensor, "axis": [0], "keepdims": False, "dtype": tf.int64, "name": "count_nonzero_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Axis 1, keepdims=True
    input_tensor = np.array([[0, 1, 0], [1, 1, 0]], dtype=np.int32)
    input_dict = {"input": input_tensor, "axis": [1], "keepdims": True, "dtype": tf.int64, "name": "count_nonzero_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Multi-axis
    input_tensor = np.array([[0, 1, 0], [1, 1, 0]], dtype=np.int32)
    input_dict = {"input": input_tensor, "axis": [0, 1], "keepdims": False, "dtype": tf.int64, "name": "count_nonzero_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array
    input_tensor = np.array([[[0, 1], [0, 1]], [[1, 0], [1, 0]]], dtype=np.int32)
    input_dict = {"input": input_tensor, "axis": [0, 1], "keepdims": False, "dtype": tf.int64, "name": "count_nonzero_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: String tensor (using byte strings)
    input_tensor = np.array([b"", b"a", b"  ", b"b", b""], dtype=np.object_)
    input_dict = {"input": input_tensor, "axis": [], "keepdims": False, "dtype": tf.int64, "name": "count_nonzero_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Bool tensor
    input_tensor = np.array([[False, True], [True, False]], dtype=np.bool_)
    input_dict = {"input": input_tensor, "axis": [], "keepdims": False, "dtype": tf.int64, "name": "count_nonzero_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Negative axis
    input_tensor = np.array([[0, 1, 0], [1, 1, 0]], dtype=np.int32)
    input_dict = {"input": input_tensor, "axis": [-1], "keepdims": False, "dtype": tf.int64, "name": "count_nonzero_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Empty array
    input_tensor = np.array([], dtype=np.int32)
    input_dict = {"input": input_tensor, "axis": [], "keepdims": False, "dtype": tf.int64, "name": "count_nonzero_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Large array with many zeros
    input_tensor = np.zeros((100, 100), dtype=np.int32)
    input_tensor[0, 0] = 1
    input_tensor[50, 50] = 1
    input_dict = {"input": input_tensor, "axis": [], "keepdims": False, "dtype": tf.int64, "name": "count_nonzero_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.count_nonzero"] = tf_math_count_nonzero_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.count_nonzero' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.count_nonzero'.")

check_valid('tf.math.count_nonzero', generated_inputs['tf.math.count_nonzero'], lib="tf", suffix=0)

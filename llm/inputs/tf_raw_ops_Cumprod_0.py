
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_cumprod_inputs():
    list_of_inputs = []

    # Input 1: Basic case, float32, axis 0, exclusive=False, reverse=False
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    axis = np.array(0, dtype=np.int32)
    exclusive = False
    reverse = False
    input_dict = {"x": x, "axis": axis, "exclusive": exclusive, "reverse": reverse, "name": "cumprod_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: int32, axis 0, exclusive=True, reverse=False
    x = np.array([1, 2, 3], dtype=np.int32)
    axis = np.array(0, dtype=np.int32)
    exclusive = True
    reverse = False
    input_dict = {"x": x, "axis": axis, "exclusive": exclusive, "reverse": reverse, "name": "cumprod_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float64, axis 0, exclusive=False, reverse=True
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    axis = np.array(0, dtype=np.int32)
    exclusive = False
    reverse = True
    input_dict = {"x": x, "axis": axis, "exclusive": exclusive, "reverse": reverse, "name": "cumprod_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D, axis 0, exclusive=False, reverse=False
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    axis = np.array(0, dtype=np.int32)
    exclusive = False
    reverse = False
    input_dict = {"x": x, "axis": axis, "exclusive": exclusive, "reverse": reverse, "name": "cumprod_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D, axis 1, exclusive=True, reverse=True
    x = np.array([[1, 2], [3, 4]], dtype=np.float32)
    axis = np.array(1, dtype=np.int32)
    exclusive = True
    reverse = True
    input_dict = {"x": x, "axis": axis, "exclusive": exclusive, "reverse": reverse, "name": "cumprod_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: int64, axis 0, exclusive=True, reverse=False
    x = np.array([1, 2, 3], dtype=np.int64)
    axis = np.array(0, dtype=np.int32)
    exclusive = True
    reverse = False
    input_dict = {"x": x, "axis": axis, "exclusive": exclusive, "reverse": reverse, "name": "cumprod_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: uint8, axis 0, exclusive=False, reverse=True
    x = np.array([1, 2, 3], dtype=np.uint8)
    axis = np.array(0, dtype=np.int32)
    exclusive = False
    reverse = True
    input_dict = {"x": x, "axis": axis, "exclusive": exclusive, "reverse": reverse, "name": "cumprod_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 8: 3D, axis -1, exclusive=False, reverse=False
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32)
    axis = np.array(-1, dtype=np.int32)
    exclusive = False
    reverse = False
    input_dict = {"x": x, "axis": axis, "exclusive": exclusive, "reverse": reverse, "name": "cumprod_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D, int16, axis 0, exclusive=True, reverse=True
    x = np.array([5, 6, 7], dtype=np.int16)
    axis = np.array(0, dtype=np.int32)
    exclusive = True
    reverse = True
    input_dict = {"x": x, "axis": axis, "exclusive": exclusive, "reverse": reverse, "name": "cumprod_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D, float32, axis -2, exclusive=False, reverse=False
    x = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
    axis = np.array(-2, dtype=np.int32)
    exclusive = False
    reverse = False
    input_dict = {"x": x, "axis": axis, "exclusive": exclusive, "reverse": reverse, "name": "cumprod_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Cumprod"] = tf_raw_ops_cumprod_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Cumprod' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Cumprod'.")

check_valid('tf.raw_ops.Cumprod', generated_inputs['tf.raw_ops.Cumprod'], lib="tf", suffix=0)

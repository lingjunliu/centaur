
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_mod_inputs():
    list_of_inputs = []

    # Input 1: int32, basic
    x = np.array([5, 3, 9, 21], dtype=np.int32)
    y = np.array([2, 4, 7, 4], dtype=np.int32)
    name = "mod_example_1"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: int64, broadcasting
    x = np.array([5, 3, 9, 21], dtype=np.int64)
    y = np.array([4], dtype=np.int64)
    name = "mod_example_2"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float32, basic
    x = np.array([5.0, 3.0, 9.0, 21.0], dtype=np.float32)
    y = np.array([2.0, 4.0, 7.0, 4.0], dtype=np.float32)
    name = "mod_example_3"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float64, broadcasting
    x = np.array([5.0, 3.0, 9.0, 21.0], dtype=np.float64)
    y = np.array([4.0], dtype=np.float64)
    name = "mod_example_4"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: int32, multi-dimensional
    x = np.array([[5, 3], [9, 21]], dtype=np.int32)
    y = np.array([[2, 4], [7, 4]], dtype=np.int32)
    name = "mod_example_5"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: int64, multi-dimensional with broadcasting
    x = np.array([[5, 3], [9, 21]], dtype=np.int64)
    y = np.array([4,2], dtype=np.int64)
    name = "mod_example_6"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float32, negative values
    x = np.array([-5.0, 3.0, -9.0, 21.0], dtype=np.float32)
    y = np.array([2.0, -4.0, 7.0, -4.0], dtype=np.float32)
    name = "mod_example_7"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float64, negative values with broadcasting
    x = np.array([-5.0, 3.0, -9.0, 21.0], dtype=np.float64)
    y = np.array([-4.0], dtype=np.float64)
    name = "mod_example_8"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))


    # Input 11: int64, large numbers
    x = np.array([2**31 -1, 2**30, 2**63 - 1, 2**50], dtype=np.int64)
    y = np.array([1000, 2000, 3000, 4000], dtype=np.int64)
    name = "mod_example_11"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: float64, test with 2D
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    y = np.array([[0.5, 1.0], [1.5, 2.0]], dtype=np.float64)
    name = "mod_example_12"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 13: bfloat16,
    x = np.array([5.0, 3.0, 9.0, 21.0], dtype=tf.bfloat16.as_numpy_dtype())
    y = np.array([2.0, 4.0, 7.0, 4.0], dtype=tf.bfloat16.as_numpy_dtype())
    name = "mod_example_13"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 14: half,
    x = np.array([5.0, 3.0, 9.0, 21.0], dtype=np.float16)
    y = np.array([2.0, 4.0, 7.0, 4.0], dtype=np.float16)
    name = "mod_example_14"
    input_dict = {"x": x, "y": y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.Mod"] = tf_raw_ops_mod_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Mod' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Mod'.")

check_valid('tf.raw_ops.Mod', generated_inputs['tf.raw_ops.Mod'], lib="tf", suffix=0)

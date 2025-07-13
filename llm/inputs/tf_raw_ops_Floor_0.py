
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_floor_inputs():
    list_of_inputs = []

    # Input 1: float32, scalar
    x = np.float32(3.14)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float32, 1D array
    x = np.array([-1.5, 0.0, 2.7, -3.2], dtype=np.float32)
    input_dict = {"x": x, "name": "floor_1d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float32, 2D array
    x = np.array([[1.1, 2.2], [3.3, 4.4]], dtype=np.float32)
    input_dict = {"x": x, "name": "floor_2d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float64, scalar
    x = np.float64(-2.718)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: float64, 1D array
    x = np.array([5.5, -6.6, 7.7, -8.8], dtype=np.float64)
    input_dict = {"x": x, "name": "floor_1d_double"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float64, 2D array
    x = np.array([[-1.1, -2.2], [-3.3, -4.4]], dtype=np.float64)
    input_dict = {"x": x, "name": "floor_2d_double"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: bfloat16, scalar
    x = np.float16(1.618)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: bfloat16, 1D array
    x = np.array([9.9, -10.10, 11.11, -12.12], dtype=np.float16)
    input_dict = {"x": x, "name": "floor_1d_bfloat16"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: bfloat16, 2D array
    x = np.array([[13.13, -14.14], [15.15, -16.16]], dtype=np.float16)
    input_dict = {"x": x, "name": "floor_2d_bfloat16"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: half, scalar
    x = np.float16(-0.5)
    input_dict = {"x": x, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
inputs = tf_raw_ops_floor_inputs()
generated_inputs["tf.raw_ops.Floor"] = []
for input_dict in inputs:
    x = tf.convert_to_tensor(input_dict["x"])
    name = input_dict["name"]
    generated_inputs["tf.raw_ops.Floor"].append({"x": x, "name": name})

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Floor' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Floor'.")

check_valid('tf.raw_ops.Floor', generated_inputs['tf.raw_ops.Floor'], lib="tf", suffix=0)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_xdivy_inputs():
    list_of_inputs = []

    # Input 1: float32, basic case
    x = np.array([1.0, 2.0, 0.0, 4.0], dtype=np.float32)
    y = np.array([1.0, 2.0, 3.0, 0.0], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": None}
    list_of_inputs.append(input_dict)

    # Input 2: float64, negative values
    x = np.array([-1.0, -2.0, 0.0, -4.0], dtype=np.float64)
    y = np.array([1.0, -2.0, -3.0, 0.0], dtype=np.float64)
    input_dict = {"x": x, "y": y, "name": "negative_values"}
    list_of_inputs.append(input_dict)

    # Input 3: half (float16), 2D array
    x = np.array([[1.0, 2.0], [0.0, 4.0]], dtype=np.float16)
    y = np.array([[1.0, 2.0], [3.0, 0.0]], dtype=np.float16)
    input_dict = {"x": x, "y": y, "name": "2d_array"}
    list_of_inputs.append(input_dict)

    # Input 4: complex64
    x = np.array([1+1j, 2+2j, 0+0j, 4+4j], dtype=np.complex64)
    y = np.array([1+0j, 2+0j, 3+0j, 0+0j], dtype=np.complex64)
    input_dict = {"x": x, "y": y, "name": "complex_numbers"}
    list_of_inputs.append(input_dict)

    # Input 5: complex128, with some zeros
    x = np.array([1+1j, 2+2j, 0+0j, 4+4j], dtype=np.complex128)
    y = np.array([1+0j, 2+0j, 3+0j, 0+0j], dtype=np.complex128)
    input_dict = {"x": x, "y": y, "name": "complex_numbers_2"}
    list_of_inputs.append(input_dict)

    # Input 6: bfloat16
    x = np.array([1.0, 2.0, 0.0, 4.0], dtype=tf.bfloat16.as_numpy_dtype)
    y = np.array([1.0, 2.0, 3.0, 0.0], dtype=tf.bfloat16.as_numpy_dtype)
    input_dict = {"x": x, "y": y, "name": "bfloat16_example"}
    list_of_inputs.append(input_dict)

     # Input 7: float32, 3D array
    x = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 0.0]]], dtype=np.float32)
    y = np.array([[[1.0, 2.0], [0.0, 4.0]], [[5.0, 0.0], [7.0, 8.0]]], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": "3d_array"}
    list_of_inputs.append(input_dict)

    # Input 8: float64, different shapes that broadcast
    x = np.array([1.0, 2.0], dtype=np.float64)
    y = np.array([[1.0, 2.0], [3.0, 0.0]], dtype=np.float64)
    input_dict = {"x": x, "y": y, "name": "broadcast_1"}
    list_of_inputs.append(input_dict)

    # Input 9: float32, name provided
    x = np.array([1.0, 2.0, 0.0, 4.0], dtype=np.float32)
    y = np.array([1.0, 2.0, 3.0, 0.0], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": "division"}
    list_of_inputs.append(input_dict)

    # Input 10: float64, scalar division
    x = np.array(5.0, dtype=np.float64)
    y = np.array(2.0, dtype=np.float64)
    input_dict = {"x": x, "y": y, "name": "scalar_division"}
    list_of_inputs.append(input_dict)

    return list_of_inputs

generated_inputs = {}
inputs = tf_raw_ops_xdivy_inputs()
generated_inputs["tf.raw_ops.Xdivy"] = []
for input_dict in inputs:
  generated_inputs["tf.raw_ops.Xdivy"].append({"kwargs": input_dict})

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.Xdivy' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Xdivy'.")

check_valid('tf.raw_ops.Xdivy', generated_inputs['tf.raw_ops.Xdivy'], lib="tf", suffix=0)

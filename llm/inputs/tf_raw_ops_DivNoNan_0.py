
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_div_no_nan_inputs():
    list_of_inputs = []

    # Input 1: Basic float32
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([2.0, 0.0, 3.0], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": "basic_float32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Complex64
    x = np.array([1+1j, 2+2j, 3+3j], dtype=np.complex64)
    y = np.array([2+0j, 0+0j, 3+0j], dtype=np.complex64)
    input_dict = {"x": x, "y": y, "name": "complex64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float64, broadcasting
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    y = np.array([2.0, 0.0], dtype=np.float64)
    input_dict = {"x": x, "y": y, "name": "float64_broadcast"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: bfloat16
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32).astype(tf.bfloat16.as_numpy_dtype)
    y = np.array([2.0, 0.0, 3.0], dtype=np.float32).astype(tf.bfloat16.as_numpy_dtype)
    input_dict = {"x": x, "y": y, "name": "bfloat16"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: half
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32).astype(np.float16)
    y = np.array([2.0, 0.0, 3.0], dtype=np.float32).astype(np.float16)
    input_dict = {"x": x, "y": y, "name": "half"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: complex128
    x = np.array([1+1j, 2+2j, 3+3j], dtype=np.complex128)
    y = np.array([2+0j, 0+0j, 3+0j], dtype=np.complex128)
    input_dict = {"x": x, "y": y, "name": "complex128"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.DivNoNan"] = []
for input_dict in tf_raw_ops_div_no_nan_inputs():
  generated_inputs["tf.raw_ops.DivNoNan"].append({k: tf.convert_to_tensor(v) if k in ('x', 'y') else v for k, v in input_dict.items()})

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.DivNoNan' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DivNoNan'.")

check_valid('tf.raw_ops.DivNoNan', generated_inputs['tf.raw_ops.DivNoNan'], lib="tf", suffix=0)

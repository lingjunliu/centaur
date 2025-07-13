
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_MulNoNan_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 multiplication
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([4.0, 5.0, 0.0], dtype=np.float32)
    input_dict = {"x": tf.constant(x), "y": tf.constant(y), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic float64 multiplication with zero
    x = np.array([1.0, 2.0, float('inf')], dtype=np.float64)
    y = np.array([4.0, 0.0, 0.0], dtype=np.float64)
    input_dict = {"x": tf.constant(x), "y": tf.constant(y), "name": "mul_nonan_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Broadcasting example (1D and scalar)
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array(2.0, dtype=np.float32)
    input_dict = {"x": tf.constant(x), "y": tf.constant(y), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Broadcasting example (2D and 1D)
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    y = np.array([0.0, 2.0], dtype=np.float32)
    input_dict = {"x": tf.constant(x), "y": tf.constant(y), "name": "mul_nonan_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Complex64 multiplication
    x = np.array([1 + 1j, 2 + 2j, 3 + 3j], dtype=np.complex64)
    y = np.array([4 + 4j, 5 + 5j, 0 + 0j], dtype=np.complex64)
    input_dict = {"x": tf.constant(x), "y": tf.constant(y), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Complex128 multiplication with NaN
    x = np.array([1 + 1j, 2 + 2j, float('nan') + 0j], dtype=np.complex128)
    y = np.array([4 + 4j, 0 + 0j, 0 + 0j], dtype=np.complex128)
    input_dict = {"x": tf.constant(x), "y": tf.constant(y), "name": "mul_nonan_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: bfloat16 multiplication
    x = np.array([1.0, 2.0, 3.0], dtype=np.float16).astype(tf.bfloat16.as_numpy_dtype)
    y = np.array([4.0, 5.0, 0.0], dtype=np.float16).astype(tf.bfloat16.as_numpy_dtype)
    input_dict = {"x": tf.constant(x), "y": tf.constant(y), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: half (float16) multiplication
    x = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    y = np.array([4.0, 5.0, 0.0], dtype=np.float16)
    input_dict = {"x": tf.constant(x), "y": tf.constant(y), "name": "mul_nonan_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Multi-dimensional array
    x = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    y = np.array([[[0.0, 1.0], [1.0, 0.0]], [[1.0, 1.0], [0.0, 0.0]]], dtype=np.float32)
    input_dict = {"x": tf.constant(x), "y": tf.constant(y), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Negative values
    x = np.array([-1.0, 2.0, -3.0], dtype=np.float32)
    y = np.array([4.0, -5.0, 0.0], dtype=np.float32)
    input_dict = {"x": tf.constant(x), "y": tf.constant(y), "name": "mul_nonan_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.MulNoNan"] = tf_raw_ops_MulNoNan_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.MulNoNan' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MulNoNan'.")

check_valid('tf.raw_ops.MulNoNan', generated_inputs['tf.raw_ops.MulNoNan'], lib="tf", suffix=0)

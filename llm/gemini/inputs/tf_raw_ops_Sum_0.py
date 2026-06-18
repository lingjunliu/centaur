
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_Sum_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D float32 array, sum along axis 0
    input_val = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    axis_val = np.array([0], dtype=np.int32)
    keep_dims = False
    name = "sum_1"
    list_of_inputs.append({
        "input": input_val,
        "axis": axis_val,
        "keep_dims": keep_dims,
        "name": name
    })

    # Input 2: 2D int32 array with negative values, keep dimensions
    input_val = np.array([[1, -2, 3], [4, 5, -6]], dtype=np.int32)
    axis_val = np.array([1], dtype=np.int32)
    keep_dims = True
    name = "sum_2"
    list_of_inputs.append({
        "input": input_val,
        "axis": axis_val,
        "keep_dims": keep_dims,
        "name": name
    })

    # Input 3: 3D float64 array, reduce multiple axes
    input_val = np.array([[[1.5, 2.5], [3.5, 4.5]], [[5.5, 6.5], [7.5, 8.5]]], dtype=np.float64)
    axis_val = np.array([0, 2], dtype=np.int64)
    keep_dims = False
    name = "sum_3"
    list_of_inputs.append({
        "input": input_val,
        "axis": axis_val,
        "keep_dims": keep_dims,
        "name": name
    })

    # Input 4: 1D int64 array, scalar reduction axis
    input_val = np.array([-10, 20, -30, 40], dtype=np.int64)
    axis_val = np.array([0], dtype=np.int32)
    keep_dims = True
    name = "sum_4"
    list_of_inputs.append({
        "input": input_val,
        "axis": axis_val,
        "keep_dims": keep_dims,
        "name": name
    })

    # Input 5: Complex numbers (complex64) sum
    input_val = np.array([[1+2j, 3+4j], [5+6j, 7+8j]], dtype=np.complex64)
    axis_val = np.array([1], dtype=np.int32)
    keep_dims = False
    name = "sum_5"
    list_of_inputs.append({
        "input": input_val,
        "axis": axis_val,
        "keep_dims": keep_dims,
        "name": name
    })

    # Input 6: Unsigned integers (uint8), sum along axis 0
    input_val = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    axis_val = np.array([0], dtype=np.int64)
    keep_dims = True
    name = "sum_6"
    list_of_inputs.append({
        "input": input_val,
        "axis": axis_val,
        "keep_dims": keep_dims,
        "name": name
    })

    # Input 7: 3D int16 array, negative indexing for axis
    input_val = np.array([[[1, -2], [3, -4]], [[5, -6], [7, -8]]], dtype=np.int16)
    axis_val = np.array([-1], dtype=np.int32)
    keep_dims = False
    name = "sum_7"
    list_of_inputs.append({
        "input": input_val,
        "axis": axis_val,
        "keep_dims": keep_dims,
        "name": name
    })

    # Input 8: High dimensional random float32 array, fully reduced (all axes)
    input_val = np.random.randn(2, 3, 4).astype(np.float32)
    axis_val = np.array([0, 1, 2], dtype=np.int32)
    keep_dims = False
    name = "sum_8"
    list_of_inputs.append({
        "input": input_val,
        "axis": axis_val,
        "keep_dims": keep_dims,
        "name": name
    })

    # Input 9: Small int8 array with negative axis
    input_val = np.array([-5, 5, -10, 10], dtype=np.int8)
    axis_val = np.array([-1], dtype=np.int64)
    keep_dims = True
    name = "sum_9"
    list_of_inputs.append({
        "input": input_val,
        "axis": axis_val,
        "keep_dims": keep_dims,
        "name": name
    })

    # Input 10: 4D array of ones, complex reduction
    input_val = np.ones((2, 2, 2, 2), dtype=np.float32)
    axis_val = np.array([1, 3], dtype=np.int32)
    keep_dims = True
    name = "sum_10"
    list_of_inputs.append({
        "input": input_val,
        "axis": axis_val,
        "keep_dims": keep_dims,
        "name": name
    })

    return list_of_inputs

generated_inputs["tf.raw_ops.Sum"] = tf_raw_ops_Sum_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Sum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Sum'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Sum', generated_inputs['tf.raw_ops.Sum'], lib="tf", suffix=0)

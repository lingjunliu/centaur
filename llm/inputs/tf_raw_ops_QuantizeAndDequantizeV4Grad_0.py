
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_quantize_and_dequantize_v4_grad_inputs():
    list_of_inputs = []

    # Input 1
    gradients = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_tensor = np.array([0.5, 1.5, 2.5], dtype=np.float32)
    input_min = np.array([0.0], dtype=np.float32)
    input_max = np.array([3.0], dtype=np.float32)
    axis = -1

    input_dict = {
        "gradients": gradients,
        "input": input_tensor,
        "input_min": input_min,
        "input_max": input_max,
        "axis": axis,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    gradients = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    input_tensor = np.array([-0.5, -1.5, -2.5], dtype=np.float32)
    input_min = np.array([-3.0], dtype=np.float32)
    input_max = np.array([0.0], dtype=np.float32)
    axis = -1
    input_dict = {
        "gradients": gradients,
        "input": input_tensor,
        "input_min": input_min,
        "input_max": input_max,
        "axis": axis,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    gradients = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_tensor = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float32)
    input_min = np.array([0.0], dtype=np.float32)
    input_max = np.array([4.0], dtype=np.float32)
    axis = -1
    input_dict = {
        "gradients": gradients,
        "input": input_tensor,
        "input_min": input_min,
        "input_max": input_max,
        "axis": axis,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    gradients = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    input_tensor = np.array([0.5, 1.5, 2.5], dtype=np.float64)
    input_min = np.array([0.0], dtype=np.float64)
    input_max = np.array([3.0], dtype=np.float64)
    axis = 0
    input_dict = {
        "gradients": gradients,
        "input": input_tensor,
        "input_min": input_min,
        "input_max": input_max,
        "axis": axis,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    gradients = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_tensor = np.array([0.5, 1.5, 2.5], dtype=np.float32)
    input_min = np.array([0.0], dtype=np.float32)
    input_max = np.array([3.0], dtype=np.float32)
    axis = 0
    input_dict = {
        "gradients": gradients,
        "input": input_tensor,
        "input_min": input_min,
        "input_max": input_max,
        "axis": axis,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    gradients = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    input_tensor = np.array([[[0.5, 1.5], [2.5, 3.5]], [[4.5, 5.5], [6.5, 7.5]]], dtype=np.float32)
    input_min = np.array([0.0], dtype=np.float32)
    input_max = np.array([8.0], dtype=np.float32)
    axis = 1
    input_dict = {
        "gradients": gradients,
        "input": input_tensor,
        "input_min": input_min,
        "input_max": input_max,
        "axis": axis,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7
    gradients = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_tensor = np.array([0.5, 1.5, 2.5], dtype=np.float32)
    input_min = np.array([0.0], dtype=np.float32)
    input_max = np.array([3.0], dtype=np.float32)
    axis = -1
    input_dict = {
        "gradients": gradients,
        "input": input_tensor,
        "input_min": input_min,
        "input_max": input_max,
        "axis": axis,
        "name": "test_name"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    gradients = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    input_tensor = np.array([0.5, 1.5, 2.5], dtype=np.float16)
    input_min = np.array([0.0], dtype=np.float16)
    input_max = np.array([3.0], dtype=np.float16)
    axis = -1
    input_dict = {
        "gradients": gradients,
        "input": input_tensor,
        "input_min": input_min,
        "input_max": input_max,
        "axis": axis,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    gradients = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    input_tensor = np.array([0.5, 1.5, 2.5], dtype=np.float16)
    input_min = np.array([0.0], dtype=np.float16)
    input_max = np.array([3.0], dtype=np.float16)
    axis = -1
    input_dict = {
        "gradients": gradients,
        "input": input_tensor,
        "input_min": input_min,
        "input_max": input_max,
        "axis": axis,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 10
    gradients = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    input_tensor = np.array([0.5, 1.5, 2.5], dtype=np.float64)
    input_min = np.array([0.0], dtype=np.float64)
    input_max = np.array([3.0], dtype=np.float64)
    axis = -1
    input_dict = {
        "gradients": gradients,
        "input": input_tensor,
        "input_min": input_min,
        "input_max": input_max,
        "axis": axis,
        "name": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
inputs = tf_raw_ops_quantize_and_dequantize_v4_grad_inputs()
generated_inputs["tf.raw_ops.QuantizeAndDequantizeV4Grad"] = []
for input_dict in inputs:
    generated_inputs["tf.raw_ops.QuantizeAndDequantizeV4Grad"].append(input_dict)

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.QuantizeAndDequantizeV4Grad' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QuantizeAndDequantizeV4Grad'.")

check_valid('tf.raw_ops.QuantizeAndDequantizeV4Grad', generated_inputs['tf.raw_ops.QuantizeAndDequantizeV4Grad'], lib="tf", suffix=0)

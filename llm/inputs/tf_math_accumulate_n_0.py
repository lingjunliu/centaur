
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_accumulate_n_inputs():
    list_of_inputs = []

    def create_input_dict(inputs, shape, tensor_dtype, name):
        return {"inputs": inputs, "shape": shape, "tensor_dtype": tensor_dtype, "name": name}

    # Input 1
    inputs_val = [np.array([[1, 2], [3, 4]], dtype=np.int32), np.array([[5, 0], [0, 6]], dtype=np.int32)]
    shape_val = [2, 2]
    tensor_dtype_val = np.int32
    name_val = "test_accumulate_1"
    list_of_inputs.append(copy.deepcopy(create_input_dict(inputs_val, shape_val, tensor_dtype_val, name_val)))

    # Input 2
    inputs_val = [np.array([1.0, 2.0, 3.0], dtype=np.float32), np.array([4.0, 5.0, 6.0], dtype=np.float32)]
    shape_val = [3]
    tensor_dtype_val = np.float32
    name_val = "test_accumulate_2"
    list_of_inputs.append(copy.deepcopy(create_input_dict(inputs_val, shape_val, tensor_dtype_val, name_val)))

    # Input 3
    inputs_val = [np.array([[-1, -2], [-3, -4]], dtype=np.int32), np.array([[5, 0], [0, 6]], dtype=np.int32)]
    shape_val = [2, 2]
    tensor_dtype_val = np.int32
    name_val = "test_accumulate_3"
    list_of_inputs.append(copy.deepcopy(create_input_dict(inputs_val, shape_val, tensor_dtype_val, name_val)))

    # Input 4
    inputs_val = [np.array([1], dtype=np.int64), np.array([2], dtype=np.int64), np.array([3], dtype=np.int64)]
    shape_val = [1]
    tensor_dtype_val = np.int64
    name_val = "test_accumulate_4"
    list_of_inputs.append(copy.deepcopy(create_input_dict(inputs_val, shape_val, tensor_dtype_val, name_val)))

    # Input 5
    inputs_val = [np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64), np.array([[5.5, 0.5], [0.5, 6.5]], dtype=np.float64)]
    shape_val = [2, 2]
    tensor_dtype_val = np.float64
    name_val = "test_accumulate_5"
    list_of_inputs.append(copy.deepcopy(create_input_dict(inputs_val, shape_val, tensor_dtype_val, name_val)))

    # Input 6
    inputs_val = [np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32), np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]], dtype=np.int32)]
    shape_val = [2, 2, 2]
    tensor_dtype_val = np.int32
    name_val = "test_accumulate_6"
    list_of_inputs.append(copy.deepcopy(create_input_dict(inputs_val, shape_val, tensor_dtype_val, name_val)))

    # Input 7
    inputs_val = [np.array([1, 2, 3], dtype=np.int32), np.array([4, 5, 6], dtype=np.int32), np.array([7, 8, 9], dtype=np.int32)]
    shape_val = [3]
    tensor_dtype_val = np.int32
    name_val = "test_accumulate_7"
    list_of_inputs.append(copy.deepcopy(create_input_dict(inputs_val, shape_val, tensor_dtype_val, name_val)))

    # Input 8
    inputs_val = [np.array([0.1, 0.2, 0.3], dtype=np.float32), np.array([0.4, 0.5, 0.6], dtype=np.float32), np.array([0.7, 0.8, 0.9], dtype=np.float32)]
    shape_val = [3]
    tensor_dtype_val = np.float32
    name_val = "test_accumulate_8"
    list_of_inputs.append(copy.deepcopy(create_input_dict(inputs_val, shape_val, tensor_dtype_val, name_val)))

    # Input 9
    inputs_val = [np.array([[-1.5, 2.5], [-3.5, 4.5]], dtype=np.float64), np.array([[5.5, -0.5], [-0.5, 6.5]], dtype=np.float64), np.array([[1.0, 2.0], [3.0, -4.0]], dtype=np.float64)]
    shape_val = [2, 2]
    tensor_dtype_val = np.float64
    name_val = "test_accumulate_9"
    list_of_inputs.append(copy.deepcopy(create_input_dict(inputs_val, shape_val, tensor_dtype_val, name_val)))

    # Input 10
    inputs_val = [np.array([1000, 2000, 3000], dtype=np.int64), np.array([4000, 5000, 6000], dtype=np.int64)]
    shape_val = [3]
    tensor_dtype_val = np.int64
    name_val = "test_accumulate_10"
    list_of_inputs.append(copy.deepcopy(create_input_dict(inputs_val, shape_val, tensor_dtype_val, name_val)))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.math.accumulate_n"] = tf_math_accumulate_n_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.math.accumulate_n' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.accumulate_n'.")

check_valid('tf.math.accumulate_n', generated_inputs['tf.math.accumulate_n'], lib="tf", suffix=0)

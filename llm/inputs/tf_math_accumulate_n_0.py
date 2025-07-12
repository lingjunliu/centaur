
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_accumulate_n_inputs():
    list_of_inputs = []

    # Input 1: Basic case with integers
    inputs = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
    shape = [2, 2]
    tensor_dtype = np.int32
    name = "sum_tensors_1"
    input_dict = {"inputs": inputs, "shape": shape, "tensor_dtype": tensor_dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different shape, float32
    inputs = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]
    shape = [3]
    tensor_dtype = np.float32
    name = "sum_tensors_2"
    input_dict = {"inputs": inputs, "shape": shape, "tensor_dtype": tensor_dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D tensor
    inputs = [np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])]
    shape = [2, 2, 2]
    tensor_dtype = np.int32
    name = "sum_tensors_3"
    input_dict = {"inputs": inputs, "shape": shape, "tensor_dtype": tensor_dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Empty shape
    inputs = [np.array(5), np.array(10)]
    shape = []
    tensor_dtype = np.int32
    name = "sum_tensors_4"
    input_dict = {"inputs": inputs, "shape": shape, "tensor_dtype": tensor_dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Float64 type
    inputs = [np.array([[1.5, 2.5], [3.5, 4.5]]), np.array([[5.5, 6.5], [7.5, 8.5]])]
    shape = [2, 2]
    tensor_dtype = np.float64
    name = "sum_tensors_5"
    input_dict = {"inputs": inputs, "shape": shape, "tensor_dtype": tensor_dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Negative values
    inputs = [np.array([[-1, 2], [3, -4]]), np.array([[5, -6], [-7, 8]])]
    shape = [2, 2]
    tensor_dtype = np.int32
    name = "sum_tensors_6"
    input_dict = {"inputs": inputs, "shape": shape, "tensor_dtype": tensor_dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 7: More tensors
    inputs = [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])]
    shape = [2]
    tensor_dtype = np.int32
    name = "sum_tensors_7"
    input_dict = {"inputs": inputs, "shape": shape, "tensor_dtype": tensor_dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8:  uint8
    inputs = [np.array([1, 2], dtype=np.uint8), np.array([3, 4], dtype=np.uint8)]
    shape = [2]
    tensor_dtype = np.uint8
    name = "sum_tensors_8"
    input_dict = {"inputs": inputs, "shape": shape, "tensor_dtype": tensor_dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9:  int64
    inputs = [np.array([1, 2], dtype=np.int64), np.array([3, 4], dtype=np.int64)]
    shape = [2]
    tensor_dtype = np.int64
    name = "sum_tensors_9"
    input_dict = {"inputs": inputs, "shape": shape, "tensor_dtype": tensor_dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10:  Complex64
    inputs = [np.array([1+1j, 2+2j], dtype=np.complex64), np.array([3+3j, 4+4j], dtype=np.complex64)]
    shape = [2]
    tensor_dtype = np.complex64
    name = "sum_tensors_10"
    input_dict = {"inputs": inputs, "shape": shape, "tensor_dtype": tensor_dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

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

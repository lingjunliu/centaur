
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_SparseSliceGrad_inputs():
    list_of_inputs = []

    # Input 1
    backprop_val_grad = np.array([1.0, 2.0], dtype=np.float32)
    input_indices = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64)
    input_start = np.array([0, 0], dtype=np.int64)
    output_indices = np.array([[0, 0], [0, 1]], dtype=np.int64)
    input_dict = {"backprop_val_grad": backprop_val_grad, "input_indices": input_indices, "input_start": input_start, "output_indices": output_indices, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    backprop_val_grad = np.array([3, 4], dtype=np.int32)
    input_indices = np.array([[2, 3], [4, 5], [6, 7]], dtype=np.int64)
    input_start = np.array([2, 3], dtype=np.int64)
    output_indices = np.array([[0, 0], [2, 2]], dtype=np.int64)
    input_dict = {"backprop_val_grad": backprop_val_grad, "input_indices": input_indices, "input_start": input_start, "output_indices": output_indices, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    backprop_val_grad = np.array([1.5, 2.5, 3.5], dtype=np.float64)
    input_indices = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]], dtype=np.int64)
    input_start = np.array([0, 0, 0], dtype=np.int64)
    output_indices = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.int64)
    input_dict = {"backprop_val_grad": backprop_val_grad, "input_indices": input_indices, "input_start": input_start, "output_indices": output_indices, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    backprop_val_grad = np.array([5, 6, 7, 8], dtype=np.int64)
    input_indices = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=np.int64)
    input_start = np.array([1, 2, 3], dtype=np.int64)
    output_indices = np.array([[0, 0, 0], [3, 3, 3], [6, 6, 6], [9, 9, 9]], dtype=np.int64)
    input_dict = {"backprop_val_grad": backprop_val_grad, "input_indices": input_indices, "input_start": input_start, "output_indices": output_indices, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 5
    backprop_val_grad = np.array([1, 2, 3], dtype=np.complex64)
    input_indices = np.array([[0, 0], [0, 1], [1, 0]], dtype=np.int64)
    input_start = np.array([0, 0], dtype=np.int64)
    output_indices = np.array([[0, 0], [0, 1], [1, 0]], dtype=np.int64)
    input_dict = {"backprop_val_grad": backprop_val_grad, "input_indices": input_indices, "input_start": input_start, "output_indices": output_indices, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    backprop_val_grad = np.array([10, 20], dtype=np.uint8)
    input_indices = np.array([[0, 0], [0, 1], [1, 0]], dtype=np.int64)
    input_start = np.array([0, 0], dtype=np.int64)
    output_indices = np.array([[0, 0], [0, 1]], dtype=np.int64)
    input_dict = {"backprop_val_grad": backprop_val_grad, "input_indices": input_indices, "input_start": input_start, "output_indices": output_indices, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    backprop_val_grad = np.array([1, 2, 3], dtype=np.int16)
    input_indices = np.array([[0, 0], [0, 1], [1, 0]], dtype=np.int64)
    input_start = np.array([0, 0], dtype=np.int64)
    output_indices = np.array([[0, 0], [0, 1], [1, 0]], dtype=np.int64)
    input_dict = {"backprop_val_grad": backprop_val_grad, "input_indices": input_indices, "input_start": input_start, "output_indices": output_indices, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    backprop_val_grad = np.array([1, 2], dtype=np.int8)
    input_indices = np.array([[0, 0], [0, 1], [1, 0]], dtype=np.int64)
    input_start = np.array([0, 0], dtype=np.int64)
    output_indices = np.array([[0, 0], [0, 1]], dtype=np.int64)
    input_dict = {"backprop_val_grad": backprop_val_grad, "input_indices": input_indices, "input_start": input_start, "output_indices": output_indices, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    backprop_val_grad = np.array([1, 2], dtype=np.complex128)
    input_indices = np.array([[0, 0], [0, 1], [1, 0]], dtype=np.int64)
    input_start = np.array([0, 0], dtype=np.int64)
    output_indices = np.array([[0, 0], [0, 1]], dtype=np.int64)
    input_dict = {"backprop_val_grad": backprop_val_grad, "input_indices": input_indices, "input_start": input_start, "output_indices": output_indices, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    backprop_val_grad = np.array([1, 2], dtype=np.float16)
    input_indices = np.array([[0, 0], [0, 1], [1, 0]], dtype=np.int64)
    input_start = np.array([0, 0], dtype=np.int64)
    output_indices = np.array([[0, 0], [0, 1]], dtype=np.int64)
    input_dict = {"backprop_val_grad": backprop_val_grad, "input_indices": input_indices, "input_start": input_start, "output_indices": output_indices, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    backprop_val_grad = np.array([1], dtype=np.float32)
    input_indices = np.array([[0]], dtype=np.int64)
    input_start = np.array([0], dtype=np.int64)
    output_indices = np.array([[0]], dtype=np.int64)
    input_dict = {"backprop_val_grad": backprop_val_grad, "input_indices": input_indices, "input_start": input_start, "output_indices": output_indices, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    backprop_val_grad = np.array([1.0], dtype=np.float32)
    input_indices = np.array([[0, 0]], dtype=np.int64)
    input_start = np.array([0, 0], dtype=np.int64)
    output_indices = np.array([[0, 0]], dtype=np.int64)
    input_dict = {"backprop_val_grad": backprop_val_grad, "input_indices": input_indices, "input_start": input_start, "output_indices": output_indices, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 13
    backprop_val_grad = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_indices = np.array([[0, 0], [0, 1], [1, 0]], dtype=np.int64)
    input_start = np.array([0, 0], dtype=np.int64)
    output_indices = np.array([[0, 0], [0, 1], [1, 0]], dtype=np.int64)
    input_dict = {"backprop_val_grad": backprop_val_grad, "input_indices": input_indices, "input_start": input_start, "output_indices": output_indices, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.SparseSliceGrad"] = tf_raw_ops_SparseSliceGrad_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.SparseSliceGrad' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.SparseSliceGrad'.")

check_valid('tf.raw_ops.SparseSliceGrad', generated_inputs['tf.raw_ops.SparseSliceGrad'], lib="tf", suffix=0)

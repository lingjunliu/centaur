
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_BatchMatMul_inputs():
    list_of_inputs = []

    # Input 1: Basic example with float32
    x = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
    y = np.array([[[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    input_dict = {"x": x, "y": y, "adj_x": False, "adj_y": False, "grad_x": False, "grad_y": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different batch size and adjoint x
    x = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    y = np.array([[[5.0, 6.0], [7.0, 8.0]], [[9.0, 10.0], [11.0, 12.0]]], dtype=np.float32)
    input_dict = {"x": x, "y": y, "adj_x": True, "adj_y": False, "grad_x": False, "grad_y": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Adjoint y
    x = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
    y = np.array([[[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    input_dict = {"x": x, "y": y, "adj_x": False, "adj_y": True, "grad_x": False, "grad_y": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Both adjoint
    x = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
    y = np.array([[[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    input_dict = {"x": x, "y": y, "adj_x": True, "adj_y": True, "grad_x": False, "grad_y": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different matrix dimensions
    x = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], dtype=np.float32)
    y = np.array([[[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]], dtype=np.float32)
    input_dict = {"x": x, "y": y, "adj_x": False, "adj_y": False, "grad_x": False, "grad_y": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Complex64 type
    x = np.array([[[1.0+1j, 2.0+2j], [3.0+3j, 4.0+4j]]], dtype=np.complex64)
    y = np.array([[[5.0+5j, 6.0+6j], [7.0+7j, 8.0+8j]]], dtype=np.complex64)
    input_dict = {"x": x, "y": y, "adj_x": False, "adj_y": False, "grad_x": False, "grad_y": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Int32 type
    x = np.array([[[1, 2], [3, 4]]], dtype=np.int32)
    y = np.array([[[5, 6], [7, 8]]], dtype=np.int32)
    input_dict = {"x": x, "y": y, "adj_x": False, "adj_y": False, "grad_x": False, "grad_y": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 8: Multiple batches with int64
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int64)
    y = np.array([[[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.int64)
    input_dict = {"x": x, "y": y, "adj_x": False, "adj_y": False, "grad_x": False, "grad_y": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: half
    x = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float16)
    y = np.array([[[5.0, 6.0], [7.0, 8.0]]], dtype=np.float16)
    input_dict = {"x": x, "y": y, "adj_x": False, "adj_y": False, "grad_x": False, "grad_y": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Complex128 type
    x = np.array([[[1.0+1j, 2.0+2j], [3.0+3j, 4.0+4j]]], dtype=np.complex128)
    y = np.array([[[5.0+5j, 6.0+6j], [7.0+7j, 8.0+8j]]], dtype=np.complex128)
    input_dict = {"x": x, "y": y, "adj_x": False, "adj_y": False, "grad_x": False, "grad_y": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.BatchMatMul"] = tf_raw_ops_BatchMatMul_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.BatchMatMul' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.BatchMatMul'.")

check_valid('tf.raw_ops.BatchMatMul', generated_inputs['tf.raw_ops.BatchMatMul'], lib="tf", suffix=0)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_BatchMatMulV2_inputs():
    list_of_inputs = []

    # Input 1
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    y = np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]], dtype=np.float32)
    adj_x = False
    adj_y = False
    grad_x = False
    grad_y = False
    name = "batch_matmul_1"
    input_dict = {"x": x, "y": y, "adj_x": adj_x, "adj_y": adj_y, "grad_x": grad_x, "grad_y": grad_y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    x = np.array([[1, 2], [3, 4]], dtype=np.float32)
    y = np.array([[9, 10], [11, 12]], dtype=np.float32)
    adj_x = True
    adj_y = True
    grad_x = False
    grad_y = False
    name = "batch_matmul_2"
    input_dict = {"x": x, "y": y, "adj_x": adj_x, "adj_y": adj_y, "grad_x": grad_x, "grad_y": grad_y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    x = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.float32)
    y = np.array([[[7, 8], [9, 10], [11, 12]]], dtype=np.float32)
    adj_x = False
    adj_y = False
    grad_x = True
    grad_y = False
    name = "batch_matmul_3"
    input_dict = {"x": x, "y": y, "adj_x": adj_x, "adj_y": adj_y, "grad_x": grad_x, "grad_y": grad_y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    x = np.array([[[1, 2], [3, 4]]], dtype=np.float32)
    y = np.array([[[5, 6], [7, 8]]], dtype=np.float32)
    adj_x = True
    adj_y = False
    grad_x = False
    grad_y = True
    name = "batch_matmul_4"
    input_dict = {"x": x, "y": y, "adj_x": adj_x, "adj_y": adj_y, "grad_x": grad_x, "grad_y": grad_y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.complex64)
    y = np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]], dtype=np.complex64)
    adj_x = False
    adj_y = True
    grad_x = True
    grad_y = True
    name = "batch_matmul_5"
    input_dict = {"x": x, "y": y, "adj_x": adj_x, "adj_y": adj_y, "grad_x": grad_x, "grad_y": grad_y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    y = np.array([[7, 8], [9, 10], [11, 12]], dtype=np.float64)
    adj_x = False
    adj_y = False
    grad_x = False
    grad_y = False
    name = "batch_matmul_6"
    input_dict = {"x": x, "y": y, "adj_x": adj_x, "adj_y": adj_y, "grad_x": grad_x, "grad_y": grad_y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    x = np.array([[[1, 2], [3, 4]]], dtype=np.int32)
    y = np.array([[[5, 6], [7, 8]]], dtype=np.int32)
    adj_x = True
    adj_y = True
    grad_x = True
    grad_y = False
    name = "batch_matmul_7"
    input_dict = {"x": x, "y": y, "adj_x": adj_x, "adj_y": adj_y, "grad_x": grad_x, "grad_y": grad_y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float16)
    y = np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]], dtype=np.float16)
    adj_x = True
    adj_y = False
    grad_x = True
    grad_y = False
    name = "batch_matmul_9"
    input_dict = {"x": x, "y": y, "adj_x": adj_x, "adj_y": adj_y, "grad_x": grad_x, "grad_y": grad_y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    x = np.array([[1, 2], [3, 4]], dtype=np.float32)
    y = np.array([[[5, 6], [7, 8]]], dtype=np.float32)
    adj_x = False
    adj_y = False
    grad_x = False
    grad_y = False
    name = "batch_matmul_10"
    input_dict = {"x": x, "y": y, "adj_x": adj_x, "adj_y": adj_y, "grad_x": grad_x, "grad_y": grad_y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.BatchMatMulV2"] = tf_raw_ops_BatchMatMulV2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.BatchMatMulV2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.BatchMatMulV2'.")

check_valid('tf.raw_ops.BatchMatMulV2', generated_inputs['tf.raw_ops.BatchMatMulV2'], lib="tf", suffix=0)

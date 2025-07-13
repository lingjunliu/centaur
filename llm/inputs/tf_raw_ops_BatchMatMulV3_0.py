
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_batch_mat_mul_v3_inputs():
    list_of_inputs = []

    # Input 1
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    y = np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]], dtype=np.float32)
    Tout = tf.float32
    adj_x = False
    adj_y = False
    grad_x = False
    grad_y = False
    name = "matmul1"
    input_dict = {"x": x, "y": y, "Tout": Tout, "adj_x": adj_x, "adj_y": adj_y, "grad_x": grad_x, "grad_y": grad_y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    x = np.array([[[1, 2], [3, 4]]], dtype=np.float64)
    y = np.array([[[9, 10], [11, 12]]], dtype=np.float64)
    Tout = tf.float64
    adj_x = True
    adj_y = True
    grad_x = True
    grad_y = True
    name = "matmul2"
    input_dict = {"x": x, "y": y, "Tout": Tout, "adj_x": adj_x, "adj_y": adj_y, "grad_x": grad_x, "grad_y": grad_y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    x = np.array([[1, 2], [3, 4]], dtype=np.float32)
    y = np.array([[9, 10], [11, 12]], dtype=np.float32)
    Tout = tf.float32
    adj_x = False
    adj_y = False
    grad_x = False
    grad_y = False
    name = "matmul3"
    input_dict = {"x": x, "y": y, "Tout": Tout, "adj_x": adj_x, "adj_y": adj_y, "grad_x": grad_x, "grad_y": grad_y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    x = np.array([[[1+1j, 2+2j], [3+3j, 4+4j]]], dtype=np.complex64)
    y = np.array([[[9+9j, 10+10j], [11+11j, 12+12j]]], dtype=np.complex64)
    Tout = tf.complex64
    adj_x = True
    adj_y = False
    grad_x = False
    grad_y = False
    name = "matmul4"
    input_dict = {"x": x, "y": y, "Tout": Tout, "adj_x": adj_x, "adj_y": adj_y, "grad_x": grad_x, "grad_y": grad_y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 5
    x = np.array([[[1, 2, 3], [4, 5, 6]], [[7,8,9], [10,11,12]]], dtype=np.float32)
    y = np.array([[[13, 14], [15, 16], [17,18]], [[19,20], [21,22], [23,24]]], dtype=np.float32)
    Tout = tf.float32
    adj_x = False
    adj_y = False
    grad_x = False
    grad_y = False
    name = "matmul5"
    input_dict = {"x": x, "y": y, "Tout": Tout, "adj_x": adj_x, "adj_y": adj_y, "grad_x": grad_x, "grad_y": grad_y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    x = np.array([[[1, 2], [3, 4]]], dtype=np.float32)
    y = np.array([[[5], [6]]], dtype=np.float32)
    Tout = tf.float32
    adj_x = False
    adj_y = False
    grad_x = False
    grad_y = False
    name = "matmul6"
    input_dict = {"x": x, "y": y, "Tout": Tout, "adj_x": adj_x, "adj_y": adj_y, "grad_x": grad_x, "grad_y": grad_y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    x = np.array([[[1, 2], [3, 4]]], dtype=np.float32)
    y = np.array([[[5], [6]]], dtype=np.float32)
    Tout = tf.float32
    adj_x = True
    adj_y = True
    grad_x = False
    grad_y = False
    name = "matmul7"
    input_dict = {"x": x, "y": y, "Tout": Tout, "adj_x": adj_x, "adj_y": adj_y, "grad_x": grad_x, "grad_y": grad_y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    x = np.random.rand(2,3,4).astype(np.float32)
    y = np.random.rand(2,4,5).astype(np.float32)
    Tout = tf.float32
    adj_x = False
    adj_y = False
    grad_x = False
    grad_y = False
    name = "matmul8"
    input_dict = {"x": x, "y": y, "Tout": Tout, "adj_x": adj_x, "adj_y": adj_y, "grad_x": grad_x, "grad_y": grad_y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    x = np.random.rand(2,3,4).astype(np.float64)
    y = np.random.rand(2,4,5).astype(np.float64)
    Tout = tf.float64
    adj_x = True
    adj_y = True
    grad_x = False
    grad_y = False
    name = "matmul9"
    input_dict = {"x": x, "y": y, "Tout": Tout, "adj_x": adj_x, "adj_y": adj_y, "grad_x": grad_x, "grad_y": grad_y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    y = np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]], dtype=np.float32)
    Tout = tf.float32
    adj_x = True
    adj_y = False
    grad_x = False
    grad_y = False
    name = "matmul10"
    input_dict = {"x": x, "y": y, "Tout": Tout, "adj_x": adj_x, "adj_y": adj_y, "grad_x": grad_x, "grad_y": grad_y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11
    x = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.float32)
    y = np.array([[[7, 8], [9, 10], [11, 12]]], dtype=np.float32)
    Tout = tf.float32
    adj_x = False
    adj_y = False
    grad_x = False
    grad_y = False
    name = "matmul11"
    input_dict = {"x": x, "y": y, "Tout": Tout, "adj_x": adj_x, "adj_y": adj_y, "grad_x": grad_x, "grad_y": grad_y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    x = np.array([1, 2, 3, 4], dtype=np.float32)
    y = np.array([5, 6, 7, 8], dtype=np.float32)
    Tout = tf.float32
    adj_x = False
    adj_y = False
    grad_x = False
    grad_y = False
    name = "matmul12"
    input_dict = {"x": x, "y": y, "Tout": Tout, "adj_x": adj_x, "adj_y": adj_y, "grad_x": grad_x, "grad_y": grad_y, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.BatchMatMulV3"] = tf_raw_ops_batch_mat_mul_v3_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.BatchMatMulV3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.BatchMatMulV3'.")

check_valid('tf.raw_ops.BatchMatMulV3', generated_inputs['tf.raw_ops.BatchMatMulV3'], lib="tf", suffix=0)

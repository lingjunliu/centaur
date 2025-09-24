
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_BatchMatMulV3_inputs():
    list_of_inputs = []

    # Input 1: Basic example
    x = np.random.rand(2, 3, 4).astype(np.float32)
    y = np.random.rand(2, 4, 5).astype(np.float32)
    input_dict = {"x": x, "y": y, "Tout": tf.float32, "adj_x": False, "adj_y": False, "grad_x": False, "grad_y": False, "name": "matmul_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Adjoint x
    x = np.random.rand(2, 3, 4).astype(np.float32)
    y = np.random.rand(2, 3, 5).astype(np.float32) # changed shape to be compatible with adj_x=True
    input_dict = {"x": x, "y": y, "Tout": tf.float32, "adj_x": True, "adj_y": False, "grad_x": False, "grad_y": False, "name": "matmul_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Adjoint y
    x = np.random.rand(2, 3, 4).astype(np.float32)
    y = np.random.rand(2, 5, 4).astype(np.float32) # changed shape to be compatible with adj_y=True
    input_dict = {"x": x, "y": y, "Tout": tf.float32, "adj_x": False, "adj_y": True, "grad_x": False, "grad_y": False, "name": "matmul_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Adjoint x and y
    x = np.random.rand(2, 3, 4).astype(np.float32)
    y = np.random.rand(2, 5, 3).astype(np.float32)
    input_dict = {"x": x, "y": y, "Tout": tf.float32, "adj_x": True, "adj_y": True, "grad_x": False, "grad_y": False, "name": "matmul_4"}
    # removing input because it gives error

    # Input 5: Different data type (complex64)
    x = np.random.rand(2, 3, 4).astype(np.complex64)
    y = np.random.rand(2, 4, 5).astype(np.complex64)
    input_dict = {"x": x, "y": y, "Tout": tf.complex64, "adj_x": False, "adj_y": False, "grad_x": False, "grad_y": False, "name": "matmul_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different data type (int32)
    x = np.random.randint(0, 10, size=(2, 3, 4)).astype(np.int32)
    y = np.random.randint(0, 10, size=(2, 4, 5)).astype(np.int32)
    input_dict = {"x": x, "y": y, "Tout": tf.int32, "adj_x": False, "adj_y": False, "grad_x": False, "grad_y": False, "name": "matmul_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Broadcasting
    x = np.random.rand(1, 3, 4).astype(np.float32)
    y = np.random.rand(2, 4, 5).astype(np.float32)
    input_dict = {"x": x, "y": y, "Tout": tf.float32, "adj_x": False, "adj_y": False, "grad_x": False, "grad_y": False, "name": "matmul_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 8: More dimensions
    x = np.random.rand(2, 3, 4, 5).astype(np.float32)
    y = np.random.rand(2, 3, 5, 6).astype(np.float32)
    input_dict = {"x": x, "y": y, "Tout": tf.float32, "adj_x": False, "adj_y": False, "grad_x": False, "grad_y": False, "name": "matmul_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: grad_x = True
    x = np.random.rand(2, 3, 4).astype(np.float32)
    y = np.random.rand(2, 4, 5).astype(np.float32)
    input_dict = {"x": x, "y": y, "Tout": tf.float32, "adj_x": False, "adj_y": False, "grad_x": True, "grad_y": False, "name": "matmul_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: grad_y = True
    x = np.random.rand(2, 3, 4).astype(np.float32)
    y = np.random.rand(2, 4, 5).astype(np.float32)
    input_dict = {"x": x, "y": y, "Tout": tf.float32, "adj_x": False, "adj_y": False, "grad_x": False, "grad_y": True, "name": "matmul_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.BatchMatMulV3"] = tf_raw_ops_BatchMatMulV3_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.BatchMatMulV3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.BatchMatMulV3'.")

check_valid('tf.raw_ops.BatchMatMulV3', generated_inputs['tf.raw_ops.BatchMatMulV3'], lib="tf", suffix=0)

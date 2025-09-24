
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_matmul_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 matrices
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    input_dict = {"a": a, "b": b, "transpose_a": False, "transpose_b": False, "grad_a": False, "grad_b": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: transpose_a = True
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    input_dict = {"a": a, "b": b, "transpose_a": True, "transpose_b": False, "grad_a": False, "grad_b": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: transpose_b = True
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    input_dict = {"a": a, "b": b, "transpose_a": False, "transpose_b": True, "grad_a": False, "grad_b": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Both transpose_a and transpose_b are True
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    input_dict = {"a": a, "b": b, "transpose_a": True, "transpose_b": True, "grad_a": False, "grad_b": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Different shapes, float64
    a = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
    b = np.array([[4.0], [5.0], [6.0]], dtype=np.float64)
    input_dict = {"a": a, "b": b, "transpose_a": False, "transpose_b": False, "grad_a": False, "grad_b": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: int32
    a = np.array([[1, 2], [3, 4]], dtype=np.int32)
    b = np.array([[5, 6], [7, 8]], dtype=np.int32)
    input_dict = {"a": a, "b": b, "transpose_a": False, "transpose_b": False, "grad_a": False, "grad_b": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: int64
    a = np.array([[1, 2], [3, 4]], dtype=np.int64)
    b = np.array([[5, 6], [7, 8]], dtype=np.int64)
    input_dict = {"a": a, "b": b, "transpose_a": False, "transpose_b": False, "grad_a": False, "grad_b": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: complex64
    a = np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex64)
    b = np.array([[5+5j, 6+6j], [7+7j, 8+8j]], dtype=np.complex64)
    input_dict = {"a": a, "b": b, "transpose_a": False, "transpose_b": False, "grad_a": False, "grad_b": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: complex128
    a = np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex128)
    b = np.array([[5+5j, 6+6j], [7+7j, 8+8j]], dtype=np.complex128)
    input_dict = {"a": a, "b": b, "transpose_a": False, "transpose_b": False, "grad_a": False, "grad_b": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: bfloat16
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32).astype(np.float16) #Convert float32 array to float16 numpy array
    b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32).astype(np.float16) #Convert float32 array to float16 numpy array
    input_dict = {"a": a, "b": b, "transpose_a": False, "transpose_b": False, "grad_a": False, "grad_b": False, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.MatMul"] = tf_raw_ops_matmul_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.MatMul' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MatMul'.")

check_valid('tf.raw_ops.MatMul', generated_inputs['tf.raw_ops.MatMul'], lib="tf", suffix=0)

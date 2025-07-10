
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_pinv_inputs():
    list_of_inputs = []

    # Input 1: Basic square matrix
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    rcond = np.array(1e-15, dtype=np.float32)
    validate_args = False
    name = "pinv_basic"
    input_dict = {"a": a, "rcond": rcond, "validate_args": validate_args, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Rectangular matrix (more rows than columns)
    a = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    rcond = np.array(1e-10, dtype=np.float32)
    validate_args = True
    name = "pinv_rect_rows"
    input_dict = {"a": a, "rcond": rcond, "validate_args": validate_args, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Rectangular matrix (more columns than rows)
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    rcond = np.array(1e-5, dtype=np.float32)
    validate_args = False
    name = "pinv_rect_cols"
    input_dict = {"a": a, "rcond": rcond, "validate_args": validate_args, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Singular matrix
    a = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    rcond = np.array(1e-8, dtype=np.float32)
    validate_args = True
    name = "pinv_singular"
    input_dict = {"a": a, "rcond": rcond, "validate_args": validate_args, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Batch of matrices
    a = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    rcond = np.array(1e-7, dtype=np.float32)
    validate_args = False
    name = "pinv_batch"
    input_dict = {"a": a, "rcond": rcond, "validate_args": validate_args, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different rcond value
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    rcond = np.array(0.1, dtype=np.float64)
    validate_args = True
    name = "pinv_rcond"
    input_dict = {"a": a, "rcond": rcond, "validate_args": validate_args, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Larger matrix
    a = np.random.rand(5, 5).astype(np.float32)
    rcond = np.array(1e-6, dtype=np.float32)
    validate_args = False
    name = "pinv_large"
    input_dict = {"a": a, "rcond": rcond, "validate_args": validate_args, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D batch, rectangular
    a = np.random.rand(2, 3, 2).astype(np.float32)
    rcond = np.array(1e-9, dtype=np.float32)
    validate_args = True
    name = "pinv_batch_rect"
    input_dict = {"a": a, "rcond": rcond, "validate_args": validate_args, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: rcond as a Tensor with batch shape
    a = np.random.rand(2, 2, 2).astype(np.float32)
    rcond = np.array([1e-5, 1e-6]).astype(np.float32)
    validate_args = False
    name = "pinv_batch_rcond"
    input_dict = {"a": a, "rcond": rcond, "validate_args": validate_args, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D batch
    a = np.random.rand(2, 2, 3, 3).astype(np.float32)
    rcond = np.array(1e-7, dtype=np.float32)
    validate_args = True
    name = "pinv_4d"
    input_dict = {"a": a, "rcond": rcond, "validate_args": validate_args, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.linalg.pinv"] = tf_linalg_pinv_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.linalg.pinv' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.pinv'.")

check_valid('tf.linalg.pinv', generated_inputs['tf.linalg.pinv'], lib="tf", suffix=0)

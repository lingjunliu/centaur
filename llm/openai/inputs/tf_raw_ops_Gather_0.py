
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_gather_inputs():
    list_of_inputs = []

    params = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    indices = np.array(3, dtype=np.int32)
    input_dict = {
        "validate_indices": True,
        "name": "gather_scalar_idx",
        "params": params,
        "indices": indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    params = np.array([-5, -1, 0, 7, 9], dtype=np.int32)
    indices = np.array([0, 4, 2], dtype=np.int64)
    input_dict = {
        "validate_indices": False,
        "name": "gather_vector_idx_int64",
        "params": params,
        "indices": indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    params = np.arange(12, dtype=np.float64).reshape(3, 4)
    indices = np.array([2, 0], dtype=np.int32)
    input_dict = {
        "validate_indices": True,
        "name": "gather_2d_from_rows",
        "params": params,
        "indices": indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    params = np.arange(2 * 3 * 4, dtype=np.int64).reshape(2, 3, 4)
    indices = np.array([[1, 0], [0, 1]], dtype=np.int32)
    input_dict = {
        "validate_indices": True,
        "name": "gather_matrix_indices",
        "params": params,
        "indices": indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    base = np.arange(3 * 2 * 2 * 2).reshape(3, 2, 2, 2)
    params = (base % 2 == 0)
    indices = np.array([1, 2], dtype=np.int64)
    input_dict = {
        "validate_indices": False,
        "name": "gather_bool",
        "params": params,
        "indices": indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    real = np.arange(6, dtype=np.float32).reshape(2, 3)
    imag = np.arange(6, dtype=np.float32).reshape(2, 3)
    params = (real + 1j * imag).astype(np.complex64)
    indices = np.array(1, dtype=np.int64)
    input_dict = {
        "validate_indices": True,
        "name": "gather_complex",
        "params": params,
        "indices": indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    params = np.arange(10, dtype=np.float16).reshape(5, 1, 2)
    indices = np.array([4, 3, 2, 1, 0], dtype=np.int32)
    input_dict = {
        "validate_indices": True,
        "name": "gather_permute",
        "params": params,
        "indices": indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    params = np.empty((2, 0, 3), dtype=np.float32)
    indices = np.array([1, 0], dtype=np.int32)
    input_dict = {
        "validate_indices": True,
        "name": "gather_zero_len_inner",
        "params": params,
        "indices": indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    params = np.array([[[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]], dtype=np.uint8)
    indices = np.array([0, 0, 0], dtype=np.int64)
    input_dict = {
        "validate_indices": False,
        "name": "gather_repeat",
        "params": params,
        "indices": indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    params = np.array([[1.0, -1.0],
                       [2.0, -2.0],
                       [3.0, -3.0]], dtype=np.float32)
    indices = np.array([[[0, 1]], [[2, 0]]], dtype=np.int64)
    input_dict = {
        "validate_indices": True,
        "name": "gather_3d_indices",
        "params": params,
        "indices": indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    params = np.arange(8, dtype=np.int8).reshape(4, 2)
    indices = np.array(2, dtype=np.int32)
    input_dict = {
        "validate_indices": True,
        "name": "gather_int8_scalar_idx",
        "params": params,
        "indices": indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Gather"] = tf_raw_ops_gather_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Gather' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Gather'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Gather', generated_inputs['tf.raw_ops.Gather'], lib="tf", suffix=0)

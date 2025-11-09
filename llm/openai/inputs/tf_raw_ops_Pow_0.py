
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_pow_inputs():
    list_of_inputs = []

    x = np.array([2.0, 3.0, -4.0], dtype=np.float32)
    y = np.array([3.0, 2.0, 1.5], dtype=np.float32)
    input_dict = {"name": "pow_float32_vector", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1.5, 2.0], [-3.0, 0.5]], dtype=np.float64)
    y = np.array(2.0, dtype=np.float64)
    input_dict = {"name": "pow_float64_broadcast_scalar", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[-2, -1], [0, 1]], [[2, 3], [4, 5]]], dtype=np.int32)
    y = np.array([[[0, 1], [2, 3]], [[4, 0], [1, 2]]], dtype=np.int32)
    input_dict = {"name": "pow_int32_3d", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array(-2, dtype=np.int8)
    y = np.array([0, 1, 2, 3], dtype=np.int8)
    input_dict = {"name": "pow_int8_scalar_vector_broadcast", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([10**6, -2, 0, 1], dtype=np.int64)
    y = np.array([0, 1, 5, 63], dtype=np.int64)
    input_dict = {"name": "pow_int64_vector", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1.0, 2.0, 3.0, 4.0],
                  [5.0, 6.0, 7.0, 8.0],
                  [9.0, 10.0, 11.0, 12.0]], dtype=np.float16)
    y = np.array([0.5, 1.0, 2.0, -1.0], dtype=np.float16)
    input_dict = {"name": "pow_float16_broadcast", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[1+1j, -2+0j], [0+2j, -3-4j]], dtype=np.complex64)
    y = np.array([[2+0j, 0.5+0j], [3+0j, -1+0j]], dtype=np.complex64)
    input_dict = {"name": "pow_complex64_matrix", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[1-1j, 2+3j, -1+2j],
                   [0+1j, -2-2j, 3-1j]]], dtype=np.complex128)
    y = np.array([[[1+0j]], [[-0.5+0.5j]]], dtype=np.complex128)
    input_dict = {"name": "pow_complex128_broadcast_nd", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-32768, -1, 0, 12345], dtype=np.int16)
    y = np.zeros((4,), dtype=np.int16)
    input_dict = {"name": "pow_int16_zeros_exp", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array(3.5, dtype=np.float32)
    y = np.array(-2.0, dtype=np.float32)
    input_dict = {"name": "pow_float32_scalar_negexp", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[2.0], [3.0], [4.0]]], dtype=np.float64)
    y = np.array([[[0.0, 1.0, 2.0, -1.0]],
                  [[1.5, -0.5, 0.25, 3.0]]], dtype=np.float64)
    input_dict = {"name": "pow_float64_broadcast_nd", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-10, -1, 1, 2, 10], dtype=np.int32)
    y = np.ones((5,), dtype=np.int32)
    input_dict = {"name": "pow_int32_ones_exp", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array(0+1j, dtype=np.complex64)
    y = np.array(10+0j, dtype=np.complex64)
    input_dict = {"name": "pow_complex64_scalar", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[0.0, -1.0, 2.5]], dtype=np.float64)
    y = np.array([[3.0, -0.5, 0.0]], dtype=np.float64)
    input_dict = {"name": "pow_float64_edge_cases", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Pow"] = tf_raw_ops_pow_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Pow' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Pow'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Pow', generated_inputs['tf.raw_ops.Pow'], lib="tf", suffix=0)

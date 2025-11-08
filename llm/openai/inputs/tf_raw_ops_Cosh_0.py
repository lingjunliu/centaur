
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_cosh_inputs():
    list_of_inputs = []

    # Input 1: float32 vector with infinities
    x = np.array([-np.inf, -9.0, -0.5, 1.0, 1.2, 2.0, 10.0, np.inf], dtype=np.float32)
    name = "cosh_case_float32_vector_inf"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64 scalar
    x = np.array(0.0, dtype=np.float64)
    name = "cosh_case_float64_scalar_zero"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: float16 2D matrix with negatives and positives
    x = np.array([[-1.0, -2.0], [3.0, 4.0]], dtype=np.float16)
    name = "cosh_case_float16_matrix"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: float32 3D tensor
    x = np.array([[[0.1, -0.2, 0.3]], [[-0.4, 0.5, -0.6]]], dtype=np.float32)
    name = "cosh_case_float32_3d"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: complex64 vector
    x = np.array([1+0j, 0+1j, -1+2j, -3-4j], dtype=np.complex64)
    name = "cosh_case_complex64_vector"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: complex128 2D matrix
    x = np.array([[1+1j, -2+3j], [0-1j, -0.5+0.5j]], dtype=np.complex128)
    name = "cosh_case_complex128_matrix"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: float32 large magnitudes
    x = np.array([100.0, -100.0, 50.0, -50.0], dtype=np.float32)
    name = "cosh_case_float32_large_magnitudes"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: float32 empty vector
    x = np.array([], dtype=np.float32)
    name = "cosh_case_float32_empty"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float64 empty first-dimension (0x3)
    x = np.zeros((0, 3), dtype=np.float64)
    name = "cosh_case_float64_empty_rows"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float32 with NaNs
    x = np.array([np.nan, -np.nan, 1.0, -1.0], dtype=np.float32)
    name = "cosh_case_float32_nans"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: float32 very small values (subnormals)
    x = np.array([1e-8, -1e-8, 1e-10, -1e-10], dtype=np.float32)
    name = "cosh_case_float32_small_values"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: float32 4D tensor
    x = (np.arange(24, dtype=np.float32).reshape(2, 3, 2, 2) - 12.0) / 5.0
    name = "cosh_case_float32_4d"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Cosh"] = tf_raw_ops_cosh_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Cosh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Cosh'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Cosh', generated_inputs['tf.raw_ops.Cosh'], lib="tf", suffix=0)

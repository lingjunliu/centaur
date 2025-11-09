
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_math_pow_inputs():
    list_of_inputs = []

    # Input 1: float32 vector and scalar exponent (0-D)
    x = np.array([1.0, 2.0, 3.0, 4.0, -5.0], dtype=np.float32)
    y = np.array(2.5, dtype=np.float32)
    name = "pow_float32_vec_scalar"
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y, "name": name}))

    # Input 2: float64 matrix elementwise
    x = np.array([[2.0, 3.0, 4.0], [0.5, 1.5, -2.5]], dtype=np.float64)
    y = np.array([[8.0, 2.0, 0.5], [3.0, 0.0, 1.0]], dtype=np.float64)
    name = "pow_float64_matrix"
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y, "name": name}))

    # Input 3: int32 2x2 positive exponents
    x = np.array([[2, 3], [4, 5]], dtype=np.int32)
    y = np.array([[3, 2], [1, 0]], dtype=np.int32)
    name = "pow_int32_2x2"
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y, "name": name}))

    # Input 4: int64 broadcasting (2,1,3) with (1,4,1)
    x = np.array([[[1, 2, 3]], [[4, 5, 6]]], dtype=np.int64)  # shape (2,1,3)
    y = np.array([[[0], [1], [2], [3]]], dtype=np.int64)      # shape (1,4,1)
    name = "pow_int64_broadcast"
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y, "name": name}))

    # Input 5: complex64 vector elementwise complex exponents
    x = np.array([1+2j, -1+1j, 0+1j, 2-3j], dtype=np.complex64)
    y = np.array([2+0j, 0.5+0.5j, -1+0j, 1-1j], dtype=np.complex64)
    name = "pow_complex64_vector"
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y, "name": name}))

    # Input 6: complex128 matrix with scalar exponent
    x = np.array([[1+0j, -2+2j], [3-1j, -4-4j]], dtype=np.complex128)
    y = np.array(2-0.5j, dtype=np.complex128)
    name = "pow_complex128_scalar_exp"
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y, "name": name}))

    # Input 7: float16 3-D arrays
    x = np.array([[[1.0, 2.0], [3.0, 4.0]],
                  [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float16)
    y = np.array([[[0.5, 1.0], [1.5, 2.0]],
                  [[2.5, 3.0], [0.0, -1.0]]], dtype=np.float16)
    name = "pow_float16_3d"
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y, "name": name}))

    # Input 8: float32 negative bases with fractional exponents
    x = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    y = np.array([0.5, 1.5, 2.5], dtype=np.float32)
    name = "pow_float32_negative_bases_fractional_exp"
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y, "name": name}))

    # Input 9: float32 broadcasting (2,3,1) with (1,3,4)
    x = np.array([[[1.0], [2.0], [3.0]],
                  [[4.0], [5.0], [6.0]]], dtype=np.float32)  # (2,3,1)
    y = np.array([[[1.0, 2.0, 3.0, 4.0],
                   [0.5, 1.5, 2.5, 3.5],
                   [0.0, -1.0, -2.0, -3.0]]], dtype=np.float32)  # (1,3,4)
    name = "pow_float32_broadcast_2"
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y, "name": name}))

    # Input 10: float64 zeros and zero exponent
    x = np.array([0.0, 1.0, -1.0], dtype=np.float64)
    y = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    name = "pow_float64_zero_exp"
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y, "name": name}))

    # Input 11: int32 vector with zero exponents
    x = np.array([10, -3, 0, 2], dtype=np.int32)
    y = np.array([0, 0, 0, 0], dtype=np.int32)
    name = "pow_int32_zero_exp"
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y, "name": name}))

    # Input 12: float32 scalar base with vector exponents (broadcast)
    x = np.array(1.1, dtype=np.float32)
    y = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    name = "pow_float32_scalar_base_vector_exp"
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y, "name": name}))

    return list_of_inputs

generated_inputs["tf.math.pow"] = tf_math_pow_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.pow' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.pow'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.pow', generated_inputs['tf.math.pow'], lib="tf", suffix=0)

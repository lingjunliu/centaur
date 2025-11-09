
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_LinearOperatorLowerTriangular_inputs():
    list_of_inputs = []

    # Input 1
    tril = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    input_dict = {
        "tril": tril,
        "is_non_singular": True,
        "is_self_adjoint": False,
        "is_positive_definite": False,
        "is_square": True,
        "name": "lt1_float32_2x2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    tril = np.diag(np.array([1.0, 2.0, 3.0], dtype=np.float64))
    input_dict = {
        "tril": tril,
        "is_non_singular": True,
        "is_self_adjoint": True,
        "is_positive_definite": True,
        "is_square": True,
        "name": "lt2_float64_diag_3x3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    tril = np.array([
        [0.0, -1.0, 2.0, 3.0],
        [5.0, -1.0, 0.5, 2.0],
        [-7.0, 8.0, 2.0, -3.0],
        [4.0, -6.0, 1.0, 0.5]
    ], dtype=np.float32)
    input_dict = {
        "tril": tril,
        "is_non_singular": False,
        "is_self_adjoint": False,
        "is_positive_definite": False,
        "is_square": True,
        "name": "lt3_float32_4x4_singular_hint"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4 (batch [2, 2, 2])
    tril = np.array([
        [[2.0, -1.0], [0.3, -4.0]],
        [[-3.0, 5.0], [1.2, 6.0]]
    ], dtype=np.float64)
    input_dict = {
        "tril": tril,
        "is_non_singular": True,
        "is_self_adjoint": False,
        "is_positive_definite": False,
        "is_square": True,
        "name": "lt4_float64_batch2_2x2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5 (batch [3, 4, 4])
    tril = np.random.randn(3, 4, 4).astype(np.float32)
    input_dict = {
        "tril": tril,
        "is_non_singular": False,
        "is_self_adjoint": False,
        "is_positive_definite": False,
        "is_square": True,
        "name": "lt5_float32_batch3_4x4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6 (batch [2, 3, 4, 4])
    tril = np.random.randn(2, 3, 4, 4).astype(np.float64)
    input_dict = {
        "tril": tril,
        "is_non_singular": True,
        "is_self_adjoint": False,
        "is_positive_definite": False,
        "is_square": True,
        "name": "lt6_float64_batch2x3_4x4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7 (complex64, 2x2 diagonal positive)
    tril = np.array([[2.0 + 0.0j, 0.0 + 0.0j],
                     [0.0 + 0.0j, 3.0 + 0.0j]], dtype=np.complex64)
    input_dict = {
        "tril": tril,
        "is_non_singular": True,
        "is_self_adjoint": True,
        "is_positive_definite": True,
        "is_square": True,
        "name": "lt7_complex64_diag_2x2_pd"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8 (complex128 batch [2, 2, 2])
    tril = np.array([
        [[1.0 + 2.0j, -3.0 + 0.0j],
         [4.0 - 1.0j, -2.0 + 0.5j]],
        [[-1.0 + 0.0j, 2.0 + 1.0j],
         [3.0 + 4.0j, 1.0 - 2.0j]]
    ], dtype=np.complex128)
    input_dict = {
        "tril": tril,
        "is_non_singular": False,
        "is_self_adjoint": False,
        "is_positive_definite": False,
        "is_square": True,
        "name": "lt8_complex128_batch2_2x2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9 (float16 5x5)
    tril = np.tril(np.array([
        [0.5, -1.2, 0.0, 2.1, -3.0],
        [1.0, 0.8, -0.7, 0.0, 1.5],
        [2.2, -1.1, 1.3, -0.4, 0.0],
        [0.0, 1.7, -2.5, 0.6, 0.2],
        [-1.4, 0.0, 0.9, -0.8, 2.0]
    ], dtype=np.float16))
    input_dict = {
        "tril": tril,
        "is_non_singular": True,
        "is_self_adjoint": False,
        "is_positive_definite": False,
        "is_square": True,
        "name": "lt9_float16_5x5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10 (batch [1, 2, 3, 3])
    tril = np.arange(1, 1 + 1*2*3*3, dtype=np.float32).reshape(1, 2, 3, 3)
    input_dict = {
        "tril": tril,
        "is_non_singular": True,
        "is_self_adjoint": False,
        "is_positive_definite": False,
        "is_square": True,
        "name": "lt10_float32_batch1x2_3x3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11 (1x1 negative)
    tril = np.array([[-2.0]], dtype=np.float64)
    input_dict = {
        "tril": tril,
        "is_non_singular": True,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "name": "lt11_float64_1x1_negative"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12 (batch [2, 5, 5])
    tril = np.tril(np.random.uniform(-5, 5, size=(2, 5, 5)).astype(np.float32))
    input_dict = {
        "tril": tril,
        "is_non_singular": False,
        "is_self_adjoint": False,
        "is_positive_definite": False,
        "is_square": True,
        "name": "lt12_float32_batch2_5x5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.linalg.LinearOperatorLowerTriangular"] = tf_linalg_LinearOperatorLowerTriangular_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.linalg.LinearOperatorLowerTriangular' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.LinearOperatorLowerTriangular'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.linalg.LinearOperatorLowerTriangular', generated_inputs['tf.linalg.LinearOperatorLowerTriangular'], lib="tf", suffix=0)

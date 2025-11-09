
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)
np.random.seed(42)

def tf_linalg_linearoperatorhouseholder_inputs():
    list_of_inputs = []

    # Input 1
    vec1 = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=np.float64)
    input_dict = {
        "reflection_axis": vec1,
        "is_non_singular": True,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "name": "hh_vec2_f64_1d"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    vec2 = np.array([[3.0], [-4.0], [0.0]], dtype=np.float32)
    input_dict = {
        "reflection_axis": vec2,
        "is_non_singular": True,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "name": "hh_col3_f32_2d"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    rs3 = np.random.RandomState(0)
    vec3 = rs3.randn(5, 4).astype(np.float32)
    input_dict = {
        "reflection_axis": vec3,
        "is_non_singular": True,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "name": "hh_batch5x4_f32"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    rs4 = np.random.RandomState(1)
    vec4 = rs4.randn(2, 3, 6, 1).astype(np.float64)
    input_dict = {
        "reflection_axis": vec4,
        "is_non_singular": True,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "name": "hh_batch2x3_col6_f64"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    vec5 = np.array([1 + 2j, -0.5 + 0.3j, 2 - 1j, 0 - 3j], dtype=np.complex64)
    input_dict = {
        "reflection_axis": vec5,
        "is_non_singular": True,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "name": "hh_vec4_c64_1d"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    rs6a = np.random.RandomState(2)
    rs6b = np.random.RandomState(3)
    vec6 = (rs6a.randn(3, 5) + 1j * rs6b.randn(3, 5)).astype(np.complex128)
    input_dict = {
        "reflection_axis": vec6,
        "is_non_singular": True,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "name": "hh_batch3x5_c128"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    vec7 = np.array([-2.5], dtype=np.float32)
    input_dict = {
        "reflection_axis": vec7,
        "is_non_singular": True,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "name": "hh_scalar1_f32"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    vec8 = np.array([((-1.0) ** i) * (i + 1) / 10.0 for i in range(10)], dtype=np.float16)
    input_dict = {
        "reflection_axis": vec8,
        "is_non_singular": True,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "name": "hh_vec10_f16"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    vec9 = np.array([1e-12, -1e-11, 2e-12, -3e-12, 4e-12, -5e-12, 6e-12, -7e-12], dtype=np.float32)
    input_dict = {
        "reflection_axis": vec9,
        "is_non_singular": True,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "name": "hh_small_vec8_f32"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    rs10 = np.random.RandomState(4)
    vec10 = rs10.randn(2, 2, 2, 7).astype(np.float64)
    input_dict = {
        "reflection_axis": vec10,
        "is_non_singular": True,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "name": "hh_batch2x2x2_vec7_f64"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    vec11 = np.array([[0 + 1j], [0 + 2j], [0 - 3j], [0 + 4j]], dtype=np.complex64)
    input_dict = {
        "reflection_axis": vec11,
        "is_non_singular": True,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "name": "hh_col4_imag_c64"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12
    vec12 = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    input_dict = {
        "reflection_axis": vec12,
        "is_non_singular": True,
        "is_self_adjoint": True,
        "is_positive_definite": False,
        "is_square": True,
        "name": "hh_onehot4_f32"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.linalg.LinearOperatorHouseholder"] = tf_linalg_linearoperatorhouseholder_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.linalg.LinearOperatorHouseholder' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.LinearOperatorHouseholder'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.linalg.LinearOperatorHouseholder', generated_inputs['tf.linalg.LinearOperatorHouseholder'], lib="tf", suffix=0)

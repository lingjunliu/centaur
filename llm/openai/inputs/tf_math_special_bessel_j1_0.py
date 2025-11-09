
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_math_special_bessel_j1_inputs():
    list_of_inputs = []

    x = np.array([0.5, 1.0, 2.0, 4.0], dtype=np.float32)
    name = "case_vec_f32_basic"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[0.0, -1.5, 3.2], [4.5, -6.7, 8.9]], dtype=np.float64)
    name = "case_matrix_f64_negpos"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array(0.0, dtype=np.float32)
    name = "case_scalar_zero_f32"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[0.1, 0.2], [0.3, 0.4]], [[-0.5, -0.6], [0.7, -0.8]]], dtype=np.float16)
    name = "case_3d_f16_small_vals"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([], dtype=np.float32)
    name = "case_empty_1d_f32"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1e-6, 1e-3, 1e-1, 1.0, 10.0, 100.0, 1e3], dtype=np.float64)
    name = "case_wide_range_f64"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-0.1, -1.0, -5.5, -10.0], dtype=np.float32)
    name = "case_negatives_f32"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    base = np.arange(20, dtype=np.float64)
    x = base[::2]
    name = "case_noncontiguous_view_f64"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = (np.arange(6, dtype=np.float32).reshape(1, 2, 1, 3) / 10.0)
    name = "case_4d_f32_arange"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.empty((0, 3), dtype=np.float32)
    name = "case_empty_2d_f32"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.linspace(-3.0, 3.0, num=7, dtype=np.float32)
    name = "case_linspace_f32"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.arange(4, dtype=np.float64).reshape(1, 1, 2, 1, 2) - 1.5
    name = "case_5d_f64_shifted"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.math.special.bessel_j1"] = tf_math_special_bessel_j1_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.special.bessel_j1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.special.bessel_j1'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.special.bessel_j1', generated_inputs['tf.math.special.bessel_j1'], lib="tf", suffix=0)

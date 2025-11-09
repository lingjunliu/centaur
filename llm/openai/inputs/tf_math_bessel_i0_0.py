
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

np.random.seed(42)

def tf_math_bessel_i0_inputs():
    list_of_inputs = []

    x = np.array(0.0, dtype=np.float32)
    name = "i0_scalar_f32_zero"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array([-1.0, -0.5, 0.5, 1.0], dtype=np.float32)
    name = "i0_vector_f32_mixed"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array([[0.1, -0.2], [3.0, -4.0]], dtype=np.float64)
    name = "i0_matrix_f64_small_big"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.random.uniform(-2, 2, size=(2, 3, 4)).astype(np.float16)
    name = "i0_tensor3d_f16_uniform"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array([], dtype=np.float32)
    name = "i0_empty_vector_f32"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array([20.0, 50.0, -20.0, -50.0], dtype=np.float64)
    name = "i0_large_magnitude_f64"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.arange(-5, 5, dtype=np.float32)[::2]
    name = "i0_noncontiguous_slice_f32"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    base = np.arange(12, dtype=np.float32).reshape(3, 4)
    x = base.T
    name = "i0_transposed_view_f32"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.zeros((3, 3), dtype=np.float16)
    name = "i0_zeros_matrix_f16"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array([np.nan, np.inf, -np.inf, 0.0], dtype=np.float64)
    name = "i0_nan_inf_f64"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array([1e-8, -1e-8, 1e-12, -1e-12], dtype=np.float32)
    name = "i0_very_small_values_f32"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.linspace(-3, 3, 12, dtype=np.float64).reshape(2, 1, 2, 3)
    name = "i0_tensor4d_f64_linspace"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    return list_of_inputs

generated_inputs["tf.math.bessel_i0"] = tf_math_bessel_i0_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.bessel_i0' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.bessel_i0'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.bessel_i0', generated_inputs['tf.math.bessel_i0'], lib="tf", suffix=0)

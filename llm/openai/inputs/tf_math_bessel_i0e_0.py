
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)
np.random.seed(42)

def tf_math_bessel_i0e_inputs():
    list_of_inputs = []

    x = np.array([-1.0, -0.5, 0.5, 1.0], dtype=np.float32)
    name = "vector_f32_basic"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array([[0.0, 2.0, -2.0], [10.0, -10.0, 1e-3]], dtype=np.float64)
    name = "matrix_f64_mixed_vals"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array(3.0, dtype=np.float32)
    name = "scalar_f32_positive"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array([], dtype=np.float32)
    name = "empty_vector_f32"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.linspace(-20.0, 20.0, 9, dtype=np.float64)
    name = "linspace_f64_wide_range"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.arange(-5, 6, dtype=np.float32)
    name = "range_f32_neg_to_pos"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.random.uniform(-50, 50, size=(3, 4, 2)).astype(np.float16)
    name = "random_3d_f16"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array([[np.inf, -np.inf, np.nan]], dtype=np.float32)
    name = "special_values_inf_nan_f32"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x_base = np.arange(24, dtype=np.float64).reshape(4, 6)
    x = x_base[:, ::2] - 12.5
    name = "noncontiguous_slice_f64"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.zeros((0, 3), dtype=np.float32)
    name = "empty_2d_rows_f32"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array([1e-8, -1e-8, 1e-12, -1e-12], dtype=np.float64)
    name = "very_small_values_f64"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.array([100.0, -100.0, 300.0, -300.0], dtype=np.float32)
    name = "large_magnitude_f32"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = (np.eye(5, dtype=np.float32) - 0.5)
    name = "identity_shifted_f32"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    x = np.transpose(np.arange(8, dtype=np.float16).reshape(2, 2, 2), (1, 0, 2)).astype(np.float16) - 4
    name = "transposed_3d_f16"
    list_of_inputs.append(copy.deepcopy({"x": x, "name": name}))

    return list_of_inputs

generated_inputs["tf.math.bessel_i0e"] = tf_math_bessel_i0e_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.bessel_i0e' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.bessel_i0e'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.bessel_i0e', generated_inputs['tf.math.bessel_i0e'], lib="tf", suffix=0)

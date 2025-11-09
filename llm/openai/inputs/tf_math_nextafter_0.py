
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_nextafter_inputs():
    list_of_inputs = []

    x1 = np.array([0.0, -1.0, 3.5], dtype=np.float32)
    x2 = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    name = "simple_1d_f32"
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2, "name": name}))

    x1 = np.array([[1.0, 2.0], [-3.0, 4.0]], dtype=np.float64)
    x2 = np.array([[2.0, 1.0], [-4.0, 3.0]], dtype=np.float64)
    name = "matrix_f64"
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2, "name": name}))

    x1 = np.array([[0.0], [1.0], [-1.0]], dtype=np.float32)
    x2 = np.array([[1.0, 0.0, -2.0, 3.0]], dtype=np.float32)
    name = "broadcast_f32"
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2, "name": name}))

    x1 = np.array(-0.0, dtype=np.float64)
    x2 = np.array(0.0, dtype=np.float64)
    name = "negzero_to_poszero_scalar_f64"
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2, "name": name}))

    x1 = np.array([np.inf, -np.inf, 1.0], dtype=np.float64)
    x2 = np.array([0.0, 0.0, np.inf], dtype=np.float64)
    name = "infinities_f64"
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2, "name": name}))

    x1 = np.array([np.nan, 1.0, -2.0], dtype=np.float32)
    x2 = np.array([2.0, np.nan, -3.0], dtype=np.float32)
    name = "nans_mixed_f32"
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2, "name": name}))

    min_subnormal_f32 = np.nextafter(np.float32(0.0), np.float32(1.0))
    x1 = np.zeros((2, 2, 3), dtype=np.float32)
    x2 = np.full((2, 2, 3), min_subnormal_f32, dtype=np.float32)
    name = "zeros_towards_min_subnormal_f32"
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2, "name": name}))

    base = np.linspace(-5.0, 6.0, 12, dtype=np.float64).reshape(3, 4)
    x1 = base[:, ::2]
    x2 = x1 + 0.1
    name = "non_contiguous_f64"
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2, "name": name}))

    x1 = np.array([3.4e38, -3.4e38], dtype=np.float32)
    x2 = np.array([np.inf, -np.inf], dtype=np.float32)
    name = "near_max_f32"
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2, "name": name}))

    x1 = np.array([[[[-1.0, 0.0, 1.0]]], [[[2.0, -2.0, 0.5]]]], dtype=np.float64)
    x2 = np.arange(12, dtype=np.float64).reshape(1, 3, 4, 1)
    name = "broadcast_4d_f64"
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2, "name": name}))

    x1 = np.array([0.1, -0.1, 0.0], dtype=np.float32)
    x2 = np.array([0.1, -0.1, 0.0], dtype=np.float32)
    name = "x1_equals_x2_f32"
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2, "name": name}))

    x1 = np.array([], dtype=np.float64)
    x2 = np.array([], dtype=np.float64)
    name = "empty_f64"
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2, "name": name}))

    min_subnormal_f64 = np.nextafter(np.float64(0.0), np.float64(1.0))
    x1 = np.array([-0.0, 0.0, min_subnormal_f64], dtype=np.float64)
    x2 = np.array([-1.0, 1.0, 0.0], dtype=np.float64)
    name = "signed_zero_and_subnormal_f64"
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2, "name": name}))

    x1 = np.linspace(-1.0, 1.0, 5, dtype=np.float32)
    x2 = np.array(0.0, dtype=np.float32)
    name = "vector_towards_scalar_f32"
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2, "name": name}))

    return list_of_inputs

generated_inputs["tf.math.nextafter"] = tf_math_nextafter_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.nextafter' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.nextafter'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.nextafter', generated_inputs['tf.math.nextafter'], lib="tf", suffix=0)

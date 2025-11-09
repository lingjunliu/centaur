
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_math_is_inf_inputs():
    list_of_inputs = []

    x = np.array([5.0, np.inf, 6.8, np.inf], dtype=np.float32)
    input_dict = {"x": x, "name": "simple_1d_f32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([np.NINF, -1.0, 0.0, 1.0, np.PINF], dtype=np.float64)
    input_dict = {"x": x, "name": "neg_and_pos_inf_f64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array(np.inf, dtype=np.float16)
    input_dict = {"x": x, "name": "scalar_inf_f16"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[-np.inf, np.nan], [3.4, np.inf]], dtype=np.float16)
    input_dict = {"x": x, "name": "matrix_with_nan_and_inf_f16"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array(
        [
            [[0.0, 1.0, np.inf], [-np.inf, 5.5, np.nan]],
            [[2.2, 3.3, 4.4], [6.6, -7.7, np.inf]],
        ],
        dtype=np.float32,
    )
    input_dict = {"x": x, "name": "tensor3d_mixed_f32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([], dtype=np.float64)
    input_dict = {"x": x, "name": "empty_1d_f64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.empty((0, 3), dtype=np.float32)
    input_dict = {"x": x, "name": "empty_2d_f32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1e308, -1e308, 1e309, -1e309], dtype=np.float64)
    input_dict = {"x": x, "name": "extremal_values_f64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[[np.inf, -np.inf, 0.0]], [[1.0, 2.0, 3.0]]]], dtype=np.float32)
    input_dict = {"x": x, "name": "four_d_shape_f32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    base = np.linspace(-5, 5, 10, dtype=np.float32)
    x = base[::2].copy()
    x[1] = np.inf
    x[3] = -np.inf
    input_dict = {"x": x, "name": "strided_view_f32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([65504.0, 70000.0, -70000.0, -65504.0], dtype=np.float16)
    input_dict = {"x": x, "name": "f16_overflow_values"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-1.0, -2.5, -3.14, -0.0], dtype=np.float32)
    input_dict = {"x": x, "name": "finite_negatives_only_f32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.math.is_inf"] = tf_math_is_inf_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.is_inf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.is_inf'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.is_inf', generated_inputs['tf.math.is_inf'], lib="tf", suffix=0)

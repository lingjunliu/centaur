
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_math_lgamma_inputs():
    list_of_inputs = []

    x = np.array([0.0, 0.5, 1.0, 4.5, -4.0, -5.6], dtype=np.float32)
    name = "case1_vector32"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array(5.0, dtype=np.float64)
    name = "case2_scalar64"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.arange(1, 10, dtype=np.float32).reshape(3, 3)
    name = "case3_matrix32"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array(
        [
            [[-2.5, -1.2, 0.0], [0.3, 2.0, 3.3]],
            [[-0.5, 1.5, 4.0], [5.5, -3.7, 7.2]],
        ],
        dtype=np.float64,
    )
    name = "case4_tensor3d64"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-10.5, -1.5, 0.1, 10.0], dtype=np.float16)
    name = "case5_vector16"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([20.0, 50.0, 100.0], dtype=np.float64)
    name = "case6_large_values64"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([np.nan, np.inf, -np.inf, 3.14, -3.14], dtype=np.float32)
    name = "case7_special32"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[[[0.2, 1.2, 2.2]], [[-0.2, -1.2, -2.2]]]], dtype=np.float32)
    name = "case8_tensor4d32"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([], dtype=np.float32)
    name = "case9_empty32"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([-0.999, -1.001, -1.999, -2.001, -10.0001], dtype=np.float64)
    name = "case10_near_neg_integers64"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([1e-10, 1e-20, 1e-5, 1e-2], dtype=np.float64)
    name = "case11_small_positive64"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x = np.array([[0.25, 0.75], [1.25, 1.75], [-0.25, -0.75]], dtype=np.float32)
    name = "case12_rect32"
    input_dict = {"x": x, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.math.lgamma"] = tf_math_lgamma_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.lgamma' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.lgamma'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.lgamma', generated_inputs['tf.math.lgamma'], lib="tf", suffix=0)

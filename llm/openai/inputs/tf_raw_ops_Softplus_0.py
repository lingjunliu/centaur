
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def tf_raw_ops_Softplus_inputs():
    list_of_inputs = []

    features = np.array(-1.5, dtype=np.float32)
    name = "softplus_scalar_f32_neg"
    input_dict = {"name": name, "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([-10.0, -1.0, 0.0, 1.0, 10.0], dtype=np.float32)
    name = "softplus_vector_f32_mixed"
    input_dict = {"name": name, "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([[-100.0, -5.0, 0.0], [1.0, 5.0, 100.0]], dtype=np.float64)
    name = "softplus_matrix_f64_extremes"
    input_dict = {"name": name, "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([[[-2.0, -0.1, 0.0, 0.1]], [[2.0, 10.0, -10.0, 3.0]]], dtype=np.float16)
    name = "softplus_3d_f16"
    input_dict = {"name": name, "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([[[[-8.0, -1.0], [0.0, 1.0]], [[2.0, 5.0], [10.0, 15.0]]]], dtype=np.float32)
    name = "softplus_4d_f32"
    input_dict = {"name": name, "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([], dtype=np.float32)
    name = "softplus_empty_1d_f32"
    input_dict = {"name": name, "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.empty((0, 3), dtype=np.float64)
    name = "softplus_empty_2d_f64"
    input_dict = {"name": name, "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([15.0, -15.0, 12.0, -12.0, 0.0], dtype=np.float16)
    name = "softplus_extreme_f16"
    input_dict = {"name": name, "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([np.nan, np.inf, -np.inf, 0.0, -3.5], dtype=np.float32)
    name = "softplus_specials_f32"
    input_dict = {"name": name, "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    base = np.arange(12, dtype=np.float64).reshape(3, 4)
    features = base[::2, ::2]
    name = "softplus_noncontiguous_f64"
    input_dict = {"name": name, "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array(
        [
            [[[[ -2.0, -1.0, 0.0], [1.0, 2.0, 3.0]]]],
            [[[[10.0, -10.0, 0.5], [-0.5, 3.0, -3.0]]]]
        ],
        dtype=np.float64
    )
    name = "softplus_5d_f64"
    input_dict = {"name": name, "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Softplus"] = tf_raw_ops_Softplus_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Softplus' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Softplus'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Softplus', generated_inputs['tf.raw_ops.Softplus'], lib="tf", suffix=0)

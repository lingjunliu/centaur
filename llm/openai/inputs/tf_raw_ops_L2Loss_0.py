
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)
np.random.seed(42)

def tf_raw_ops_L2Loss_inputs():
    list_of_inputs = []

    t = np.array(3.5, dtype=np.float32)
    input_dict = {"name": "scalar_f32", "t": t}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = np.array([-3.0, -1.5, 0.0, 2.5, 4.0], dtype=np.float32)
    input_dict = {"name": "vector_f32_neg_pos", "t": t}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = np.array([[-1.0, 2.0, -3.0], [4.5, 0.0, -6.5]], dtype=np.float64)
    input_dict = {"name": "matrix_f64", "t": t}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = np.random.randn(2, 3, 4).astype(np.float16)
    input_dict = {"name": "tensor3d_f16", "t": t}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = np.array([], dtype=np.float32)
    input_dict = {"name": "empty_1d_f32", "t": t}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = np.empty((2, 0), dtype=np.float32)
    input_dict = {"name": "empty_2d_f32", "t": t}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = np.array([1e154, -1e154, 3.14e153], dtype=np.float64)
    input_dict = {"name": "large_vals_f64", "t": t}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = np.array([np.nan, np.inf, -np.inf, 1.0, -2.0], dtype=np.float32)
    input_dict = {"name": "nan_inf_f32", "t": t}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = (np.ones((1, 2, 1, 3, 2), dtype=np.float32) * -0.75).astype(np.float32)
    input_dict = {"name": "high_dim_f32", "t": t}
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.arange(30, dtype=np.float64).reshape(5, 6)
    t = a[::2, ::2]
    input_dict = {"name": "non_contiguous_slice_f64", "t": t}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = np.array([1e-5, -1e-5, 2e-6, -2e-6], dtype=np.float16)
    input_dict = {"name": "subnormal_like_f16", "t": t}
    list_of_inputs.append(copy.deepcopy(input_dict))

    t = np.random.uniform(-5.0, 5.0, size=(2, 2, 2, 2)).astype(np.float32)
    input_dict = {"name": "random_4d_f32", "t": t}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.L2Loss"] = tf_raw_ops_L2Loss_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.L2Loss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.L2Loss'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.L2Loss', generated_inputs['tf.raw_ops.L2Loss'], lib="tf", suffix=0)

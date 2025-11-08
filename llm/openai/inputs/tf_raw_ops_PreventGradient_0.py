
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_preventgradient_inputs():
    list_of_inputs = []

    x1 = np.array([1, 2, 3], dtype=np.int32)
    input_dict = {"message": "no gradient for int32", "name": "pg_int32_vec", "input": x1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x2 = np.array([-1.5, 0.0, 2.5], dtype=np.float32)
    input_dict = {"message": "no gradient for float32", "name": "pg_float32_vec", "input": x2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x3 = np.array([[True, False], [False, True]], dtype=np.bool_)
    input_dict = {"message": "block grad on bool", "name": "pg_bool_mat", "input": x3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x4 = np.array(42, dtype=np.int64)
    input_dict = {"message": "scalar int64 no grad", "name": "pg_int64_scalar", "input": x4}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x5 = np.array([[1+2j, 3-4j]], dtype=np.complex64)
    input_dict = {"message": "complex64 not differentiable here", "name": "pg_complex64_row", "input": x5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x6 = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    input_dict = {"message": "float64 3D no grad", "name": "pg_float64_3d", "input": x6}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x7 = np.array([b'alpha', b'beta', b'gamma'], dtype=object)
    input_dict = {"message": "string tensor no grad", "name": "pg_string_vec", "input": x7}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x8 = np.array([], dtype=np.int32)
    input_dict = {"message": "empty int32 vector", "name": "pg_empty_int32", "input": x8}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x9 = np.ones((5, 0, 3), dtype=np.float32)
    input_dict = {"message": "empty-dim float32", "name": "pg_empty_dim", "input": x9}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x10 = np.array([[[1], [2]], [[3], [4]]], dtype=np.uint8)
    input_dict = {"message": "uint8 small 3D", "name": "pg_uint8_3d", "input": x10}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x11 = np.array(-3.14, dtype=np.float16)
    input_dict = {"message": "float16 scalar", "name": "pg_float16_scalar", "input": x11}
    list_of_inputs.append(copy.deepcopy(input_dict))

    x12 = np.linspace(0, 1, 7, dtype=np.float32).reshape(7, 1)
    input_dict = {"message": "linspace float32", "name": "pg_float32_col", "input": x12}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.PreventGradient"] = tf_raw_ops_preventgradient_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.PreventGradient' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.PreventGradient'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.PreventGradient', generated_inputs['tf.raw_ops.PreventGradient'], lib="tf", suffix=0)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_debuggradientidentity_inputs():
    list_of_inputs = []

    arr = np.array(3.14, dtype=np.float32)
    input_dict = {"name": "scalar_f32_pi", "input": arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array([-5, 0, 7, 42], dtype=np.int32)
    input_dict = {"name": "vec_i32_mixed", "input": arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array([[np.nan, -np.inf], [np.inf, -1.5]], dtype=np.float64)
    input_dict = {"name": "mat_f64_nan_inf", "input": arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array([[[True, False, True], [False, False, True]], [[True, True, False], [False, True, False]]], dtype=bool)
    input_dict = {"name": "tensor_bool_3d", "input": arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array([], dtype=np.float32)
    input_dict = {"name": "empty_f32", "input": arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.zeros((1, 2, 0, 3), dtype=np.int64)
    input_dict = {"name": "int64_zero_dim_axis", "input": arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array([1+2j, -3+0.5j, 0+0j], dtype=np.complex64)
    input_dict = {"name": "vec_c64", "input": arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array([[-1.2, 3.4], [0.0, 65504.0]], dtype=np.float16)
    input_dict = {"name": "mat_f16_range", "input": arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.arange(12, dtype=np.uint8).reshape(2, 1, 2, 3)
    input_dict = {"name": "tensor_u8_4d", "input": arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    arr = np.array(3-4j, dtype=np.complex128)
    input_dict = {"name": "scalar_c128", "input": arr}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.DebugGradientIdentity"] = tf_raw_ops_debuggradientidentity_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.DebugGradientIdentity' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DebugGradientIdentity'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.DebugGradientIdentity', generated_inputs['tf.raw_ops.DebugGradientIdentity'], lib="tf", suffix=0)

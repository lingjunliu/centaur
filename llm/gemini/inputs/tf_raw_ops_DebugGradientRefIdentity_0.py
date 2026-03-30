
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_debug_gradient_ref_identity_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 tensor
    input1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict1 = {"input": tf.compat.v1.Variable(input1), "name": "float32_1d"}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: 2D int32 tensor
    input2 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    input_dict2 = {"input": tf.compat.v1.Variable(input2), "name": "int32_2d"}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Scalar bool tensor
    input3 = np.array(True, dtype=np.bool_)
    input_dict3 = {"input": tf.compat.v1.Variable(input3), "name": "bool_scalar"}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: 1D string tensor
    input4 = np.array(["hello", "world"], dtype=np.string_)
    input_dict4 = {"input": tf.compat.v1.Variable(input4), "name": "string_1d"}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: 2D float64 tensor with negative values
    input5 = np.array([[-1.0, 2.0], [-3.0, 4.0]], dtype=np.float64)
    input_dict5 = {"input": tf.compat.v1.Variable(input5), "name": "float64_2d_negative"}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.DebugGradientRefIdentity"] = tf_raw_ops_debug_gradient_ref_identity_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.DebugGradientRefIdentity' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DebugGradientRefIdentity'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.DebugGradientRefIdentity', generated_inputs['tf.raw_ops.DebugGradientRefIdentity'], lib="tf", suffix=0)

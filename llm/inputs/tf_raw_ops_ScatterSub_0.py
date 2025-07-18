
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_scatter_sub_inputs():
    """
    This function generates a list of valid inputs for the tf.raw_ops.ScatterSub function.
    The provided inputs are syntactically correct according to the API documentation.
    However, this specific raw op is designed for TensorFlow's graph mode and expects
    a mutable tf.Variable as the 'ref' argument. In an eager execution environment,
    passing NumPy arrays (which become immutable tf.Tensors) will likely result in a
    runtime error, as the op lacks a corresponding eager kernel for Tensors.
    """
    list_of_inputs = []

    # Input 1: The most basic 1D float32 case.
    input_dict_1 = {
        'ref': np.array([10.0, 20.0, 30.0], dtype=np.float32),
        'indices': np.array([0, 2], dtype=np.int32),
        'updates': np.array([1.0, 3.0], dtype=np.float32),
        'use_locking': False,
        'name': "minimal_float32"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: The most basic 1D int32 case.
    input_dict_2 = {
        'ref': np.array([10, 20, 30], dtype=np.int32),
        'indices': np.array([1], dtype=np.int32),
        'updates': np.array([5], dtype=np.int32),
        'use_locking': False,
        'name': "minimal_int32"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: A basic 2D float case with locking enabled.
    input_dict_3 = {
        'ref': np.ones((3, 2), dtype=np.float32),
        'indices': np.array([0, 1], dtype=np.int32),
        'updates': np.full((2, 2), 0.5, dtype=np.float32),
        'use_locking': True,
        'name': "minimal_2d_float32_locking"
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    return list_of_inputs

generated_inputs["tf.raw_ops.ScatterSub"] = tf_raw_ops_scatter_sub_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ScatterSub' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ScatterSub'.")

check_valid('tf.raw_ops.ScatterSub', generated_inputs['tf.raw_ops.ScatterSub'], lib="tf", suffix=0)

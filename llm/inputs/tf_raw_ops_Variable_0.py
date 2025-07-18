
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_variable_inputs():
    """
    NOTE: The tf.raw_ops.Variable op is designed for TensorFlow 1.x's graph mode
    and is explicitly not supported in eager execution, which is the default in
    TensorFlow 2.x. Calling this function in an eager context will always raise
    a RuntimeError, regardless of the inputs. The provided inputs are valid
    for the operation's signature if it were to be run in a TF1-style graph.
    """
    list_of_inputs = []

    # Input 1: Basic scalar float32
    input_dict_1 = {
        'shape': [],
        'dtype': np.float32,
        'container': '',
        'shared_name': '',
        'name': 'v1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 1D vector of int32
    input_dict_2 = {
        'shape': [10],
        'dtype': np.int32,
        'container': '',
        'shared_name': '',
        'name': 'v2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 2D matrix of float32
    input_dict_3 = {
        'shape': [4, 4],
        'dtype': np.float32,
        'container': '',
        'shared_name': '',
        'name': 'v3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: 3D tensor of int32
    input_dict_4 = {
        'shape': [2, 3, 5],
        'dtype': np.int32,
        'container': '',
        'shared_name': '',
        'name': 'v4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Scalar with float64 dtype
    input_dict_5 = {
        'shape': [],
        'dtype': np.float64,
        'container': '',
        'shared_name': '',
        'name': 'v5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Matrix with a container name
    input_dict_6 = {
        'shape': [8, 2],
        'dtype': np.float32,
        'container': 'my_container',
        'shared_name': '',
        'name': 'v6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Vector with a shared name
    input_dict_7 = {
        'shape': [128],
        'dtype': np.int32,
        'container': '',
        'shared_name': 'my_shared_var',
        'name': 'v7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Matrix with both container and shared name
    input_dict_8 = {
        'shape': [6, 6],
        'dtype': np.float32,
        'container': 'another_container',
        'shared_name': 'another_shared_var',
        'name': 'v8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))
    
    # Input 9: High-rank tensor
    input_dict_9 = {
        'shape': [1, 2, 3, 4],
        'dtype': np.float32,
        'container': '',
        'shared_name': '',
        'name': 'v9'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Vector with int64 dtype
    input_dict_10 = {
        'shape': [20],
        'dtype': np.int64,
        'container': '',
        'shared_name': '',
        'name': 'v10'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.Variable"] = tf_raw_ops_variable_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Variable' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Variable'.")

check_valid('tf.raw_ops.Variable', generated_inputs['tf.raw_ops.Variable'], lib="tf", suffix=0)

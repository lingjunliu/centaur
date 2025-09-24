
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_variable_inputs():
    """
    The API `tf.raw_ops.Variable` is incompatible with eager execution,
    which is the default in modern TensorFlow. This will always raise a
    RuntimeError when called in an eager context. The inputs below are

    syntactically correct but cannot prevent this fundamental error.
    """
    list_of_inputs = []

    # Input 1: Minimal float32
    input_dict = {
        'shape': [1],
        'dtype': np.float32,
        'container': '',
        'shared_name': '',
        'name': 'v1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Minimal int32
    input_dict = {
        'shape': [2, 2],
        'dtype': np.int32,
        'container': '',
        'shared_name': '',
        'name': 'v2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Minimal float64
    input_dict = {
        'shape': [3],
        'dtype': np.float64,
        'container': 'a',
        'shared_name': 'b',
        'name': 'v3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Minimal int64
    input_dict = {
        'shape': [4],
        'dtype': np.int64,
        'container': 'a',
        'shared_name': 'c',
        'name': 'v4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Scalar
    input_dict = {
        'shape': [],
        'dtype': np.float32,
        'container': '',
        'shared_name': 's',
        'name': 'v5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6: uint8
    input_dict = {
        'shape': [10],
        'dtype': np.uint8,
        'container': 'u',
        'shared_name': 'u_s',
        'name': 'v6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: bool
    input_dict = {
        'shape': [5],
        'dtype': np.bool_,
        'container': '',
        'shared_name': '',
        'name': 'v7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

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

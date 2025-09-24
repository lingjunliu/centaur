
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_conditionalaccumulator_inputs():
    list_of_inputs = []

    # The 'ConditionalAccumulator' op is not compatible with eager execution.
    # The following inputs are generated based on the API's signature. They are
    # syntactically valid but are expected to fail at runtime in an eager context.

    # Input 1
    input_dict = {
        'dtype': np.dtype('float32'),
        'shape': [1],
        'container': '',
        'shared_name': '',
        'reduction_type': 'MEAN',
        'name': 'accumulator_v1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        'dtype': np.dtype('int32'),
        'shape': [32, 32],
        'container': '',
        'shared_name': '',
        'reduction_type': 'SUM',
        'name': 'accumulator_v2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        'dtype': np.dtype('float64'),
        'shape': [],
        'container': 'container_a',
        'shared_name': '',
        'reduction_type': 'MEAN',
        'name': 'accumulator_v3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        'dtype': np.dtype('half'),
        'shape': [2, 8],
        'container': '',
        'shared_name': 'shared_name_b',
        'reduction_type': 'SUM',
        'name': 'accumulator_v4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        'dtype': np.dtype('int64'),
        'shape': [10, 20, 30],
        'container': 'container_c',
        'shared_name': 'shared_name_c',
        'reduction_type': 'MEAN',
        'name': 'accumulator_v5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        'dtype': np.dtype('complex64'),
        'shape': [5],
        'container': '',
        'shared_name': '',
        'reduction_type': 'MEAN',
        'name': 'accumulator_v6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        'dtype': np.dtype('uint8'),
        'shape': [256],
        'container': '',
        'shared_name': '',
        'reduction_type': 'SUM',
        'name': 'accumulator_v7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        'dtype': np.dtype('int8'),
        'shape': [4, 4],
        'container': '',
        'shared_name': 'shared_name_d',
        'reduction_type': 'MEAN',
        'name': 'accumulator_v8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        'dtype': np.dtype('complex128'),
        'shape': [2, 2, 2],
        'container': 'container_e',
        'shared_name': '',
        'reduction_type': 'SUM',
        'name': 'accumulator_v9'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        'dtype': np.dtype('uint16'),
        'shape': [100],
        'container': 'container_f',
        'shared_name': 'shared_name_f',
        'reduction_type': 'MEAN',
        'name': 'accumulator_v10'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11
    input_dict = {
        'dtype': np.dtype('uint32'),
        'shape': [64],
        'container': '',
        'shared_name': '',
        'reduction_type': 'SUM',
        'name': 'accumulator_v11'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12
    input_dict = {
        'dtype': np.dtype('uint64'),
        'shape': [8, 8, 8],
        'container': '',
        'shared_name': 'shared_name_g',
        'reduction_type': 'MEAN',
        'name': 'accumulator_v12'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.ConditionalAccumulator"] = tf_raw_ops_conditionalaccumulator_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ConditionalAccumulator' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ConditionalAccumulator'.")

check_valid('tf.raw_ops.ConditionalAccumulator', generated_inputs['tf.raw_ops.ConditionalAccumulator'], lib="tf", suffix=0)

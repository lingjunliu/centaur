
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_depth_to_space_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.DepthToSpace function.
    """
    list_of_inputs = []

    # Input 1: Basic NHWC case from documentation
    input_dict_1 = {
        'input': np.arange(1, 5, dtype=np.int32).reshape([1, 1, 1, 4]),
        'block_size': 2,
        'data_format': 'NHWC',
        'name': 'test_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Basic case with float32 and NHWC
    input_dict_2 = {
        'input': np.arange(1, 5, dtype=np.float32).reshape([1, 1, 1, 4]),
        'block_size': 2,
        'data_format': 'NHWC',
        'name': 'test_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Larger block_size (3) with NHWC
    input_dict_3 = {
        'input': np.arange(1, 10, dtype=np.int64).reshape([1, 1, 1, 9]),
        'block_size': 3,
        'data_format': 'NHWC',
        'name': 'test_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Larger block_size (3) with NHWC, float64 and multiple output channels
    input_dict_4 = {
        'input': np.arange(1, 73, dtype=np.float64).reshape([1, 2, 2, 18]),
        'block_size': 3,
        'data_format': 'NHWC',
        'name': 'test_4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: More complex spatial dimensions with NHWC
    input_dict_5 = {
        'input': np.arange(1, 17, dtype=np.int32).reshape([1, 2, 2, 4]),
        'block_size': 2,
        'data_format': 'NHWC',
        'name': 'test_5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: More complex spatial dimensions with NHWC and float32
    input_dict_6 = {
        'input': np.arange(1, 17, dtype=np.float32).reshape([1, 2, 2, 4]),
        'block_size': 2,
        'data_format': 'NHWC',
        'name': 'test_6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Larger batch size and multiple output channels (NHWC)
    input_dict_7 = {
        'input': np.arange(1, 17, dtype=np.int16).reshape([2, 1, 2, 4]),
        'block_size': 2,
        'data_format': 'NHWC',
        'name': 'test_7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Larger batch size and multiple output channels (NHWC) with uint8
    input_dict_8 = {
        'input': np.arange(1, 33, dtype=np.uint8).reshape([2, 2, 1, 8]),
        'block_size': 2,
        'data_format': 'NHWC',
        'name': 'test_8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: block_size = 4, NHWC
    input_dict_9 = {
        'input': np.arange(1, 33, dtype=np.float32).reshape([1, 1, 2, 16]),
        'block_size': 4,
        'data_format': 'NHWC',
        'name': 'test_9'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: block_size = 4, NHWC with int32
    input_dict_10 = {
        'input': np.arange(1, 33, dtype=np.int32).reshape([1, 1, 2, 16]),
        'block_size': 4,
        'data_format': 'NHWC',
        'name': 'test_10'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: Another example from documentation
    input_dict_11 = {
        'input': np.arange(1, 13, dtype=np.float32).reshape([1, 1, 1, 12]),
        'block_size': 2,
        'data_format': 'NHWC',
        'name': 'test_11'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    return list_of_inputs

generated_inputs["tf.raw_ops.DepthToSpace"] = tf_raw_ops_depth_to_space_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.DepthToSpace' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DepthToSpace'.")

check_valid('tf.raw_ops.DepthToSpace', generated_inputs['tf.raw_ops.DepthToSpace'], lib="tf", suffix=0)

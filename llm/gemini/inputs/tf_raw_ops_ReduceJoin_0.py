
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_reducejoin_inputs():
    list_of_inputs = []

    # Case 1: Basic 2D reduction along axis 0
    input_dict1 = {
        'inputs': np.array([["a", "b"], ["c", "d"]], dtype=np.object_),
        'reduction_indices': np.array([0], dtype=np.int32),
        'keep_dims': False,
        'separator': '',
        'name': 'case1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Case 2: Basic 2D reduction along axis 1
    input_dict2 = {
        'inputs': np.array([["a", "b"], ["c", "d"]], dtype=np.object_),
        'reduction_indices': np.array([1], dtype=np.int32),
        'keep_dims': False,
        'separator': '',
        'name': 'case2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Case 3: 2D reduction with a separator
    input_dict3 = {
        'inputs': np.array([["a", "b"], ["c", "d"]], dtype=np.object_),
        'reduction_indices': np.array([0], dtype=np.int32),
        'keep_dims': False,
        'separator': '.',
        'name': 'case3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Case 4: 2D reduction with keep_dims=True
    input_dict4 = {
        'inputs': np.array([["a", "b"], ["c", "d"]], dtype=np.object_),
        'reduction_indices': np.array([1], dtype=np.int32),
        'keep_dims': True,
        'separator': '',
        'name': 'case4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Case 5: 2D reduction with a negative index
    input_dict5 = {
        'inputs': np.array([["a", "b"], ["c", "d"]], dtype=np.object_),
        'reduction_indices': np.array([-1], dtype=np.int32),
        'keep_dims': False,
        'separator': '',
        'name': 'case5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Case 6: 2D reduction across multiple dimensions (order 1)
    input_dict6 = {
        'inputs': np.array([["a", "b"], ["c", "d"]], dtype=np.object_),
        'reduction_indices': np.array([1, 0], dtype=np.int32),
        'keep_dims': False,
        'separator': '',
        'name': 'case6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Case 7: 2D reduction across multiple dimensions (order 2)
    input_dict7 = {
        'inputs': np.array([["a", "b"], ["c", "d"]], dtype=np.object_),
        'reduction_indices': np.array([0, 1], dtype=np.int32),
        'keep_dims': False,
        'separator': '',
        'name': 'case7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Case 8: 3D tensor reduction
    input_dict8 = {
        'inputs': np.array([[['a'], ['b']], [['c'], ['d']]], dtype=np.object_),
        'reduction_indices': np.array([0], dtype=np.int32),
        'keep_dims': False,
        'separator': '_',
        'name': 'case8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))
    
    # Case 9: 3D tensor reduction on multiple axes
    input_dict9 = {
        'inputs': np.array([[['a', 'b'], ['c', 'd']], [['e', 'f'], ['g', 'h']]], dtype=np.object_),
        'reduction_indices': np.array([0, 2], dtype=np.int32),
        'keep_dims': False,
        'separator': '',
        'name': 'case9'
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Case 10: Reduction with empty reduction_indices (no-op)
    input_dict10 = {
        'inputs': np.array([["a", "b"], ["c", "d"]], dtype=np.object_),
        'reduction_indices': np.array([], dtype=np.int32),
        'keep_dims': False,
        'separator': '',
        'name': 'case10'
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    # Case 11: 1D tensor reduction
    input_dict11 = {
        'inputs': np.array(["hello", "world", "from", "tensorflow"], dtype=np.object_),
        'reduction_indices': np.array([0], dtype=np.int32),
        'keep_dims': False,
        'separator': ' ',
        'name': 'case11'
    }
    list_of_inputs.append(copy.deepcopy(input_dict11))

    # Case 12: 3D tensor reduction with keep_dims=True
    input_dict12 = {
        'inputs': np.array([[['a', 'b'], ['c', 'd']], [['e', 'f'], ['g', 'h']]], dtype=np.object_),
        'reduction_indices': np.array([1, 2], dtype=np.int32),
        'keep_dims': True,
        'separator': '-',
        'name': 'case12'
    }
    list_of_inputs.append(copy.deepcopy(input_dict12))

    return list_of_inputs

generated_inputs["tf.raw_ops.ReduceJoin"] = tf_raw_ops_reducejoin_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.ReduceJoin' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.ReduceJoin'.")

check_valid('tf.raw_ops.ReduceJoin', generated_inputs['tf.raw_ops.ReduceJoin'], lib="tf", suffix=0)

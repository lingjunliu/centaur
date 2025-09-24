
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_tf_ragged_constant_inputs():
    """
    Generates a list of valid inputs for tf.ragged.constant.
    The inputs are constructed based on the rule that the nesting depth of scalars in pylist (K)
    must equal `ragged_rank + 1 + len(inner_shape)`.
    All `pylist` inputs are rectangular to avoid errors in downstream processing that uses np.min/np.max.
    All inputs use `ragged_rank >= 1` to ensure a RaggedTensor is created.
    """
    list_of_inputs = []

    # K=2. Rule: 2 = rr + 1 + len(is). If rr=1, len(is) must be 0.
    # Input 1: Basic case with integers.
    input_dict_1 = {
        'pylist': [[1, 2, 3], [4, 5, 6]],
        'dtype': np.int32,
        'ragged_rank': 1,
        'inner_shape': (),
        'name': 'K2_rr1_is0_int',
        'row_splits_dtype': np.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Basic case with floats.
    input_dict_2 = {
        'pylist': [[-1.0, -2.0], [-3.0, -4.0]],
        'dtype': np.float32,
        'ragged_rank': 1,
        'inner_shape': (),
        'name': 'K2_rr1_is0_float',
        'row_splits_dtype': np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Basic case with booleans.
    input_dict_3 = {
        'pylist': [[True, False], [False, True]],
        'dtype': np.bool_,
        'ragged_rank': 1,
        'inner_shape': (),
        'name': 'K2_rr1_is0_bool',
        'row_splits_dtype': np.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # K=3. Rule: 3 = rr + 1 + len(is).
    # Input 4: rr=1, len(is)=1.
    input_dict_4 = {
        'pylist': [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
        'dtype': np.int64,
        'ragged_rank': 1,
        'inner_shape': (2,),
        'name': 'K3_rr1_is1',
        'row_splits_dtype': np.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: rr=2, len(is)=0.
    input_dict_5 = {
        'pylist': [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
        'dtype': np.int16,
        'ragged_rank': 2,
        'inner_shape': (),
        'name': 'K3_rr2_is0',
        'row_splits_dtype': np.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # K=4. Rule: 4 = rr + 1 + len(is).
    # Input 6: rr=1, len(is)=2.
    input_dict_6 = {
        'pylist': [[[[1, 2], [3, 4]]], [[[5, 6], [7, 8]]]],
        'dtype': np.float64,
        'ragged_rank': 1,
        'inner_shape': (2, 2),
        'name': 'K4_rr1_is2',
        'row_splits_dtype': np.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))
    
    # Input 7: rr=2, len(is)=1.
    input_dict_7 = {
        'pylist': [[[[1, 2]], [[3, 4]]], [[[5, 6]], [[7, 8]]]],
        'dtype': np.uint8,
        'ragged_rank': 2,
        'inner_shape': (2,),
        'name': 'K4_rr2_is1',
        'row_splits_dtype': np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: rr=3, len(is)=0.
    input_dict_8 = {
        'pylist': [[[[1, 2]], [[3, 4]]], [[[5, 6]], [[7, 8]]]],
        'dtype': np.int8,
        'ragged_rank': 3,
        'inner_shape': (),
        'name': 'K4_rr3_is0',
        'row_splits_dtype': np.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))
    
    # Input 9: Single row pylist. K=2. rr=1, len(is)=0.
    input_dict_9 = {
        'pylist': [[100, 200, 300, 400]],
        'dtype': np.uint16,
        'ragged_rank': 1,
        'inner_shape': (),
        'name': 'K2_rr1_is0_singlerow',
        'row_splits_dtype': np.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))
    
    # Input 10: Single column pylist. K=2. rr=1, len(is)=0.
    input_dict_10 = {
        'pylist': [[10], [20], [30]],
        'dtype': np.uint32,
        'ragged_rank': 1,
        'inner_shape': (),
        'name': 'K2_rr1_is0_singlecol',
        'row_splits_dtype': np.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.ragged.constant"] = get_tf_ragged_constant_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.ragged.constant' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.ragged.constant'.")

check_valid('tf.ragged.constant', generated_inputs['tf.ragged.constant'], lib="tf", suffix=0)

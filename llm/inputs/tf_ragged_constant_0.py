
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_ragged_constant_inputs():
    """
    Generates a list of valid inputs for the tf.ragged.constant function.
    This version ensures all generated dictionaries contain all keys specified in the
    signature to avoid KeyErrors, uses rectangular lists for 'pylist' to
    avoid ValueErrors in the testing harness, and corrects inputs that caused
    TypeErrors/ValueErrors during execution by ensuring parameter combinations are valid.
    """
    list_of_inputs = []

    # Input 1: Basic 2D list made into a RaggedTensor with ragged_rank=1
    input_dict_1 = {
        'pylist': [[1, 2], [3, 4]],
        'dtype': np.int32,
        'ragged_rank': 1,
        'inner_shape': (),
        'name': 'input_1',
        'row_splits_dtype': np.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Specifying float dtype with negative values
    input_dict_2 = {
        'pylist': [[-1.5, 0.5], [10.0, -9.9]],
        'dtype': np.float32,
        'ragged_rank': 1,
        'inner_shape': (),
        'name': 'input_2_float',
        'row_splits_dtype': np.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 3D list, rank 2 ragged tensor
    input_dict_3 = {
        'pylist': [[[0, 1], [2, 3]], [[4, 5], [6, 7]]],
        'dtype': np.int64,
        'ragged_rank': 2,
        'inner_shape': (),
        'name': 'input_3_rank2',
        'row_splits_dtype': np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: 3D list with specified inner_shape, making it a rank 1 ragged tensor
    input_dict_4 = {
        'pylist': [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
        'dtype': np.int32,
        'ragged_rank': 1,
        'inner_shape': (2,),
        'name': 'input_4_innershape',
        'row_splits_dtype': np.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: 2D list to be converted to a ragged tensor (ragged_rank=1)
    input_dict_5 = {
        'pylist': [[1, 2, 3], [4, 5, 6]],
        'dtype': np.uint8,
        'ragged_rank': 1,
        'inner_shape': (),
        'name': 'input_5_ragged_2d',
        'row_splits_dtype': np.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: 4D list with inner_shape
    input_dict_6 = {
        'pylist': np.arange(16).reshape(2, 1, 2, 4).tolist(),
        'dtype': np.int32,
        'ragged_rank': 2,
        'inner_shape': (4,),
        'name': 'input_6_4d',
        'row_splits_dtype': np.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Creating a ragged tensor from a 2x2 list by setting ragged_rank=1
    input_dict_7 = {
        'pylist': [[1.0, 2.0], [3.0, 4.0]],
        'dtype': np.float64,
        'ragged_rank': 1,
        'inner_shape': (),
        'name': 'input_7_ragged_rank1',
        'row_splits_dtype': np.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Empty pylist, creating a dense tensor of shape (0,)
    input_dict_8 = {
        'pylist': [],
        'dtype': np.float32,
        'ragged_rank': 0,
        'inner_shape': (0,),
        'name': 'input_8_empty',
        'row_splits_dtype': np.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Ragged tensor of strings from a rectangular list
    input_dict_9 = {
        'pylist': [[b'alpha', b'beta'], [b'gamma', b'delta']],
        'dtype': np.string_,
        'ragged_rank': 1,
        'inner_shape': (),
        'name': 'input_9_string',
        'row_splits_dtype': np.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: A higher-rank dense tensor made ragged
    input_dict_10 = {
        'pylist': np.arange(24).reshape(2, 3, 4).tolist(),
        'dtype': np.int32,
        'ragged_rank': 1,
        'inner_shape': (4,),
        'name': 'input_10_high_rank',
        'row_splits_dtype': np.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.ragged.constant"] = tf_ragged_constant_inputs()

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


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_ragged_cross_hashed_inputs():
    """
    Generates a list of valid inputs for the tf.ragged.cross_hashed function.
    """
    list_of_inputs = []

    # Input 1: Basic case from the documentation
    tensors1 = [
        np.array([['a'], ['b', 'c']], dtype=object),
        np.array([['d'], ['e']]),
        np.array([['f'], ['g']])
    ]
    inputs1 = np.empty(len(tensors1), dtype=object)
    inputs1[:] = tensors1
    input_dict_1 = {
        'inputs': inputs1,
        'num_buckets': 100,
        'hash_key': 1337,
        'name': 'basic_case'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: num_buckets = 0 (no bucketing)
    tensors2 = [
        np.array([['x'], ['y', 'z']], dtype=object),
        np.array([['1'], ['2', '3']], dtype=object)
    ]
    inputs2 = np.empty(len(tensors2), dtype=object)
    inputs2[:] = tensors2
    input_dict_2 = {
        'inputs': inputs2,
        'num_buckets': 0,
        'hash_key': 45678,
        'name': 'no_bucketing'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Mix of ragged and dense tensors
    tensors3 = [
        np.array([['a', 'b'], ['c']], dtype=object),
        np.array([['d'], ['e']])
    ]
    inputs3 = np.empty(len(tensors3), dtype=object)
    inputs3[:] = tensors3
    input_dict_3 = {
        'inputs': inputs3,
        'num_buckets': 50,
        'hash_key': 9876,
        'name': 'mixed_tensor_types'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Integer inputs
    tensors4 = [
        np.array([[1], [2, 3]], dtype=object),
        np.array([[10], [20]])
    ]
    inputs4 = np.empty(len(tensors4), dtype=object)
    inputs4[:] = tensors4
    input_dict_4 = {
        'inputs': inputs4,
        'num_buckets': 20,
        'hash_key': 112233,
        'name': 'integer_inputs'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: More than two input tensors, all dense
    tensors5 = [
        np.array([['a'], ['b']]),
        np.array([['c'], ['d']]),
        np.array([['e'], ['f']]),
        np.array([['g'], ['h']])
    ]
    inputs5 = np.empty(len(tensors5), dtype=object)
    inputs5[:] = tensors5
    input_dict_5 = {
        'inputs': inputs5,
        'num_buckets': 1000,
        'hash_key': 7890,
        'name': 'multiple_inputs_dense'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Empty rows in one of the ragged tensors
    tensors6 = [
        np.array([['a1', 'a2'], [], ['a3']], dtype=object),
        np.array([['b1'], ['b2'], ['b3']])
    ]
    inputs6 = np.empty(len(tensors6), dtype=object)
    inputs6[:] = tensors6
    input_dict_6 = {
        'inputs': inputs6,
        'num_buckets': 10,
        'hash_key': 111,
        'name': 'empty_rows'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: All inputs are dense Tensors (numpy arrays)
    tensors7 = [
        np.array([['a', 'b'], ['c', 'd']]),
        np.array([['e', 'f'], ['g', 'h']])
    ]
    inputs7 = np.empty(len(tensors7), dtype=object)
    inputs7[:] = tensors7
    input_dict_7 = {
        'inputs': inputs7,
        'num_buckets': 5,
        'hash_key': 222,
        'name': 'all_dense_tensors'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Single input tensor in the list
    tensors8 = [
        np.array([['single1', 'single2'], ['single3']], dtype=object)
    ]
    inputs8 = np.empty(len(tensors8), dtype=object)
    inputs8[:] = tensors8
    input_dict_8 = {
        'inputs': inputs8,
        'num_buckets': 100,
        'hash_key': 333,
        'name': 'single_input'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: num_buckets = 1
    tensors9 = [
        np.array([['a', 'b', 'c'], ['d']], dtype=object),
        np.array([['e'], ['f', 'g']], dtype=object)
    ]
    inputs9 = np.empty(len(tensors9), dtype=object)
    inputs9[:] = tensors9
    input_dict_9 = {
        'inputs': inputs9,
        'num_buckets': 1,
        'hash_key': 666,
        'name': 'one_bucket'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Large hash_key, dense tensors
    tensors10 = [
        np.array([['hello'], ['world']]),
        np.array([['foo'], ['bar']])
    ]
    inputs10 = np.empty(len(tensors10), dtype=object)
    inputs10[:] = tensors10
    input_dict_10 = {
        'inputs': inputs10,
        'num_buckets': 1000,
        'hash_key': 9223372036854775807,
        'name': 'large_hash_key'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: One tensor has an empty row, resulting in an empty cross for that row
    tensors11 = [
        np.array([['a', 'b'], ['c']], dtype=object),
        np.array([[], ['d']], dtype=object)
    ]
    inputs11 = np.empty(len(tensors11), dtype=object)
    inputs11[:] = tensors11
    input_dict_11 = {
        'inputs': inputs11,
        'num_buckets': 15,
        'hash_key': 444,
        'name': 'one_input_empty_row'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    return list_of_inputs

generated_inputs["tf.ragged.cross_hashed"] = tf_ragged_cross_hashed_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.ragged.cross_hashed' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.ragged.cross_hashed'.")

check_valid('tf.ragged.cross_hashed', generated_inputs['tf.ragged.cross_hashed'], lib="tf", suffix=0)

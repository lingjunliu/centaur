
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_sparse_cross_hashed_inputs():
    list_of_inputs = []

    # Helper function to find a common shape and pad arrays
    def to_stacked_array(list_of_arrs):
        if not list_of_arrs:
            # Create a 4D tensor with shape [0, 0, 0, 0] for an empty list
            return np.empty((0, 0, 0, 0), dtype=object)

        # Find the maximum rank
        max_rank = 0
        for arr in list_of_arrs:
            max_rank = max(max_rank, arr.ndim)

        # Pad ranks to be the same
        padded_arrs = []
        for arr in list_of_arrs:
            while arr.ndim < max_rank:
                arr = np.expand_dims(arr, axis=-1)
            padded_arrs.append(arr)
        
        # Find the maximum shape in each dimension
        max_shape = [0] * max_rank
        for arr in padded_arrs:
            for i, dim in enumerate(arr.shape):
                max_shape[i] = max(max_shape[i], dim)

        # Pad each array to the max shape
        final_arrs = []
        for arr in padded_arrs:
            paddings = []
            for i in range(max_rank):
                paddings.append((0, max_shape[i] - arr.shape[i]))
            final_arrs.append(np.pad(arr, pad_width=paddings, mode='constant', constant_values=''))
        
        return np.stack(final_arrs)

    # Input 1: Based on documentation example
    input_dict_1 = {
        'inputs': to_stacked_array([
            np.array([["a", ""], ["b", "c"]], dtype=object),
            np.array([["d"], ["e"]], dtype=object),
            np.array([["f"], ["g"]], dtype=object)
        ]),
        'num_buckets': 0,
        'hash_key': 1337,
        'name': 'doc_example'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Two dense tensors, with bucketing
    input_dict_2 = {
        'inputs': to_stacked_array([
            np.array([["feat1_val1", ""], ["", "feat1_val2"]], dtype=object),
            np.array([["", "feat2_val1"], ["feat2_val2", ""]], dtype=object)
        ]),
        'num_buckets': 1000,
        'hash_key': 123456789,
        'name': 'two_dense_with_bucketing'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: All dense inputs with same shape
    input_dict_3 = {
        'inputs': to_stacked_array([
            np.array([["A", "B"], ["C", "D"]], dtype=object),
            np.array([["X", "Y"], ["Z", "W"]], dtype=object)
        ]),
        'num_buckets': 10,
        'hash_key': 987654321,
        'name': 'all_dense_same_shape'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))
    
    # Input 4: A mix of string and integer-like string tensors
    input_dict_4 = {
        'inputs': to_stacked_array([
            np.array([["123", ""], ["", "456"]], dtype=object),
            np.array([["X"], ["Y"]], dtype=object)
        ]),
        'num_buckets': 0,
        'hash_key': 2023,
        'name': 'mixed_types_as_string'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: 3D Tensors
    input_dict_5 = {
        'inputs': to_stacked_array([
            np.array([[["v1", "v2"], ["v3", "v4"]]], dtype=object),
            np.array([[["d1", "d2"], ["d3", "d4"]]], dtype=object)
        ]),
        'num_buckets': 50,
        'hash_key': 112358,
        'name': '3d_tensors'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Single tensor input in a list
    input_dict_6 = {
        'inputs': to_stacked_array([
            np.array([["", "10", ""], ["", "", ""], ["20", "", "30"]], dtype=object),
        ]),
        'num_buckets': 0,
        'hash_key': 101010,
        'name': 'single_tensor_input'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))
    
    # Input 7: Empty tensors and non-empty tensors
    input_dict_7 = {
        'inputs': to_stacked_array([
            np.full((3, 3), "", dtype=object),
            np.array([["a"], ["b"], ["c"]], dtype=object)
        ]),
        'num_buckets': 100,
        'hash_key': 303030,
        'name': 'with_empty_tensor'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))
    
    # Input 8: Negative hash key
    input_dict_8 = {
        'inputs': to_stacked_array([
            np.array([["value1", ""], ["", ""]], dtype=object),
            np.array([["dense1"], ["dense2"]], dtype=object)
        ]),
        'num_buckets': 1000,
        'hash_key': -12345,
        'name': 'negative_hash_key'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))
    
    # Input 9: Large num_buckets
    input_dict_9 = {
        'inputs': to_stacked_array([
            np.array(["alpha", "beta", "gamma"], dtype=object)
        ]),
        'num_buckets': 2**30,
        'hash_key': 404040,
        'name': 'large_num_buckets'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))
    
    # Input 10: Unicode strings
    input_dict_10 = {
        'inputs': to_stacked_array([
            np.array([["你好"], ["世界"]], dtype=object),
            np.array([["こんにちは"], ["こんばんは"]], dtype=object)
        ]),
        'num_buckets': 1024,
        'hash_key': 505050,
        'name': 'unicode_strings'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: More than 2 inputs
    input_dict_11 = {
        'inputs': to_stacked_array([
            np.array([["a"], ["b"]]),
            np.array([["c"], ["d"]]),
            np.array([["e"], ["f"]]),
            np.array([["g"], ["h"]]),
        ]),
        'num_buckets': 0,
        'hash_key': 808080,
        'name': 'multiple_inputs'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    return list_of_inputs

generated_inputs["tf.sparse.cross_hashed"] = tf_sparse_cross_hashed_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.sparse.cross_hashed' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sparse.cross_hashed'.")

check_valid('tf.sparse.cross_hashed', generated_inputs['tf.sparse.cross_hashed'], lib="tf", suffix=0)

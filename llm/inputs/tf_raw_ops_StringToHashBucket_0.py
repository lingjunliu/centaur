
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_string_to_hash_bucket_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.StringToHashBucket function.
    """
    list_of_inputs = []

    # Input 1: Basic 1D tensor of strings
    input_dict = {
        'string_tensor': np.array(["hello", "world", "tensorflow", "is", "great"], dtype=object),
        'num_buckets': 10,
        'name': 'test_1d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D tensor of strings
    input_dict = {
        'string_tensor': np.array([["a", "b", "c"], ["d", "e", "f"]], dtype=object),
        'num_buckets': 5,
        'name': 'test_2d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Scalar string tensor
    input_dict = {
        'string_tensor': np.array("a_single_string", dtype=object),
        'num_buckets': 100,
        'name': 'test_scalar'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Minimum number of buckets (1)
    input_dict = {
        'string_tensor': np.array(["any", "string", "will", "do"], dtype=object),
        'num_buckets': 1,
        'name': 'test_min_buckets'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Tensor with empty strings
    input_dict = {
        'string_tensor': np.array(["", "not_empty", "", "also_not_empty"], dtype=object),
        'num_buckets': 7,
        'name': 'test_empty_strings'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Tensor with numeric and special characters
    input_dict = {
        'string_tensor': np.array(["123", "!@#$", "mix_123_!"], dtype=object),
        'num_buckets': 12,
        'name': 'test_special_chars'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Tensor with Unicode characters
    input_dict = {
        'string_tensor': np.array(["你好", "世界", "안녕하세요", "你好世界"], dtype=object),
        'num_buckets': 20,
        'name': 'test_unicode'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Large number of buckets
    input_dict = {
        'string_tensor': np.array(["a", "b", "c", "d", "e"], dtype=object),
        'num_buckets': 1000000,
        'name': 'test_large_buckets'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Empty input tensor
    input_dict = {
        'string_tensor': np.array([], dtype=object),
        'num_buckets': 10,
        'name': 'test_empty_tensor'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D tensor of strings
    input_dict = {
        'string_tensor': np.array([
            [["a", "b"], ["c", "d"]],
            [["e", "f"], ["g", "h"]]
        ], dtype=object),
        'num_buckets': 3,
        'name': 'test_3d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Tensor with repeated strings
    input_dict = {
        'string_tensor': np.array(["repeat", "unique", "repeat", "another_unique", "repeat"], dtype=object),
        'num_buckets': 15,
        'name': 'test_repeated_strings'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.StringToHashBucket"] = get_string_to_hash_bucket_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.StringToHashBucket' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.StringToHashBucket'.")

check_valid('tf.raw_ops.StringToHashBucket', generated_inputs['tf.raw_ops.StringToHashBucket'], lib="tf", suffix=0)

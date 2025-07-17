
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_string_to_hash_bucket_inputs():
    list_of_inputs = []

    # Input 1, valid
    string_tensor = np.array(["hello", "world", "tensorflow"], dtype=np.str_)
    num_buckets = 10
    input_dict = {
        "string_tensor": string_tensor,
        "num_buckets": num_buckets,
        "name": "string"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, valid
    string_tensor = np.array([["hello", "world"], ["tensorflow", "rocks"]], dtype=np.str_)
    num_buckets = 100
    input_dict = {
        "string_tensor": string_tensor,
        "num_buckets": num_buckets,
        "name": "string"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid
    string_tensor = np.array([""], dtype=np.str_)
    num_buckets = 5
    input_dict = {
        "string_tensor": string_tensor,
        "num_buckets": num_buckets,
        "name": "string"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, valid
    string_tensor = np.array(["123", "456", "789"], dtype=np.str_)
    num_buckets = 2
    input_dict = {
        "string_tensor": string_tensor,
        "num_buckets": num_buckets,
        "name": "string"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, valid
    string_tensor = np.array(["a", "b", "c", "d", "e"], dtype=np.str_)
    num_buckets = 1000
    input_dict = {
        "string_tensor": string_tensor,
        "num_buckets": num_buckets,
        "name": "string"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6, valid, large array
    string_tensor = np.array(["test" for _ in range(100)], dtype=np.str_)
    num_buckets = 10
    input_dict = {
        "string_tensor": string_tensor,
        "num_buckets": num_buckets,
        "name": "string"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7, valid, different strings
    string_tensor = np.array(["string1", "string2", "string3", "string4"], dtype=np.str_)
    num_buckets = 4
    input_dict = {
        "string_tensor": string_tensor,
        "num_buckets": num_buckets,
        "name": "string"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8, valid
    string_tensor = np.array(["example"], dtype=np.str_)
    num_buckets = 2**31 - 1
    input_dict = {
        "string_tensor": string_tensor,
        "num_buckets": num_buckets,
        "name": "string"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid, bytes-like strings
    string_tensor = np.array([b"byte_string1", b"byte_string2"], dtype=np.object_)
    num_buckets = 10
    input_dict = {
        "string_tensor": string_tensor,
        "num_buckets": num_buckets,
        "name": "string"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10, valid, bytes-like strings, empty
    string_tensor = np.array([b"", b""], dtype=np.object_)
    num_buckets = 10
    input_dict = {
        "string_tensor": string_tensor,
        "num_buckets": num_buckets,
        "name": "string"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.StringToHashBucket"] = tf_raw_ops_string_to_hash_bucket_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.StringToHashBucket' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.StringToHashBucket'.")

check_valid('tf.raw_ops.StringToHashBucket', generated_inputs['tf.raw_ops.StringToHashBucket'], lib="tf", suffix=0)

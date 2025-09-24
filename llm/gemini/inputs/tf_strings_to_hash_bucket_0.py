
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_strings_to_hash_bucket_inputs():
    list_of_inputs = []

    # Input 1: Basic string tensor and bucket size
    input_tensor = np.array(["hello", "world", "tensorflow"], dtype=np.object_)
    num_buckets = 5
    name = None
    input_dict = {"input": input_tensor, "num_buckets": num_buckets, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different strings, larger bucket size
    input_tensor = np.array(["a", "b", "c", "d", "e"], dtype=np.object_)
    num_buckets = 10
    name = "hash_buckets"
    input_dict = {"input": input_tensor, "num_buckets": num_buckets, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Empty string, small bucket size
    input_tensor = np.array([""], dtype=np.object_)
    num_buckets = 2
    name = None
    input_dict = {"input": input_tensor, "num_buckets": num_buckets, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Repeated strings
    input_tensor = np.array(["same", "same", "same"], dtype=np.object_)
    num_buckets = 4
    name = None
    input_dict = {"input": input_tensor, "num_buckets": num_buckets, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: String with special characters
    input_tensor = np.array(["!@#$%^&*()"], dtype=np.object_)
    num_buckets = 3
    name = None
    input_dict = {"input": input_tensor, "num_buckets": num_buckets, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Number as strings
    input_tensor = np.array(["123", "456", "789"], dtype=np.object_)
    num_buckets = 6
    name = "numeric_buckets"
    input_dict = {"input": input_tensor, "num_buckets": num_buckets, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Mixed alphanumeric strings
    input_tensor = np.array(["abc123", "def456", "ghi789"], dtype=np.object_)
    num_buckets = 7
    name = None
    input_dict = {"input": input_tensor, "num_buckets": num_buckets, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Long strings
    input_tensor = np.array(["This is a very long string", "Another very long string"], dtype=np.object_)
    num_buckets = 5
    name = "long_buckets"
    input_dict = {"input": input_tensor, "num_buckets": num_buckets, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Multidimensional string array
    input_tensor = np.array([["a", "b"], ["c", "d"]], dtype=np.object_)
    num_buckets = 4
    name = None
    input_dict = {"input": input_tensor, "num_buckets": num_buckets, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: String with spaces
    input_tensor = np.array(["string with spaces", "another string with spaces"], dtype=np.object_)
    num_buckets = 6
    name = None
    input_dict = {"input": input_tensor, "num_buckets": num_buckets, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.strings.to_hash_bucket"] = tf_strings_to_hash_bucket_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.strings.to_hash_bucket' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.strings.to_hash_bucket'.")

check_valid('tf.strings.to_hash_bucket', generated_inputs['tf.strings.to_hash_bucket'], lib="tf", suffix=0)

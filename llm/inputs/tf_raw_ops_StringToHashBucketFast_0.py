
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_string_to_hash_bucket_fast_inputs():
    list_of_inputs = []

    # Input 1: Basic example
    input_tensor = np.array(["Hello", "TensorFlow", "2.x"], dtype=np.object_)
    num_buckets = 3
    input_dict = {"input": input_tensor, "num_buckets": num_buckets, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different number of buckets
    input_tensor = np.array(["Hello", "TensorFlow", "2.x", "Another"], dtype=np.object_)
    num_buckets = 10
    input_dict = {"input": input_tensor, "num_buckets": num_buckets, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Single string
    input_tensor = np.array(["SingleString"], dtype=np.object_)
    num_buckets = 5
    input_dict = {"input": input_tensor, "num_buckets": num_buckets, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Empty string
    input_tensor = np.array([""], dtype=np.object_)
    num_buckets = 2
    input_dict = {"input": input_tensor, "num_buckets": num_buckets, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Array of empty strings
    input_tensor = np.array(["", "", ""], dtype=np.object_)
    num_buckets = 4
    input_dict = {"input": input_tensor, "num_buckets": num_buckets, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Long strings
    input_tensor = np.array(["This is a very long string to test the hashing function."], dtype=np.object_)
    num_buckets = 7
    input_dict = {"input": input_tensor, "num_buckets": num_buckets, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Repeated strings
    input_tensor = np.array(["repeat", "repeat", "repeat"], dtype=np.object_)
    num_buckets = 5
    input_dict = {"input": input_tensor, "num_buckets": num_buckets, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Numbers as strings
    input_tensor = np.array(["123", "456", "789"], dtype=np.object_)
    num_buckets = 6
    input_dict = {"input": input_tensor, "num_buckets": num_buckets, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Special characters
    input_tensor = np.array(["!@#$", "%^&*"], dtype=np.object_)
    num_buckets = 4
    input_dict = {"input": input_tensor, "num_buckets": num_buckets, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Another example
    input_tensor = np.array(["abc", "def", "ghi"], dtype=np.object_)
    num_buckets = 8
    input_dict = {"input": input_tensor, "num_buckets": num_buckets, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.StringToHashBucketFast"] = tf_raw_ops_string_to_hash_bucket_fast_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.StringToHashBucketFast' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.StringToHashBucketFast'.")

check_valid('tf.raw_ops.StringToHashBucketFast', generated_inputs['tf.raw_ops.StringToHashBucketFast'], lib="tf", suffix=0)

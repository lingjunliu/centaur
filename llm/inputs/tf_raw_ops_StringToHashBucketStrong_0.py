
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_string_to_hash_bucket_strong_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array(["Hello", "TF"], dtype=np.object_)
    num_buckets = 3
    key = [1, 2]
    name = None

    input_dict = {
        "input": input_tensor,
        "num_buckets": num_buckets,
        "key": key,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array(["This", "is", "a", "test"], dtype=np.object_)
    num_buckets = 5
    key = [10, 20]
    name = "test_hash"

    input_dict = {
        "input": input_tensor,
        "num_buckets": num_buckets,
        "key": key,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array(["", "Empty", "String"], dtype=np.object_)
    num_buckets = 2
    key = [0, 0]
    name = None

    input_dict = {
        "input": input_tensor,
        "num_buckets": num_buckets,
        "key": key,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array(["Longer", "String", "To", "Be", "Hashed"], dtype=np.object_)
    num_buckets = 100
    key = [2**32, 2**32 + 1]
    name = "longer_test"

    input_dict = {
        "input": input_tensor,
        "num_buckets": num_buckets,
        "key": key,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array(["Same", "Input", "Repeated", "Same"], dtype=np.object_)
    num_buckets = 4
    key = [65535, 1]
    name = None

    input_dict = {
        "input": input_tensor,
        "num_buckets": num_buckets,
        "key": key,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6 (multidimensional)
    input_tensor = np.array([["Hello", "World"], ["TensorFlow", "Rocks"]], dtype=np.object_)
    num_buckets = 7
    key = [5, 6]
    name = "multi_dim"

    input_dict = {
        "input": input_tensor,
        "num_buckets": num_buckets,
        "key": key,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.array(["Unicode", "Strings", "你好世界"], dtype=np.object_)
    num_buckets = 6
    key = [7, 8]
    name = None

    input_dict = {
        "input": input_tensor,
        "num_buckets": num_buckets,
        "key": key,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.array(["With", "Special", "Characters", "!@#$%^&*()_+"], dtype=np.object_)
    num_buckets = 8
    key = [9, 10]
    name = "special_chars"

    input_dict = {
        "input": input_tensor,
        "num_buckets": num_buckets,
        "key": key,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.array(["Very", "Long", "String"]*10, dtype=np.object_)
    num_buckets = 9
    key = [11, 12]
    name = None

    input_dict = {
        "input": input_tensor,
        "num_buckets": num_buckets,
        "key": key,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.StringToHashBucketStrong"] = tf_raw_ops_string_to_hash_bucket_strong_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.StringToHashBucketStrong' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.StringToHashBucketStrong'.")

check_valid('tf.raw_ops.StringToHashBucketStrong', generated_inputs['tf.raw_ops.StringToHashBucketStrong'], lib="tf", suffix=0)

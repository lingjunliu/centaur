
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_string_to_hash_bucket_strong_inputs():
    list_of_inputs = []

    # Input 1: Basic test
    input_tensor = np.array(["Hello", "TF", "World"], dtype=np.object_)
    num_buckets = 3
    key = [1, 2]
    input_dict = {"input": input_tensor, "num_buckets": num_buckets, "key": key, "name": "basic_test"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different strings
    input_tensor = np.array(["This is a long string", "Short", ""], dtype=np.object_)
    num_buckets = 5
    key = [12345, 67890]
    input_dict = {"input": input_tensor, "num_buckets": num_buckets, "key": key, "name": "different_strings"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Single string
    input_tensor = np.array(["Just one string"], dtype=np.object_)
    num_buckets = 10
    key = [98765, 43210]
    input_dict = {"input": input_tensor, "num_buckets": num_buckets, "key": key, "name": "single_string"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Multiple identical strings
    input_tensor = np.array(["Same", "Same", "Same"], dtype=np.object_)
    num_buckets = 2
    key = [11111, 22222]
    input_dict = {"input": input_tensor, "num_buckets": num_buckets, "key": key, "name": "identical_strings"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Larger number of buckets
    input_tensor = np.array(["A", "B", "C", "D", "E"], dtype=np.object_)
    num_buckets = 100
    key = [33333, 44444]
    input_dict = {"input": input_tensor, "num_buckets": num_buckets, "key": key, "name": "larger_buckets"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different keys
    input_tensor = np.array(["X", "Y", "Z"], dtype=np.object_)
    num_buckets = 4
    key = [55555, 66666]
    input_dict = {"input": input_tensor, "num_buckets": num_buckets, "key": key, "name": "different_keys"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Empty input tensor
    input_tensor = np.array([], dtype=np.object_)
    num_buckets = 7
    key = [77777, 88888]
    input_dict = {"input": input_tensor, "num_buckets": num_buckets, "key": key, "name": "empty_tensor"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Tensor with Unicode strings
    input_tensor = np.array(["你好", "世界", "你好世界"], dtype=np.object_)
    num_buckets = 6
    key = [99999, 10101]
    input_dict = {"input": input_tensor, "num_buckets": num_buckets, "key": key, "name": "unicode_strings"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Very large number of buckets
    input_tensor = np.array(["Large", "buckets"], dtype=np.object_)
    num_buckets = 2**10
    key = [12121, 13131]
    input_dict = {"input": input_tensor, "num_buckets": num_buckets, "key": key, "name": "very_large_buckets"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10:  Long strings and large buckets
    input_tensor = np.array(["This is a very long string to test the hashing function"], dtype=np.object_)
    num_buckets = 2**8
    key = [14141, 15151]
    input_dict = {"input": input_tensor, "num_buckets": num_buckets, "key": key, "name": "long_string_large_buckets"}
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

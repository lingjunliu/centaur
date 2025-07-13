
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_strings_to_hash_bucket_strong_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array(["Hello", "TF"], dtype=np.object_)
    num_buckets = 3
    key = [1, 2]
    name = "hash_example_1"
    input_dict = {"input": input_tensor, "num_buckets": num_buckets, "key": key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array(["string1", "string2", "string3"], dtype=np.object_)
    num_buckets = 5
    key = [100, 200]
    name = "hash_example_2"
    input_dict = {"input": input_tensor, "num_buckets": num_buckets, "key": key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array(["a", "b", "c", "d", "e"], dtype=np.object_)
    num_buckets = 10
    key = [0, 0]
    name = "hash_example_3"
    input_dict = {"input": input_tensor, "num_buckets": num_buckets, "key": key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array([["str1", "str2"], ["str3", "str4"]], dtype=np.object_)
    num_buckets = 2
    key = [65535, 4294967295]
    name = "hash_example_4"
    input_dict = {"input": input_tensor, "num_buckets": num_buckets, "key": key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array(["very_long_string"] * 5, dtype=np.object_)
    num_buckets = 7
    key = [2**32, 2**63 - 1]
    name = "hash_example_5"
    input_dict = {"input": input_tensor, "num_buckets": num_buckets, "key": key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array(["", " ", "\t", "\n"], dtype=np.object_)
    num_buckets = 4
    key = [234567, 890123]
    name = "hash_example_6"
    input_dict = {"input": input_tensor, "num_buckets": num_buckets, "key": key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.array([["abc", "def"], ["ghi", "jkl"], ["mno", "pqr"]], dtype=np.object_)
    num_buckets = 6
    key = [123, 456]
    name = "hash_example_7"
    input_dict = {"input": input_tensor, "num_buckets": num_buckets, "key": key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.array(["!@#$", "%^&*", "()_+"], dtype=np.object_)
    num_buckets = 3
    key = [789, 101112]
    name = "hash_example_8"
    input_dict = {"input": input_tensor, "num_buckets": num_buckets, "key": key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_tensor = np.array(["test1", "test2"], dtype=np.object_)
    num_buckets = 2
    key = [131415, 161718]
    name = "hash_example_9"
    input_dict = {"input": input_tensor, "num_buckets": num_buckets, "key": key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.array(["", "a", "aa", "aaa", "aaaa"], dtype=np.object_)
    num_buckets = 5
    key = [192021, 222324]
    name = "hash_example_10"
    input_dict = {"input": input_tensor, "num_buckets": num_buckets, "key": key, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.strings.to_hash_bucket_strong"] = tf_strings_to_hash_bucket_strong_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.strings.to_hash_bucket_strong' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.strings.to_hash_bucket_strong'.")

check_valid('tf.strings.to_hash_bucket_strong', generated_inputs['tf.strings.to_hash_bucket_strong'], lib="tf", suffix=0)

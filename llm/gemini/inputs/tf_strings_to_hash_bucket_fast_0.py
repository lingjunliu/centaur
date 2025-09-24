
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_strings_to_hash_bucket_fast_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array(["Hello", "TensorFlow", "2.x"], dtype=object)
    num_buckets = 3
    name = None
    input_dict = {"input": input_tensor, "num_buckets": num_buckets, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array(["a", "b", "c", "d", "e"], dtype=object)
    num_buckets = 5
    name = "hash_buckets"
    input_dict = {"input": input_tensor, "num_buckets": num_buckets, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_tensor = np.array(["very", "long", "string"], dtype=object)
    num_buckets = 10
    name = None
    input_dict = {"input": input_tensor, "num_buckets": num_buckets, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_tensor = np.array(["short"], dtype=object)
    num_buckets = 1
    name = "single_bucket"
    input_dict = {"input": input_tensor, "num_buckets": num_buckets, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_tensor = np.array(["", ""], dtype=object)
    num_buckets = 2
    name = None
    input_dict = {"input": input_tensor, "num_buckets": num_buckets, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_tensor = np.array(["123", "456", "789"], dtype=object)
    num_buckets = 7
    name = "numeric_strings"
    input_dict = {"input": input_tensor, "num_buckets": num_buckets, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_tensor = np.array(["special!@#", "characters$%^"], dtype=object)
    num_buckets = 4
    name = None
    input_dict = {"input": input_tensor, "num_buckets": num_buckets, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_tensor = np.array(["unicode", "你好"], dtype=object)
    num_buckets = 6
    name = "unicode_strings"
    input_dict = {"input": input_tensor, "num_buckets": num_buckets, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 9
    input_tensor = np.array(["", "test", ""], dtype=object)
    num_buckets = 8
    name = None
    input_dict = {"input": input_tensor, "num_buckets": num_buckets, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_tensor = np.array(["test1", "test2", "test1"], dtype=object)
    num_buckets = 9
    name = "duplicate_strings"
    input_dict = {"input": input_tensor, "num_buckets": num_buckets, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.strings.to_hash_bucket_fast"] = tf_strings_to_hash_bucket_fast_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.strings.to_hash_bucket_fast' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.strings.to_hash_bucket_fast'.")

check_valid('tf.strings.to_hash_bucket_fast', generated_inputs['tf.strings.to_hash_bucket_fast'], lib="tf", suffix=0)

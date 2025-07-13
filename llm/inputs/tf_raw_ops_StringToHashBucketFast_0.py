
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_string_to_hash_bucket_fast_inputs():
    list_of_inputs = []

    # Input 1: Basic string array and number of buckets
    input_strings = np.array(["Hello", "TensorFlow", "2.x"], dtype=np.unicode_)
    num_buckets = 3
    input_dict = {"input": tf.convert_to_tensor(input_strings, dtype=tf.string), "num_buckets": num_buckets, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Larger number of buckets
    input_strings = np.array(["a", "b", "c", "d", "e"], dtype=np.unicode_)
    num_buckets = 10
    input_dict = {"input": tf.convert_to_tensor(input_strings, dtype=tf.string), "num_buckets": num_buckets, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Single string
    input_strings = np.array(["Single"], dtype=np.unicode_)
    num_buckets = 5
    input_dict = {"input": tf.convert_to_tensor(input_strings, dtype=tf.string), "num_buckets": num_buckets, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Empty string
    input_strings = np.array([""], dtype=np.unicode_)
    num_buckets = 4
    input_dict = {"input": tf.convert_to_tensor(input_strings, dtype=tf.string), "num_buckets": num_buckets, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Array of empty strings
    input_strings = np.array(["", "", ""], dtype=np.unicode_)
    num_buckets = 6
    input_dict = {"input": tf.convert_to_tensor(input_strings, dtype=tf.string), "num_buckets": num_buckets, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Array with mixed strings
    input_strings = np.array(["", "test", ""], dtype=np.unicode_)
    num_buckets = 7
    input_dict = {"input": tf.convert_to_tensor(input_strings, dtype=tf.string), "num_buckets": num_buckets, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Larger strings
    input_strings = np.array(["This is a longer string", "Another longer string"], dtype=np.unicode_)
    num_buckets = 8
    input_dict = {"input": tf.convert_to_tensor(input_strings, dtype=tf.string), "num_buckets": num_buckets, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Strings with special characters
    input_strings = np.array(["!@#$", "%^&*"], dtype=np.unicode_)
    num_buckets = 9
    input_dict = {"input": tf.convert_to_tensor(input_strings, dtype=tf.string), "num_buckets": num_buckets, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 9: Multidimensional input
    input_strings = np.array([["string1", "string2"], ["string3", "string4"]], dtype=np.unicode_)
    num_buckets = 11
    input_dict = {"input": tf.convert_to_tensor(input_strings, dtype=tf.string), "num_buckets": num_buckets, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Unicode strings
    input_strings = np.array(["你好", "世界"], dtype=np.unicode_)
    num_buckets = 12
    input_dict = {"input": tf.convert_to_tensor(input_strings, dtype=tf.string), "num_buckets": num_buckets, "name": None}
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

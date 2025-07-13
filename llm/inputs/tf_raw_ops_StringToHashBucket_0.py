
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_string_to_hash_bucket_inputs():
    list_of_inputs = []

    # Input 1: Simple string tensor and a small number of buckets
    string_tensor = np.array(["hello", "world", "tensorflow"], dtype=np.unicode_)
    num_buckets = 5
    name = "hash_bucket_1"
    input_dict = {"string_tensor": string_tensor, "num_buckets": num_buckets, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: String tensor with empty strings
    string_tensor = np.array(["", "test", ""], dtype=np.unicode_)
    num_buckets = 10
    name = "hash_bucket_2"
    input_dict = {"string_tensor": string_tensor, "num_buckets": num_buckets, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: String tensor with special characters
    string_tensor = np.array(["!", "@", "#", "$", "%"], dtype=np.unicode_)
    num_buckets = 7
    name = "hash_bucket_3"
    input_dict = {"string_tensor": string_tensor, "num_buckets": num_buckets, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: String tensor with numbers
    string_tensor = np.array(["123", "456", "789"], dtype=np.unicode_)
    num_buckets = 3
    name = "hash_bucket_4"
    input_dict = {"string_tensor": string_tensor, "num_buckets": num_buckets, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Larger number of buckets
    string_tensor = np.array(["apple", "banana", "cherry"], dtype=np.unicode_)
    num_buckets = 100
    name = "hash_bucket_5"
    input_dict = {"string_tensor": string_tensor, "num_buckets": num_buckets, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Multidimensional string tensor
    string_tensor = np.array([["one", "two"], ["three", "four"]], dtype=np.unicode_)
    num_buckets = 4
    name = "hash_bucket_6"
    input_dict = {"string_tensor": string_tensor, "num_buckets": num_buckets, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: String tensor with long strings
    string_tensor = np.array(["This is a very long string", "Another very long string"], dtype=np.unicode_)
    num_buckets = 8
    name = "hash_bucket_8"
    input_dict = {"string_tensor": string_tensor, "num_buckets": num_buckets, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: String tensor with a mix of characters and numbers
    string_tensor = np.array(["abc123", "def456", "ghi789"], dtype=np.unicode_)
    num_buckets = 9
    name = "hash_bucket_9"
    input_dict = {"string_tensor": string_tensor, "num_buckets": num_buckets, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Larger Multidimensional string tensor
    string_tensor = np.array([[["a", "b"], ["c", "d"]], [["e", "f"], ["g", "h"]]], dtype=np.unicode_)
    num_buckets = 11
    name = "hash_bucket_10"
    input_dict = {"string_tensor": string_tensor, "num_buckets": num_buckets, "name": name}
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

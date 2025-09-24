
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_encode_base64_inputs():
    list_of_inputs = []

    # Input 1: Empty string
    input_tensor = np.array([""], dtype=np.object_)
    pad_bool = False
    name_str = "empty_string"
    input_dict = {"input": input_tensor, "pad": pad_bool, "name": name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Simple string, no padding
    input_tensor = np.array(["hello"], dtype=np.object_)
    pad_bool = False
    name_str = "hello_no_pad"
    input_dict = {"input": input_tensor, "pad": pad_bool, "name": name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Simple string, with padding
    input_tensor = np.array(["hello"], dtype=np.object_)
    pad_bool = True
    name_str = "hello_with_pad"
    input_dict = {"input": input_tensor, "pad": pad_bool, "name": name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: String with special characters
    input_tensor = np.array(["!@#$%^&*()_+=-`~[]{}|;':\",./<>?"], dtype=np.object_)
    pad_bool = False
    name_str = "special_chars"
    input_dict = {"input": input_tensor, "pad": pad_bool, "name": name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Multiple strings in a tensor
    input_tensor = np.array(["hello", "world", "tensorflow"], dtype=np.object_)
    pad_bool = False
    name_str = "multiple_strings"
    input_dict = {"input": input_tensor, "pad": pad_bool, "name": name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Multiple strings, with padding
    input_tensor = np.array(["hello", "world", "tensorflow"], dtype=np.object_)
    pad_bool = True
    name_str = "multiple_strings_pad"
    input_dict = {"input": input_tensor, "pad": pad_bool, "name": name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Long string
    input_tensor = np.array(["This is a very long string to test the base64 encoding."], dtype=np.object_)
    pad_bool = False
    name_str = "long_string"
    input_dict = {"input": input_tensor, "pad": pad_bool, "name": name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: String with numbers
    input_tensor = np.array(["1234567890"], dtype=np.object_)
    pad_bool = False
    name_str = "numbers_string"
    input_dict = {"input": input_tensor, "pad": pad_bool, "name": name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: String with a mix of everything
    input_tensor = np.array(["Hello World! 12345 @#$%^&*()"], dtype=np.object_)
    pad_bool = True
    name_str = "mixed_string"
    input_dict = {"input": input_tensor, "pad": pad_bool, "name": name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Another string with special characters
    input_tensor = np.array(["~!@#$^&*()_+=-`"], dtype=np.object_)
    pad_bool = False
    name_str = "more_special_chars"
    input_dict = {"input": input_tensor, "pad": pad_bool, "name": name_str}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.EncodeBase64"] = tf_raw_ops_encode_base64_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.EncodeBase64' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.EncodeBase64'.")

check_valid('tf.raw_ops.EncodeBase64', generated_inputs['tf.raw_ops.EncodeBase64'], lib="tf", suffix=0)

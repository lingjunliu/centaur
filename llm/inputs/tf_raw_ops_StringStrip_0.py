
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_string_strip_inputs():
    list_of_inputs = []

    # Input 1: Basic string tensor
    input_tensor = np.array(["  hello  ", " world "], dtype=np.unicode_)
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: String tensor with different kinds of whitespace
    input_tensor = np.array(["\n\t  mixed \r\n", "  whitespace\t"], dtype=np.unicode_)
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Empty string tensor
    input_tensor = np.array(["", "  ", " "], dtype=np.unicode_)
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Tensor with special characters and whitespace
    input_tensor = np.array(["  special!@#$  ", "  chars%^&*  "], dtype=np.unicode_)
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Tensor with leading and trailing non-whitespace characters
    input_tensor = np.array(["abc  def  ghi", " jkl mno pqr "], dtype=np.unicode_)
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Multidimensional tensor
    input_tensor = np.array([["  first  ", " second "], ["third  ", "  fourth"]], dtype=np.unicode_)
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Tensor with numbers as strings
    input_tensor = np.array(["  123  ", " 456 "], dtype=np.unicode_)
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Tensor with only whitespaces
    input_tensor = np.array(["   ", "  \t "], dtype=np.unicode_)
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Tensor with mixed alphanumeric and whitespace
    input_tensor = np.array(["  a1b2c  ", " 3d4e5 "], dtype=np.unicode_)
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Empty tensor
    input_tensor = np.array([], dtype=np.unicode_)
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))
    

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.StringStrip"] = tf_raw_ops_string_strip_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.StringStrip' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.StringStrip'.")

check_valid('tf.raw_ops.StringStrip', generated_inputs['tf.raw_ops.StringStrip'], lib="tf", suffix=0)

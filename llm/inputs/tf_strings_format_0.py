
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_strings_format_inputs():
    list_of_inputs = []

    # Input 1: Basic single tensor formatting
    template = "Tensor: {}"
    inputs = [np.array([1, 2, 3, 4, 5], dtype=np.int32)]
    placeholder = "{}"
    summarize = 3
    name = None
    input_dict = {"template": template, "inputs": [inputs[0]], "placeholder": placeholder, "summarize": summarize, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Multiple tensors
    template = "A: {}, B: {}"
    inputs = [np.array([1, 2], dtype=np.int32), np.array([3, 4], dtype=np.int32)]
    placeholder = "{}"
    summarize = 3
    name = None
    input_dict = {"template": template, "inputs": inputs, "placeholder": placeholder, "summarize": summarize, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different placeholder
    template = "Value is %s"
    inputs = [np.array(10, dtype=np.int32)]
    placeholder = "%s"
    summarize = 3
    name = None
    input_dict = {"template": template, "inputs": [inputs[0]], "placeholder": placeholder, "summarize": summarize, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Summarize -1 (show all)
    template = "Tensor: {}"
    inputs = [np.array([1, 2, 3, 4, 5], dtype=np.int32)]
    placeholder = "{}"
    summarize = -1
    name = None
    input_dict = {"template": template, "inputs": [inputs[0]], "placeholder": placeholder, "summarize": summarize, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Multi-dimensional tensor
    template = "Matrix: {}"
    inputs = [np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)]
    placeholder = "{}"
    summarize = 2
    name = None
    input_dict = {"template": template, "inputs": [inputs[0]], "placeholder": placeholder, "summarize": summarize, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: String tensor
    template = "String: {}"
    inputs = [np.array(["hello", "world"], dtype=np.unicode_)]
    placeholder = "{}"
    summarize = 3
    name = None
    input_dict = {"template": template, "inputs": [inputs[0]], "placeholder": placeholder, "summarize": summarize, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large summarize value
    template = "Tensor: {}"
    inputs = [np.array([1, 2, 3, 4, 5], dtype=np.int32)]
    placeholder = "{}"
    summarize = 10
    name = None
    input_dict = {"template": template, "inputs": [inputs[0]], "placeholder": placeholder, "summarize": summarize, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 8: Template with no placeholders
    template = "No placeholders"
    inputs = [] # No tensors expected as there's no placeholder
    placeholder = "{}"
    summarize = 3
    name = None
    input_dict = {"template": template, "inputs": inputs, "placeholder": placeholder, "summarize": summarize, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: More complex template
    template = "Value1: {}, Value2: {}, Value3:{}"
    inputs = [np.array(1, dtype=np.int32), np.array([2, 3], dtype=np.int32), np.array([[4, 5], [6, 7]], dtype=np.int32)]
    placeholder = "{}"
    summarize = 1
    name = None
    input_dict = {"template": template, "inputs": inputs, "placeholder": placeholder, "summarize": summarize, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: name parameter
    template = "Tensor: {}"
    inputs = [np.array([1, 2, 3], dtype=np.int32)]
    placeholder = "{}"
    summarize = 3
    name = "my_format_op"
    input_dict = {"template": template, "inputs": [inputs[0]], "placeholder": placeholder, "summarize": summarize, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.strings.format"] = tf_strings_format_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.strings.format' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.strings.format'.")

check_valid('tf.strings.format', generated_inputs['tf.strings.format'], lib="tf", suffix=0)

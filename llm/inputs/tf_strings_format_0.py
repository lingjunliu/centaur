
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_strings_format_inputs():
    list_of_inputs = []

    # Input 1: Single tensor, default placeholder and summarize
    template = "Tensor: {}"
    inputs = [np.array([1, 2, 3, 4, 5])]
    placeholder = "{}"
    summarize = 3
    name = None
    input_dict = {"template": template, "inputs": inputs, "placeholder": placeholder, "summarize": summarize, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Multiple tensors, custom placeholder
    template = "A: {}, B: {}"
    inputs = [np.array([10, 20]), np.array([30, 40])]
    placeholder = "{}"
    summarize = 2
    name = "format_example"
    input_dict = {"template": template, "inputs": inputs, "placeholder": placeholder, "summarize": summarize, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Single tensor, summarize -1 (all elements)
    template = "Matrix: {}"
    inputs = [np.array([[1, 2], [3, 4]])]
    placeholder = "{}"
    summarize = -1
    name = None
    input_dict = {"template": template, "inputs": inputs, "placeholder": placeholder, "summarize": summarize, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Multi-dimensional tensor
    template = "3D Tensor: {}"
    inputs = [np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])]
    placeholder = "{}"
    summarize = 1
    name = None
    input_dict = {"template": template, "inputs": inputs, "placeholder": placeholder, "summarize": summarize, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: No summarize
    template = "Tensor: {}"
    inputs = [np.array([1, 2, 3])]
    placeholder = "{}"
    summarize = 3
    name = None
    input_dict = {"template": template, "inputs": inputs, "placeholder": placeholder, "summarize": summarize, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Empty tensor
    template = "Empty: {}"
    inputs = [np.array([])]
    placeholder = "{}"
    summarize = 3
    name = None
    input_dict = {"template": template, "inputs": inputs, "placeholder": placeholder, "summarize": summarize, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Placeholder not default
    template = "Test: <tensor>"
    inputs = [np.array([1, 2, 3])]
    placeholder = "<tensor>"
    summarize = 3
    name = None
    input_dict = {"template": template, "inputs": inputs, "placeholder": placeholder, "summarize": summarize, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Multiple tensors, different summarize
    template = "A: {}, B: {}"
    inputs = [np.array([1, 2, 3, 4]), np.array([5, 6, 7])]
    placeholder = "{}"
    summarize = 1
    name = None
    input_dict = {"template": template, "inputs": inputs, "placeholder": placeholder, "summarize": summarize, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 9: String template with no placeholder
    template = "No placeholders here!"
    inputs = [np.array([1])] # added a dummy input
    placeholder = "{}"
    summarize = 3
    name = None
    input_dict = {"template": template, "inputs": inputs, "placeholder": placeholder, "summarize": summarize, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: More complicated template
    template = "Values are: {}, {}, {}"
    inputs = [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])]
    placeholder = "{}"
    summarize = 2
    name = None
    input_dict = {"template": template, "inputs": inputs, "placeholder": placeholder, "summarize": summarize, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Multiple tensors and placeholder
    template = "a: {}, b: {}"
    inputs = [np.array([1, 2, 3]), np.array([4, 5])]
    placeholder = "{}"
    summarize = 2
    name = None
    input_dict = {"template": template, "inputs": inputs, "placeholder": placeholder, "summarize": summarize, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12: Multiple placeholders with the same name
    template = "Value: {x}, again: {x}"
    inputs = [np.array([1,2])]
    placeholder = "{x}"
    summarize = 3
    name = None
    input_dict = {"template": template, "inputs": inputs, "placeholder": placeholder, "summarize": summarize, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 13: different datatype
    template = "Tensor: {}"
    inputs = [np.array([1.1, 2.2, 3.3])]
    placeholder = "{}"
    summarize = 3
    name = None
    input_dict = {"template": template, "inputs": inputs, "placeholder": placeholder, "summarize": summarize, "name": name}
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

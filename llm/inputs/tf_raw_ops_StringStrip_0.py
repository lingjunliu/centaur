
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_string_strip_inputs():
    list_of_inputs = []

    # Input 1: Basic string
    input_tensor = np.array("   TensorFlow   ", dtype=np.object_)
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: String with leading and trailing newlines
    input_tensor = np.array("\n\n  TensorFlow  \n\n", dtype=np.object_)
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: String with only whitespace
    input_tensor = np.array("     ", dtype=np.object_)
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Empty string
    input_tensor = np.array("", dtype=np.object_)
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: String with no whitespace
    input_tensor = np.array("TensorFlow", dtype=np.object_)
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Multiple strings in a list
    input_tensor = np.array(["   TensorFlow", "The python library    ", "  "], dtype=np.object_)
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Multiple strings with different whitespace patterns
    input_tensor = np.array(["\n TensorFlow", "The python library \t", "   ", ""], dtype=np.object_)
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: String with inner whitespace
    input_tensor = np.array("  Tensor Flow  ", dtype=np.object_)
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: String with unicode whitespace
    input_tensor = np.array(" \u3000TensorFlow\u3000 ", dtype=np.object_)
    input_dict = {"input": input_tensor, "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Batched input
    input_tensor = np.array([["   TensorFlow   ", "  Python  "], ["  ML  ", "  "]], dtype=np.object_)
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

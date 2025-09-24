
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_strings_unicode_script_inputs():
    list_of_inputs = []

    # Input 1: Basic example
    input_tensor = np.array([1, 31, 38], dtype=np.int32)
    input_dict = {"input": tf.constant(input_tensor), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Empty tensor
    input_tensor = np.array([], dtype=np.int32)
    input_dict = {"input": tf.constant(input_tensor), "name": "empty_tensor"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Tensor with invalid code points
    input_tensor = np.array([-1, 1114112, 0], dtype=np.int32)
    input_dict = {"input": tf.constant(input_tensor), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Multidimensional tensor
    input_tensor = np.array([[65, 66], [67, 68]], dtype=np.int32)
    input_dict = {"input": tf.constant(input_tensor), "name": "multi_dim"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Larger tensor
    input_tensor = np.arange(65, 75, dtype=np.int32)
    input_dict = {"input": tf.constant(input_tensor), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Tensor with a specific Unicode range
    input_tensor = np.arange(128, 256, dtype=np.int32)
    input_dict = {"input": tf.constant(input_tensor), "name": "unicode_range"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Tensor with mixed code points
    input_tensor = np.array([65, 12354, 20013, 97], dtype=np.int32)
    input_dict = {"input": tf.constant(input_tensor), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D Tensor
    input_tensor = np.array([[[65, 66], [67, 68]], [[69, 70], [71, 72]]], dtype=np.int32)
    input_dict = {"input": tf.constant(input_tensor), "name": "3d_tensor"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Tensor with a single element
    input_tensor = np.array([65], dtype=np.int32)
    input_dict = {"input": tf.constant(input_tensor), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Tensor with various ASCII characters
    input_tensor = np.array([33, 48, 65, 97, 126], dtype=np.int32)
    input_dict = {"input": tf.constant(input_tensor), "name": "ascii_chars"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.strings.unicode_script"] = tf_strings_unicode_script_inputs()
for i in range(len(generated_inputs["tf.strings.unicode_script"])):
    generated_inputs["tf.strings.unicode_script"][i]["input"] = generated_inputs["tf.strings.unicode_script"][i]["input"].numpy()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.strings.unicode_script' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.strings.unicode_script'.")

check_valid('tf.strings.unicode_script', generated_inputs['tf.strings.unicode_script'], lib="tf", suffix=0)

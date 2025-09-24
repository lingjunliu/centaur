
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_saved_model_asset_inputs():
    list_of_inputs = []

    # Input 1: Simple file name
    path = "my_asset.txt"
    input_dict = {"path": path}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Path with directory
    path = "assets/my_other_asset.txt"
    input_dict = {"path": path}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Absolute path
    path = "/tmp/my_absolute_asset.txt"
    input_dict = {"path": path}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Path with special characters
    path = "my_asset_with_$#@!.txt"
    input_dict = {"path": path}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Empty string
    path = ""
    input_dict = {"path": path}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Long path
    path = "a" * 200 + ".txt"
    input_dict = {"path": path}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Path with unicode characters
    path = "资产.txt"
    input_dict = {"path": path}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Path with spaces
    path = "my asset.txt"
    input_dict = {"path": path}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Path with multiple subdirectories
    path = "assets/subdir1/subdir2/my_asset.txt"
    input_dict = {"path": path}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Path starting with a dot
    path = "./my_asset.txt"
    input_dict = {"path": path}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.saved_model.Asset"] = tf_saved_model_asset_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.saved_model.Asset' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.saved_model.Asset'.")

check_valid('tf.saved_model.Asset', generated_inputs['tf.saved_model.Asset'], lib="tf", suffix=0)

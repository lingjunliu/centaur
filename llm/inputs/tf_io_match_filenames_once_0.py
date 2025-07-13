
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_io_match_filenames_once_inputs():
    list_of_inputs = []

    # Input 1: Simple pattern
    pattern = np.array("pattern1.txt", dtype=np.str_)
    name = "files1"
    input_dict = {"pattern": tf.convert_to_tensor(pattern, dtype=tf.string), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Pattern with directory
    pattern = np.array("data/pattern2.csv", dtype=np.str_)
    name = "files2"
    input_dict = {"pattern": tf.convert_to_tensor(pattern, dtype=tf.string), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Multiple patterns as a 1D tensor
    pattern = np.array(["pattern3.txt", "pattern4.csv"], dtype=np.str_)
    name = "files3"
    input_dict = {"pattern": tf.convert_to_tensor(pattern, dtype=tf.string), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Empty pattern
    pattern = np.array("", dtype=np.str_)
    name = "files4"
    input_dict = {"pattern": tf.convert_to_tensor(pattern, dtype=tf.string), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Pattern with a specific file
    pattern = np.array("pattern5.log", dtype=np.str_)
    name = "files5"
    input_dict = {"pattern": tf.convert_to_tensor(pattern, dtype=tf.string), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Pattern with character range
    pattern = np.array("file[0-9].txt", dtype=np.str_)
    name = "files6"
    input_dict = {"pattern": tf.convert_to_tensor(pattern, dtype=tf.string), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Pattern with wildcard at the beginning
    pattern = np.array("*.log", dtype=np.str_)
    name = "files7"
    input_dict = {"pattern": tf.convert_to_tensor(pattern, dtype=tf.string), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Complex pattern
    pattern = np.array("data/*[a-z]*.dat", dtype=np.str_)
    name = "files8"
    input_dict = {"pattern": tf.convert_to_tensor(pattern, dtype=tf.string), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Unicode filename pattern
    pattern = np.array("你好.txt", dtype=np.str_)
    name = "files9"
    input_dict = {"pattern": tf.convert_to_tensor(pattern, dtype=tf.string), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Pattern with special characters
    pattern = np.array("file?.txt", dtype=np.str_)
    name = "files10"
    input_dict = {"pattern": tf.convert_to_tensor(pattern, dtype=tf.string), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.io.match_filenames_once"] = tf_io_match_filenames_once_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.io.match_filenames_once' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.match_filenames_once'.")

check_valid('tf.io.match_filenames_once', generated_inputs['tf.io.match_filenames_once'], lib="tf", suffix=0)

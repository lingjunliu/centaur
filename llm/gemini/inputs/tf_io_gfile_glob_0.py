
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy

def tf_io_gfile_glob_inputs():
    list_of_inputs = []

    # Input 1: Simple wildcard
    input_dict = {"pattern": "*.txt"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Question mark wildcard
    input_dict = {"pattern": "file?.txt"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Character range
    input_dict = {"pattern": "file[0-9].txt"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negated character range
    input_dict = {"pattern": "file[!0-9].txt"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Specific character match
    input_dict = {"pattern": "file[abc].txt"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Escaped character
    input_dict = {"pattern": "file\\*.txt"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Multiple wildcards
    input_dict = {"pattern": "*/*.txt"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: More complex pattern
    input_dict = {"pattern": "dir?/file[0-9]*.txt"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Pattern with only a directory
    input_dict = {"pattern": "mydir/"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Pattern with no wildcards, matching a specific file
    input_dict = {"pattern": "myfile.txt"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.io.gfile.glob"] = tf_io_gfile_glob_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.io.gfile.glob' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.gfile.glob'.")

check_valid('tf.io.gfile.glob', generated_inputs['tf.io.gfile.glob'], lib="tf", suffix=0)

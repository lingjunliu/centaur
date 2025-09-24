
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_io_match_filenames_once_inputs():
    """
    Generates a list of valid inputs for the tf.io.match_filenames_once function.
    """
    list_of_inputs = []

    # Input 1: Basic wildcard '*' for a single pattern
    input_dict_1 = {
        'pattern': np.array('data/*.csv', dtype=object),
        'name': 'match_csv_files'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Wildcard '?' for a single pattern, no name
    input_dict_2 = {
        'pattern': np.array('image_??.png', dtype=object),
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Wildcard '[]' for character range
    input_dict_3 = {
        'pattern': np.array('log_202[0-9].txt', dtype=object),
        'name': 'match_logs_by_year'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Recursive wildcard '**'
    input_dict_4 = {
        'pattern': np.array('/tmp/data/**/*.json', dtype=object),
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: A 1D tensor of multiple patterns
    input_dict_5 = {
        'pattern': np.array(['/data/*.jpg', '/data/*.jpeg'], dtype=object),
        'name': 'match_all_jpegs'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: A 1D tensor with a single pattern
    input_dict_6 = {
        'pattern': np.array(['/etc/config/specific_file.conf'], dtype=object),
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: A pattern with no wildcards (matches a single file)
    input_dict_7 = {
        'pattern': np.array('README.md', dtype=object),
        'name': 'find_readme'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: A 1D tensor with mixed wildcard types
    input_dict_8 = {
        'pattern': np.array(['/var/log/sys??.*', '/home/user/docs/**/*.docx'], dtype=object),
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: A single empty string pattern (edge case)
    input_dict_9 = {
        'pattern': np.array('', dtype=object),
        'name': 'empty_pattern'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: A 1D tensor containing an empty string along with a valid pattern
    input_dict_10 = {
        'pattern': np.array(['/path/to/files/*', ''], dtype=object),
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: A 1D tensor with three different complex patterns
    input_dict_11 = {
        'pattern': np.array(['a/b[0-9]/*.tfrecord', 'c/d?/*.tfrecord', 'e/f/*/*.tfrecord'], dtype=object),
        'name': 'match_tfrecords'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))
    
    # Input 12: A pattern that looks like a directory
    input_dict_12 = {
        'pattern': np.array('/usr/local/bin/*', dtype=object),
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict_12))

    return list_of_inputs

generated_inputs["tf.io.match_filenames_once"] = tf_io_match_filenames_once_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.io.match_filenames_once' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.match_filenames_once'.")

check_valid('tf.io.match_filenames_once', generated_inputs['tf.io.match_filenames_once'], lib="tf", suffix=0)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy

def tf_critical_section_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        "name": "cs_1",
        "shared_name": None,
        "critical_section_def": None,
        "import_scope": "scope_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        "name": "cs_2",
        "shared_name": None,
        "critical_section_def": None,
        "import_scope": ""
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        "name": None,
        "shared_name": "shared_cs_3",
        "critical_section_def": None,
        "import_scope": "scope_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        "name": "cs_4",
        "shared_name": None,
        "critical_section_def": None,
        "import_scope": ""
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        "name": None,
        "shared_name": None,
        "critical_section_def": None,
        "import_scope": "scope_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        "name": "cs_6",
        "shared_name": None,
        "critical_section_def": None,
        "import_scope": "scope_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        "name": "critical_section",
        "shared_name": None,
        "critical_section_def": None,
        "import_scope": "import_scope"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 8
    input_dict = {
        "name": "another_critical_section",
        "shared_name": None,
        "critical_section_def": None,
        "import_scope": ""
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        "name": None,
        "shared_name": "yet_another_shared_name",
        "critical_section_def": None,
        "import_scope": "yet_another_import_scope"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        "name": "cs_10",
        "shared_name": None,
        "critical_section_def": None,
        "import_scope": ""
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.CriticalSection"] = tf_critical_section_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.CriticalSection' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.CriticalSection'.")

check_valid('tf.CriticalSection', generated_inputs['tf.CriticalSection'], lib="tf", suffix=0)

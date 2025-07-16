
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_DecodeJSONExample_inputs():
    list_of_inputs = []

    # Input 1
    json_examples = np.array(["{\"features\":{\"feature\":{\"f1\":{\"float_list\":{\"value\":[1.0, 2.0]}}}}}"], dtype=np.string_)
    name = None
    input_dict = {"json_examples": json_examples, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    json_examples = np.array(["{\"features\":{\"feature\":{\"f1\":{\"int64_list\":{\"value\":[1, 2]}}}}}"], dtype=np.string_)
    name = ""
    input_dict = {"json_examples": json_examples, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    json_examples = np.array(["{\"features\":{\"feature\":{\"f1\":{\"bytes_list\":{\"value\":[b'abc', b'def']}}}}}"], dtype=np.string_)
    name = "DecodeJSONExample_1"
    input_dict = {"json_examples": json_examples, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    json_examples = np.array(["{\"features\":{\"feature\":{\"f1\":{\"float_list\":{\"value\":[1.0]}}, \"f2\":{\"int64_list\":{\"value\":[1]}}}}}",
                            "{\"features\":{\"feature\":{\"f1\":{\"float_list\":{\"value\":[2.0]}}, \"f2\":{\"int64_list\":{\"value\":[2]}}}}}",
                            "{\"features\":{\"feature\":{\"f1\":{\"float_list\":{\"value\":[3.0]}}, \"f2\":{\"int64_list\":{\"value\":[3]}}}}}"], dtype=np.string_)
    name = "DecodeJSONExample_2"
    input_dict = {"json_examples": json_examples, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    json_examples = np.array(["{\"features\":{\"feature\":{}}}", "{\"features\":{\"feature\":{}}}", "{\"features\":{\"feature\":{}}}"], dtype=np.string_)
    name = "DecodeJSONExample_3"
    input_dict = {"json_examples": json_examples, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    json_examples = np.array(["{\"features\":{\"feature\":{\"f1\":{\"bytes_list\":{\"value\":[b'']}}}}}"], dtype=np.string_)
    name = "DecodeJSONExample_4"
    input_dict = {"json_examples": json_examples, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    json_examples = np.array(["{\"features\":{\"feature\":{\"f1\":{\"float_list\":{\"value\":[]}}}}}"], dtype=np.string_)
    name = "DecodeJSONExample_5"
    input_dict = {"json_examples": json_examples, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    json_examples = np.array(["{\"features\":{\"feature\":{\"f1\":{\"int64_list\":{\"value\":[]}}}}}"], dtype=np.string_)
    name = "DecodeJSONExample_6"
    input_dict = {"json_examples": json_examples, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Multiple Features, some empty
    json_examples = np.array(["{\"features\":{\"feature\":{\"f1\":{\"int64_list\":{\"value\":[1,2]}}, \"f2\": {}}}}"], dtype=np.string_)
    name = "DecodeJSONExample_7"
    input_dict = {"json_examples": json_examples, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Features with empty lists and strings
    json_examples = np.array(["{\"features\":{\"feature\":{\"f1\":{\"bytes_list\":{\"value\":[b'']}}, \"f2\": {\"float_list\": {\"value\": []}}}}}",
                             "{\"features\":{\"feature\":{\"f1\":{\"bytes_list\":{\"value\":[]}}, \"f2\": {\"float_list\": {\"value\": [1.0, 2.0]}}}}}",
                             "{\"features\":{\"feature\":{\"f1\":{\"bytes_list\":{\"value\":[b'abc']}}, \"f2\": {\"float_list\": {\"value\": []}}}}}"], dtype=np.string_)
    name = "DecodeJSONExample_8"
    input_dict = {"json_examples": json_examples, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    json_examples = np.array(["{\"features\":{\"feature\":{\"f1\":{\"bytes_list\":{\"value\":[b'abc', b'def']}}}}}"], dtype=np.string_)
    name = b"DecodeJSONExample_1"

    input_dict = {"json_examples": json_examples, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.DecodeJSONExample"] = tf_raw_ops_DecodeJSONExample_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.DecodeJSONExample' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DecodeJSONExample'.")

check_valid('tf.raw_ops.DecodeJSONExample', generated_inputs['tf.raw_ops.DecodeJSONExample'], lib="tf", suffix=0)

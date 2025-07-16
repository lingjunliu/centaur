
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_io_encode_proto_inputs():
    list_of_inputs = []

    # Input 1
    sizes = np.array([[1, 1]], dtype=np.int32)
    values = [np.array([[1]], dtype=np.int32), np.array([[2]], dtype=np.int32)]
    field_names = ["field1", "field2"]
    message_type = "TestMessage"
    descriptor_source = "local://"
    name = "encode_proto_op_1"

    input_dict = {
        "sizes": sizes,
        "values": [np.array(v) for v in values],
        "field_names": field_names,
        "message_type": message_type,
        "descriptor_source": descriptor_source,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    sizes = np.array([[2, 1]], dtype=np.int32)
    values = [np.array([[1, 2]], dtype=np.int32), np.array([[3]], dtype=np.int32)]
    field_names = ["field1", "field2"]
    message_type = "AnotherMessage"
    descriptor_source = ""
    name = None

    input_dict = {
        "sizes": sizes,
        "values": [np.array(v) for v in values],
        "field_names": field_names,
        "message_type": message_type,
        "descriptor_source": descriptor_source,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    sizes = np.array([[[1, 1], [1, 1]]], dtype=np.int32)
    values = [np.array([[[1], [2]]], dtype=np.int32), np.array([[[3], [4]]], dtype=np.int32)]
    field_names = ["field1", "field2"]
    message_type = "NestedMessage"
    descriptor_source = "local://"
    name = "encode_proto_op_3"

    input_dict = {
        "sizes": sizes,
        "values": [np.array(v) for v in values],
        "field_names": field_names,
        "message_type": message_type,
        "descriptor_source": descriptor_source,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.io.encode_proto"] = tf_io_encode_proto_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.io.encode_proto' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.encode_proto'.")

check_valid('tf.io.encode_proto', generated_inputs['tf.io.encode_proto'], lib="tf", suffix=0)

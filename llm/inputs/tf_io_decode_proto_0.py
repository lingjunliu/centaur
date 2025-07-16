
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_io_decode_proto_inputs():
    list_of_inputs = []

    # Input 1
    bytes_val = np.array([b'\x08\x01'], dtype=np.string_)
    message_type = "test.TestMessage"
    field_names = ["int_field"]
    output_types = [tf.int32]
    descriptor_source = "local://"
    message_format = "binary"
    sanitize = False
    name = "decode_proto_1"

    input_dict = {
        "bytes": bytes_val,
        "message_type": message_type,
        "field_names": field_names,
        "output_types": output_types,
        "descriptor_source": descriptor_source,
        "message_format": message_format,
        "sanitize": sanitize,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    bytes_val = np.array([b'\x08\x02'], dtype=np.string_)
    message_type = "test.TestMessage"
    field_names = ["int_field"]
    output_types = [tf.int64]
    descriptor_source = ""
    message_format = "binary"
    sanitize = True
    name = "decode_proto_2"

    input_dict = {
        "bytes": bytes_val,
        "message_type": message_type,
        "field_names": field_names,
        "output_types": output_types,
        "descriptor_source": descriptor_source,
        "message_format": message_format,
        "sanitize": sanitize,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    bytes_val = np.array([b'field1: "abc"'], dtype=np.string_)
    message_type = "test.TestMessage"
    field_names = ["field1"]
    output_types = [tf.string]
    descriptor_source = "local://"
    message_format = "text"
    sanitize = False
    name = "decode_proto_3"

    input_dict = {
        "bytes": bytes_val,
        "message_type": message_type,
        "field_names": field_names,
        "output_types": output_types,
        "descriptor_source": descriptor_source,
        "message_format": message_format,
        "sanitize": sanitize,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    bytes_val = np.array([b'\x00'], dtype=np.string_)
    message_type = "test.TestMessage"
    field_names = ["bool_field"]
    output_types = [tf.bool]
    descriptor_source = "local://"
    message_format = "binary"
    sanitize = False
    name = "decode_proto_4"

    input_dict = {
        "bytes": bytes_val,
        "message_type": message_type,
        "field_names": field_names,
        "output_types": output_types,
        "descriptor_source": descriptor_source,
        "message_format": message_format,
        "sanitize": sanitize,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 5
    bytes_val = np.array([b'\x08\x01'], dtype=np.string_)
    message_type = "test.TestMessage"
    field_names = ["int_field"]
    output_types = [tf.int32]
    descriptor_source = "local://"
    message_format = "binary"
    sanitize = False
    name = "decode_proto_5"

    input_dict = {
        "bytes": bytes_val,
        "message_type": message_type,
        "field_names": field_names,
        "output_types": output_types,
        "descriptor_source": descriptor_source,
        "message_format": message_format,
        "sanitize": sanitize,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.io.decode_proto"] = tf_io_decode_proto_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.io.decode_proto' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.decode_proto'.")

check_valid('tf.io.decode_proto', generated_inputs['tf.io.decode_proto'], lib="tf", suffix=0)

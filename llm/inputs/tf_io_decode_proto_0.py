
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_io_decode_proto_inputs():
    list_of_inputs = []

    # Input 1: Basic example with one field
    bytes_in = tf.constant([b'\x08\x96\x01']).numpy()
    message_type = "test.TestMessage"
    field_names = ["int_field"]
    output_types = ["DT_INT32"]
    descriptor_source = "local://"
    message_format = "binary"
    sanitize = False
    name = None

    input_dict = {
        "bytes": tf.convert_to_tensor(bytes_in),
        "message_type": message_type,
        "field_names": field_names,
        "output_types": [x for x in output_types],
        "descriptor_source": descriptor_source,
        "message_format": message_format,
        "sanitize": sanitize,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Example with two fields
    bytes_in = tf.constant([b'\x12\x06test12', b'\x12\x06test34']).numpy()
    message_type = "test.TestMessage"
    field_names = ["string_field"]
    output_types = ["DT_STRING"]
    descriptor_source = "local://"
    message_format = "binary"
    sanitize = False
    name = "decode_test"

    input_dict = {
        "bytes": tf.convert_to_tensor(bytes_in),
        "message_type": message_type,
        "field_names": field_names,
        "output_types": [x for x in output_types],
        "descriptor_source": descriptor_source,
        "message_format": message_format,
        "sanitize": sanitize,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different descriptor source
    bytes_in = tf.constant([b'\x08\x01']).numpy()
    message_type = "test.TestMessage"
    field_names = ["int_field"]
    output_types = ["DT_INT32"]
    descriptor_source = "bytes://some_bytes" #Invalid but accepted type
    message_format = "binary"
    sanitize = False
    name = None

    input_dict = {
        "bytes": tf.convert_to_tensor(bytes_in),
        "message_type": message_type,
        "field_names": field_names,
        "output_types": [x for x in output_types],
        "descriptor_source": descriptor_source,
        "message_format": message_format,
        "sanitize": sanitize,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Text format
    bytes_in = tf.constant([b'int_field: 1']).numpy()
    message_type = "test.TestMessage"
    field_names = ["int_field"]
    output_types = ["DT_INT32"]
    descriptor_source = "local://"
    message_format = "text"
    sanitize = False
    name = None

    input_dict = {
        "bytes": tf.convert_to_tensor(bytes_in),
        "message_type": message_type,
        "field_names": field_names,
        "output_types": [x for x in output_types],
        "descriptor_source": descriptor_source,
        "message_format": message_format,
        "sanitize": sanitize,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Multiple fields, different types
    bytes_in = tf.constant([b'\x08\x01\x12\x06test56']).numpy()
    message_type = "test.TestMessage"
    field_names = ["int_field", "string_field"]
    output_types = ["DT_INT32", "DT_STRING"]
    descriptor_source = "local://"
    message_format = "binary"
    sanitize = False
    name = None

    input_dict = {
        "bytes": tf.convert_to_tensor(bytes_in),
        "message_type": message_type,
        "field_names": field_names,
        "output_types": [x for x in output_types],
        "descriptor_source": descriptor_source,
        "message_format": message_format,
        "sanitize": sanitize,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Empty bytes tensor
    bytes_in = tf.constant([]).numpy()
    message_type = "test.TestMessage"
    field_names = ["int_field"]
    output_types = ["DT_INT32"]
    descriptor_source = "local://"
    message_format = "binary"
    sanitize = False
    name = None

    input_dict = {
        "bytes": tf.convert_to_tensor(bytes_in),
        "message_type": message_type,
        "field_names": field_names,
        "output_types": [x for x in output_types],
        "descriptor_source": descriptor_source,
        "message_format": message_format,
        "sanitize": sanitize,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Bytes tensor with multiple examples
    bytes_in = tf.constant([b'\x08\x01', b'\x08\x02', b'\x08\x03']).numpy()
    message_type = "test.TestMessage"
    field_names = ["int_field"]
    output_types = ["DT_INT32"]
    descriptor_source = "local://"
    message_format = "binary"
    sanitize = False
    name = None

    input_dict = {
        "bytes": tf.convert_to_tensor(bytes_in),
        "message_type": message_type,
        "field_names": field_names,
        "output_types": [x for x in output_types],
        "descriptor_source": descriptor_source,
        "message_format": message_format,
        "sanitize": sanitize,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Empty field names
    bytes_in = tf.constant([b'\x08\x01']).numpy()
    message_type = "test.TestMessage"
    field_names = []
    output_types = []
    descriptor_source = "local://"
    message_format = "binary"
    sanitize = False
    name = None

    input_dict = {
        "bytes": tf.convert_to_tensor(bytes_in),
        "message_type": message_type,
        "field_names": field_names,
        "output_types": [x for x in output_types],
        "descriptor_source": descriptor_source,
        "message_format": message_format,
        "sanitize": sanitize,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: sanitize = True
    bytes_in = tf.constant([b'\x08\x01']).numpy()
    message_type = "test.TestMessage"
    field_names = ["int_field"]
    output_types = ["DT_INT32"]
    descriptor_source = "local://"
    message_format = "binary"
    sanitize = True
    name = None

    input_dict = {
        "bytes": tf.convert_to_tensor(bytes_in),
        "message_type": message_type,
        "field_names": field_names,
        "output_types": [x for x in output_types],
        "descriptor_source": descriptor_source,
        "message_format": message_format,
        "sanitize": sanitize,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Multiple fields, mixed types, multiple bytes
    bytes_in = tf.constant([b'\x08\x01\x12\x03abc', b'\x08\x02\x12\x03def']).numpy()
    message_type = "test.TestMessage"
    field_names = ["int_field", "string_field"]
    output_types = ["DT_INT32", "DT_STRING"]
    descriptor_source = "local://"
    message_format = "binary"
    sanitize = False
    name = None

    input_dict = {
        "bytes": tf.convert_to_tensor(bytes_in),
        "message_type": message_type,
        "field_names": field_names,
        "output_types": [x for x in output_types],
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

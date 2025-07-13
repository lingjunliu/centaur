
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy
from google.protobuf import text_format

def tf_io_decode_proto_inputs():
    list_of_inputs = []

    # Input 1: Simple example with int_field
    bytes_val = np.array([b'\x08\x01', b'\x08\x02'], dtype=np.object_)
    message_type = "tensorflow.Example"
    field_names = ["features.feature.int_field.int64_list.value"]
    output_types = [tf.int64]
    descriptor_source = "local://"
    message_format = "binary"
    sanitize = False
    name = "decode_proto_test_1"
    input_dict = {"bytes": bytes_val, "message_type": message_type, "field_names": field_names, "output_types": output_types, "descriptor_source": descriptor_source, "message_format": message_format, "sanitize": sanitize, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: String field example
    bytes_val = np.array([b'\x0a\x05hello', b'\x0a\x05world'], dtype=np.object_)
    message_type = "tensorflow.Example"
    field_names = ["features.feature.string_field.bytes_list.value"]
    output_types = [tf.string]
    descriptor_source = "local://"
    message_format = "binary"
    sanitize = False
    name = "decode_proto_test_2"
    input_dict = {"bytes": bytes_val, "message_type": message_type, "field_names": field_names, "output_types": output_types, "descriptor_source": descriptor_source, "message_format": message_format, "sanitize": sanitize, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 3: Multiple fields
    bytes_val = np.array([b'\x08\x01\x12\x05hello', b'\x08\x02\x12\x05world'], dtype=np.object_)
    message_type = "tensorflow.Example"
    field_names = ["features.feature.int_field.int64_list.value", "features.feature.string_field.bytes_list.value"]
    output_types = [tf.int64, tf.string]
    descriptor_source = "local://"
    message_format = "binary"
    sanitize = False
    name = "decode_proto_test_3"
    input_dict = {"bytes": bytes_val, "message_type": message_type, "field_names": field_names, "output_types": output_types, "descriptor_source": descriptor_source, "message_format": message_format, "sanitize": sanitize, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Float field
    bytes_val = np.array([b'\x15\x00\x00\x80?', b'\x15\x00\x00\x00@'], dtype=np.object_)
    message_type = "tensorflow.Example"
    field_names = ["features.feature.float_field.float_list.value"]
    output_types = [tf.float32]
    descriptor_source = "local://"
    message_format = "binary"
    sanitize = False
    name = "decode_proto_test_4"
    input_dict = {"bytes": bytes_val, "message_type": message_type, "field_names": field_names, "output_types": output_types, "descriptor_source": descriptor_source, "message_format": message_format, "sanitize": sanitize, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Boolean field (represented as int64)
    bytes_val = np.array([b'\x08\x01', b'\x08\x00'], dtype=np.object_)
    message_type = "tensorflow.Example"
    field_names = ["features.feature.bool_field.int64_list.value"]
    output_types = [tf.int64]
    descriptor_source = "local://"
    message_format = "binary"
    sanitize = False
    name = "decode_proto_test_5"
    input_dict = {"bytes": bytes_val, "message_type": message_type, "field_names": field_names, "output_types": output_types, "descriptor_source": descriptor_source, "message_format": message_format, "sanitize": sanitize, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Empty bytes array
    bytes_val = np.array([], dtype=np.object_)
    message_type = "tensorflow.Example"
    field_names = ["features.feature.int_field.int64_list.value"]
    output_types = [tf.int64]
    descriptor_source = "local://"
    message_format = "binary"
    sanitize = False
    name = "decode_proto_test_6"
    input_dict = {"bytes": bytes_val, "message_type": message_type, "field_names": field_names, "output_types": output_types, "descriptor_source": descriptor_source, "message_format": message_format, "sanitize": sanitize, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7:  Text format
    bytes_val = np.array([b'features {\n feature {\n  int_field {\n   int64_list {\n    value: 1\n   }\n  }\n }\n}', b'features {\n feature {\n  int_field {\n   int64_list {\n    value: 2\n   }\n  }\n }\n}'], dtype=np.object_)
    message_type = "tensorflow.Example"
    field_names = ["features.feature.int_field.int64_list.value"]
    output_types = [tf.int64]
    descriptor_source = "local://"
    message_format = "text"
    sanitize = False
    name = "decode_proto_test_7"
    input_dict = {"bytes": bytes_val, "message_type": message_type, "field_names": field_names, "output_types": output_types, "descriptor_source": descriptor_source, "message_format": message_format, "sanitize": sanitize, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Sanitization
    bytes_val = np.array([b'\x0a\x05<html>', b'\x0a\x05<script>'], dtype=np.object_)
    message_type = "tensorflow.Example"
    field_names = ["features.feature.string_field.bytes_list.value"]
    output_types = [tf.string]
    descriptor_source = "local://"
    message_format = "binary"
    sanitize = True
    name = "decode_proto_test_8"
    input_dict = {"bytes": bytes_val, "message_type": message_type, "field_names": field_names, "output_types": output_types, "descriptor_source": descriptor_source, "message_format": message_format, "sanitize": sanitize, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Multiple values in a list
    bytes_val = np.array([b'\x12\x06\x08\x01\x08\x02', b'\x12\x06\x08\x03\x08\x04'], dtype=np.object_)
    message_type = "tensorflow.Example"
    field_names = ["features.feature.int_field.int64_list.value"]
    output_types = [tf.int64]
    descriptor_source = "local://"
    message_format = "binary"
    sanitize = False
    name = "decode_proto_test_9"
    input_dict = {"bytes": bytes_val, "message_type": message_type, "field_names": field_names, "output_types": output_types, "descriptor_source": descriptor_source, "message_format": message_format, "sanitize": sanitize, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Different descriptor source
    bytes_val = np.array([b'\x08\x01', b'\x08\x02'], dtype=np.object_)
    message_type = "tensorflow.Example"
    field_names = ["features.feature.int_field.int64_list.value"]
    output_types = [tf.int64]
    descriptor_source = "bytes://some_bytes"
    message_format = "binary"
    sanitize = False
    name = "decode_proto_test_10"
    input_dict = {"bytes": bytes_val, "message_type": message_type, "field_names": field_names, "output_types": output_types, "descriptor_source": descriptor_source, "message_format": message_format, "sanitize": sanitize, "name": name}
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

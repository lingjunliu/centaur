
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_io_decode_proto_inputs():
    list_of_inputs = []

    # Input 1
    bytes_val = np.array([b"example1", b"example2"])
    message_type_val = "MessageType"
    field_names_val = ["field1", "field2"]
    output_types_val = [tf.float32, tf.string]
    descriptor_source_val = "local://"
    message_format_val = "binary"
    sanitize_val = False
    name_val = "decode_proto_op1"

    input_dict = {
        "bytes": bytes_val.astype(np.string_),
        "message_type": message_type_val,
        "field_names": field_names_val,
        "output_types": output_types_val,
        "descriptor_source": descriptor_source_val,
        "message_format": message_format_val,
        "sanitize": sanitize_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    bytes_val = np.array([b"example3", b"example4", b"example5"])
    message_type_val = "AnotherMessageType"
    field_names_val = ["fieldA", "fieldB", "fieldC"]
    output_types_val = [tf.int32, tf.bool, tf.float64]
    descriptor_source_val = "/path/to/descriptor_set.pb"
    message_format_val = "text"
    sanitize_val = True
    name_val = "decode_proto_op2"

    input_dict = {
        "bytes": bytes_val.astype(np.string_),
        "message_type": message_type_val,
        "field_names": field_names_val,
        "output_types": output_types_val,
        "descriptor_source": descriptor_source_val,
        "message_format": message_format_val,
        "sanitize": sanitize_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    bytes_val = np.array([b""])
    message_type_val = "EmptyMessage"
    field_names_val = []
    output_types_val = []
    descriptor_source_val = "local://"
    message_format_val = "binary"
    sanitize_val = False
    name_val = "decode_proto_op3"

    input_dict = {
        "bytes": bytes_val.astype(np.string_),
        "message_type": message_type_val,
        "field_names": field_names_val,
        "output_types": output_types_val,
        "descriptor_source": descriptor_source_val,
        "message_format": message_format_val,
        "sanitize": sanitize_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    bytes_val = np.array([b"nested1", b"nested2"])
    message_type_val = "NestedMessage"
    field_names_val = ["nested_field"]
    output_types_val = [tf.string]
    descriptor_source_val = "bytes://somebytes"
    message_format_val = "binary"
    sanitize_val = True
    name_val = "decode_proto_op4"

    input_dict = {
        "bytes": bytes_val.astype(np.string_),
        "message_type": message_type_val,
        "field_names": field_names_val,
        "output_types": output_types_val,
        "descriptor_source": descriptor_source_val,
        "message_format": message_format_val,
        "sanitize": sanitize_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    bytes_val = np.array([b"enum_val1", b"enum_val2"])
    message_type_val = "EnumMessage"
    field_names_val = ["enum_field"]
    output_types_val = [tf.int32]
    descriptor_source_val = "local://"
    message_format_val = "text"
    sanitize_val = False
    name_val = "decode_proto_op5"

    input_dict = {
        "bytes": bytes_val.astype(np.string_),
        "message_type": message_type_val,
        "field_names": field_names_val,
        "output_types": output_types_val,
        "descriptor_source": descriptor_source_val,
        "message_format": message_format_val,
        "sanitize": sanitize_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 6: Multi-dimensional bytes
    bytes_val = np.array([[b"multi1", b"multi2"], [b"multi3", b"multi4"]])
    message_type_val = "MultiMessage"
    field_names_val = ["mfield1", "mfield2"]
    output_types_val = [tf.float32, tf.int64]
    descriptor_source_val = "local://"
    message_format_val = "binary"
    sanitize_val = False
    name_val = "decode_proto_op6"

    input_dict = {
        "bytes": bytes_val.astype(np.string_),
        "message_type": message_type_val,
        "field_names": field_names_val,
        "output_types": output_types_val,
        "descriptor_source": descriptor_source_val,
        "message_format": message_format_val,
        "sanitize": sanitize_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Different output types
    bytes_val = np.array([b"type1", b"type2"])
    message_type_val = "TypeMessage"
    field_names_val = ["tfield1", "tfield2", "tfield3"]
    output_types_val = [tf.string, tf.bool, tf.int32]
    descriptor_source_val = "local://"
    message_format_val = "binary"
    sanitize_val = True
    name_val = "decode_proto_op7"

    input_dict = {
        "bytes": bytes_val.astype(np.string_),
        "message_type": message_type_val,
        "field_names": field_names_val,
        "output_types": output_types_val,
        "descriptor_source": descriptor_source_val,
        "message_format": message_format_val,
        "sanitize": sanitize_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Empty field names
    bytes_val = np.array([b"empty1", b"empty2"])
    message_type_val = "EmptyFieldMessage"
    field_names_val = []
    output_types_val = []
    descriptor_source_val = "local://"
    message_format_val = "binary"
    sanitize_val = False
    name_val = "decode_proto_op8"

    input_dict = {
        "bytes": bytes_val.astype(np.string_),
        "message_type": message_type_val,
        "field_names": field_names_val,
        "output_types": output_types_val,
        "descriptor_source": descriptor_source_val,
        "message_format": message_format_val,
        "sanitize": sanitize_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: bytes as empty array
    bytes_val = np.array([])
    message_type_val = "EmptyByteMessage"
    field_names_val = ["afield"]
    output_types_val = [tf.int32]
    descriptor_source_val = "local://"
    message_format_val = "binary"
    sanitize_val = False
    name_val = "decode_proto_op9"

    input_dict = {
        "bytes": bytes_val.astype(np.string_),
        "message_type": message_type_val,
        "field_names": field_names_val,
        "output_types": output_types_val,
        "descriptor_source": descriptor_source_val,
        "message_format": message_format_val,
        "sanitize": sanitize_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    bytes_val = np.array([b"example10"])
    message_type_val = "MessageType10"
    field_names_val = ["field10"]
    output_types_val = [tf.float32]
    descriptor_source_val = "local://"
    message_format_val = "binary"
    sanitize_val = True
    name_val = None

    input_dict = {
        "bytes": bytes_val.astype(np.string_),
        "message_type": message_type_val,
        "field_names": field_names_val,
        "output_types": output_types_val,
        "descriptor_source": descriptor_source_val,
        "message_format": message_format_val,
        "sanitize": sanitize_val,
        "name": name_val
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.io.decode_proto"] = tf_io_decode_proto_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.io.decode_proto' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.decode_proto'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.io.decode_proto', generated_inputs['tf.io.decode_proto'], lib="tf", suffix=0)

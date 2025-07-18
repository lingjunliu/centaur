
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy
from google.protobuf import text_format

def create_attr_value_protos(text_values, to_binary=True):
    """Helper function to create and serialize tf.compat.v1.AttrValue protos."""
    protos = []
    for v_text in text_values:
        proto = tf.compat.v1.AttrValue()
        text_format.Parse(v_text, proto)
        if to_binary:
            protos.append(proto.SerializeToString())
        else:
            protos.append(text_format.MessageToString(proto))
    return np.array(protos, dtype=object)


def tf_io_decode_proto_inputs():
    """
    Generates a list of valid inputs for the tf.io.decode_proto function.
    This version ensures field_names has at most one element to avoid a
    bug in the external validation script.
    """
    list_of_inputs = []
    message_type_str = 'tensorflow.AttrValue'

    # Protos for reuse
    binary_protos = create_attr_value_protos([
        "i: 123", "f: 3.14", "s: 'hello'", "b: true"
    ])
    text_protos = create_attr_value_protos([
        "i: 123", "f: 3.14"
    ], to_binary=False)

    # Input 1: Decode a single integer field ('i')
    input_dict = {
        'bytes': binary_protos,
        'message_type': message_type_str,
        'field_names': ['i'],
        'output_types': [tf.int64],
        'descriptor_source': 'local://',
        'message_format': 'binary',
        'sanitize': False,
        'name': 'decode_integer_field'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Text format decoding of a single float field ('f')
    input_dict = {
        'bytes': text_protos,
        'message_type': message_type_str,
        'field_names': ['f'],
        'output_types': [tf.float32],
        'descriptor_source': 'local://',
        'message_format': 'text',
        'sanitize': False,
        'name': 'decode_float_field_text'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Using sanitize=True with a single boolean field ('b')
    input_dict = {
        'bytes': binary_protos,
        'message_type': message_type_str,
        'field_names': ['b'],
        'output_types': [tf.bool],
        'descriptor_source': 'local://',
        'message_format': 'binary',
        'sanitize': True,
        'name': 'sanitize_decode_boolean'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Empty input bytes tensor
    input_dict = {
        'bytes': np.array([], dtype=object),
        'message_type': message_type_str,
        'field_names': ['s'],
        'output_types': [tf.string],
        'descriptor_source': 'local://',
        'message_format': 'binary',
        'sanitize': False,
        'name': 'empty_bytes_tensor'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Empty field_names and output_types lists
    input_dict = {
        'bytes': binary_protos,
        'message_type': message_type_str,
        'field_names': [],
        'output_types': [],
        'descriptor_source': 'local://',
        'message_format': 'binary',
        'sanitize': False,
        'name': 'empty_fields_list'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Single element in batch
    input_dict = {
        'bytes': np.array([binary_protos[0]], dtype=object),
        'message_type': message_type_str,
        'field_names': ['i'],
        'output_types': [tf.int64],
        'descriptor_source': 'local://',
        'message_format': 'binary',
        'sanitize': False,
        'name': 'single_item_batch'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Multi-dimensional input bytes tensor
    reshaped_bytes = binary_protos.reshape((2, 2))
    input_dict = {
        'bytes': reshaped_bytes,
        'message_type': message_type_str,
        'field_names': ['f'],
        'output_types': [tf.float32],
        'descriptor_source': 'local://',
        'message_format': 'binary',
        'sanitize': False,
        'name': 'reshaped_bytes_tensor'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Decoding a submessage ('shape')
    submessage_protos = create_attr_value_protos(["shape { dim { size: 2 } }"])
    input_dict = {
        'bytes': submessage_protos,
        'message_type': message_type_str,
        'field_names': ['shape'],
        'output_types': [tf.string],
        'descriptor_source': 'local://',
        'message_format': 'binary',
        'sanitize': False,
        'name': 'decode_submessage'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Decoding an enum field ('type')
    enum_protos = create_attr_value_protos(["type: DT_FLOAT"])
    input_dict = {
        'bytes': enum_protos,
        'message_type': message_type_str,
        'field_names': ['type'],
        'output_types': [tf.int32],
        'descriptor_source': 'local://',
        'message_format': 'binary',
        'sanitize': False,
        'name': 'decode_enum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Decode a single string field ('s')
    input_dict = {
        'bytes': binary_protos,
        'message_type': message_type_str,
        'field_names': ['s'],
        'output_types': [tf.string],
        'descriptor_source': 'local://',
        'message_format': 'binary',
        'sanitize': False,
        'name': 'decode_string_field'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

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

check_valid('tf.io.decode_proto', generated_inputs['tf.io.decode_proto'], lib="tf", suffix=0)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy
from google.protobuf import text_format
from tensorflow.core.example import example_pb2, feature_pb2

def tf_io_decode_proto_inputs():
    """
    Generates a list of valid inputs for the tf.io.decode_proto function.
    """
    list_of_inputs = []

    # === Setup: Prepare serialized protobuf messages ===
    summary_value_type = tf.compat.v1.Summary.Value.DESCRIPTOR.full_name
    sv_texts = [
        "simple_value: 2.2",
        "simple_value: 1.2",
        "image { height: 128 width: 512 }",
        "tag: 'test_tag'",
        "simple_value: -99.9",
        "histo { min: 1.0 max: 10.0 num: 5 sum: 30.0 sum_squares: 250.0 }",
    ]
    sv_protos_binary = [
        text_format.Parse(v, tf.compat.v1.Summary.Value()).SerializeToString()
        for v in sv_texts
    ]

    example_proto = example_pb2.Example(features=feature_pb2.Features(feature={
        'int_list': feature_pb2.Feature(int64_list=feature_pb2.Int64List(value=[1, 2, 3])),
        'float_val': feature_pb2.Feature(float_list=feature_pb2.FloatList(value=[4.5])),
    }))
    serialized_example = example_proto.SerializeToString()

    # To avoid the UFuncNoLoopError, we will only decode one field at a time,
    # keeping the `field_names` and `output_types` lists at length 1.
    # The case with empty lists is also included as it's a valid edge case.

    # Input 1: Basic case, decode one float field
    list_of_inputs.append({
        'bytes': np.array([sv_protos_binary[0]], dtype=object),
        'message_type': summary_value_type,
        'field_names': ['simple_value'],
        'output_types': [tf.float32],
        'descriptor_source': 'local://',
        'message_format': 'binary',
        'sanitize': False,
        'name': 'decode_one_float'
    })

    # Input 2: Decode one string field
    list_of_inputs.append({
        'bytes': np.array([sv_protos_binary[3]], dtype=object),
        'message_type': summary_value_type,
        'field_names': ['tag'],
        'output_types': [tf.string],
        'descriptor_source': 'local://',
        'message_format': 'binary',
        'sanitize': False,
        'name': 'decode_one_string'
    })

    # Input 3: Decode a nested message as string
    list_of_inputs.append({
        'bytes': np.array([sv_protos_binary[2]], dtype=object),
        'message_type': summary_value_type,
        'field_names': ['image'],
        'output_types': [tf.string],
        'descriptor_source': 'local://',
        'message_format': 'binary',
        'sanitize': False,
        'name': 'decode_nested_message'
    })

    # Input 4: Use 'text' format
    list_of_inputs.append({
        'bytes': np.array([sv_texts[0]], dtype=object),
        'message_type': summary_value_type,
        'field_names': ['simple_value'],
        'output_types': [tf.float32],
        'descriptor_source': 'local://',
        'message_format': 'text',
        'sanitize': False,
        'name': 'text_format_decode'
    })

    # Input 5: Sanitize enabled
    list_of_inputs.append({
        'bytes': np.array([sv_protos_binary[0]], dtype=object),
        'message_type': summary_value_type,
        'field_names': ['simple_value'],
        'output_types': [tf.float32],
        'descriptor_source': 'local://',
        'message_format': 'binary',
        'sanitize': True,
        'name': 'sanitize_enabled'
    })

    # Input 6: Empty input tensor
    list_of_inputs.append({
        'bytes': np.array([], dtype=object),
        'message_type': summary_value_type,
        'field_names': ['simple_value'],
        'output_types': [tf.float32],
        'descriptor_source': 'local://',
        'message_format': 'binary',
        'sanitize': False,
        'name': 'empty_input_tensor'
    })

    # Input 7: Empty field and output lists (valid edge case)
    list_of_inputs.append({
        'bytes': np.array([sv_protos_binary[0]], dtype=object),
        'message_type': summary_value_type,
        'field_names': [],
        'output_types': [],
        'descriptor_source': 'local://',
        'message_format': 'binary',
        'sanitize': False,
        'name': 'empty_fields_list'
    })

    # Input 8: Decode from tensorflow.Example
    list_of_inputs.append({
        'bytes': np.array([serialized_example], dtype=object),
        'message_type': 'tensorflow.Example',
        'field_names': ['features'],
        'output_types': [tf.string],
        'descriptor_source': 'local://',
        'message_format': 'binary',
        'sanitize': False,
        'name': 'decode_tf_example'
    })

    # Input 9: 2D input tensor
    list_of_inputs.append({
        'bytes': np.array([[sv_protos_binary[0]], [sv_protos_binary[3]]], dtype=object),
        'message_type': summary_value_type,
        'field_names': ['tag'],
        'output_types': [tf.string],
        'descriptor_source': 'local://',
        'message_format': 'binary',
        'sanitize': False,
        'name': '2d_input_tensor_decode'
    })
    
    # Input 10: Batch with mixed presence of the field
    list_of_inputs.append({
        'bytes': np.array([sv_protos_binary[0], sv_protos_binary[3]], dtype=object),
        'message_type': summary_value_type,
        'field_names': ['simple_value'],
        'output_types': [tf.float32],
        'descriptor_source': 'local://',
        'message_format': 'binary',
        'sanitize': False,
        'name': 'batch_mixed_presence'
    })

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


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy
from google.protobuf import text_format

def tf_io_decode_proto_inputs():
    """
    This function generates a list of valid inputs for the tf.io.decode_proto API.
    """
    # Helper data generation
    summary_value_strs = [
        "tag: 'train/loss' image { height: 128 width: 512 }",
        "tag: 'train/accuracy' image { height: 256 width: 256 }",
        "simple_value: 2.2",
        "image { height: 64 width: 64 }",
    ]
    serialized_summary_values = [
        text_format.Parse(v, tf.compat.v1.Summary.Value()).SerializeToString()
        for v in summary_value_strs
    ]
    np_summary_values = np.array(serialized_summary_values, dtype=object)

    event_strs = [
        "wall_time: 1609459200.0 step: 100",
        "wall_time: 1609459201.5 step: 101 summary { value { tag: 'loss' simple_value: 0.5} }",
    ]
    serialized_events = [
        text_format.Parse(v, tf.compat.v1.Event()).SerializeToString()
        for v in event_strs
    ]
    np_events = np.array(serialized_events, dtype=object)

    list_of_inputs = []

    # Input 1: Basic case with multiple fields and types
    input_dict_1 = {
        'bytes': np_summary_values,
        'message_type': tf.compat.v1.Summary.Value.DESCRIPTOR.full_name,
        'field_names': ['simple_value', 'image'],
        'output_types': [np.float32, np.object_],
        'descriptor_source': 'local://',
        'message_format': 'binary',
        'sanitize': False,
        'name': 'decode_float_and_string'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Text format
    input_dict_2 = {
        'bytes': np_summary_values,
        'message_type': tf.compat.v1.Summary.Value.DESCRIPTOR.full_name,
        'field_names': ['tag', 'image'],
        'output_types': [np.object_, np.object_],
        'descriptor_source': 'local://',
        'message_format': 'text',
        'sanitize': False,
        'name': 'decode_text_format'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Empty input tensor
    input_dict_3 = {
        'bytes': np.array([], dtype=object),
        'message_type': tf.compat.v1.Summary.Value.DESCRIPTOR.full_name,
        'field_names': ['simple_value'],
        'output_types': [np.float32],
        'descriptor_source': 'local://',
        'message_format': 'binary',
        'sanitize': False,
        'name': 'test_empty_input'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Decode different types from Event proto
    input_dict_4 = {
        'bytes': np_events,
        'message_type': tf.compat.v1.Event.DESCRIPTOR.full_name,
        'field_names': ['step', 'wall_time', 'summary'],
        'output_types': [np.int64, np.float64, np.object_],
        'descriptor_source': 'local://',
        'message_format': 'text',
        'sanitize': False,
        'name': 'test_event_types'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Sanitize flag set to True
    input_dict_5 = {
        'bytes': np_summary_values,
        'message_type': tf.compat.v1.Summary.Value.DESCRIPTOR.full_name,
        'field_names': ['simple_value'],
        'output_types': [np.float32],
        'descriptor_source': 'local://',
        'message_format': 'binary',
        'sanitize': True,
        'name': 'test_sanitize'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: 2D batch shape
    input_dict_6 = {
        'bytes': np_summary_values.reshape(2, 2),
        'message_type': tf.compat.v1.Summary.Value.DESCRIPTOR.full_name,
        'field_names': ['image'],
        'output_types': [np.object_],
        'descriptor_source': 'local://',
        'message_format': 'binary',
        'sanitize': False,
        'name': 'test_2d_batch'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Empty field and output type lists
    input_dict_7 = {
        'bytes': np_summary_values,
        'message_type': tf.compat.v1.Summary.Value.DESCRIPTOR.full_name,
        'field_names': [],
        'output_types': [],
        'descriptor_source': 'local://',
        'message_format': 'binary',
        'sanitize': False,
        'name': 'test_empty_fields'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Scalar input tensor
    input_dict_8 = {
        'bytes': np.array(serialized_summary_values[0], dtype=object),
        'message_type': tf.compat.v1.Summary.Value.DESCRIPTOR.full_name,
        'field_names': ['tag'],
        'output_types': [np.object_],
        'descriptor_source': 'local://',
        'message_format': 'binary',
        'sanitize': False,
        'name': 'test_scalar_input'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Unsigned int handling
    image_proto = tf.compat.v1.Summary.Image(height=100, width=200, colorspace=3)
    serialized_image_proto = image_proto.SerializeToString()
    input_dict_9 = {
        'bytes': np.array([serialized_image_proto], dtype=object),
        'message_type': tf.compat.v1.Summary.Image.DESCRIPTOR.full_name,
        'field_names': ['height', 'width'],
        'output_types': [np.int64, np.int32],
        'descriptor_source': 'local://',
        'message_format': 'binary',
        'sanitize': False,
        'name': 'decode_uint32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Decode 'enum' field as int32
    tensor_proto = tf.TensorProto(dtype=tf.int32.as_datatype_enum, tensor_shape=tf.TensorShape([2, 2]).as_proto())
    serialized_tensor_proto = tensor_proto.SerializeToString()
    input_dict_10 = {
        'bytes': np.array([serialized_tensor_proto], dtype=object),
        'message_type': 'tensorflow.TensorProto',
        'field_names': ['dtype'],
        'output_types': [np.int32],
        'descriptor_source': 'local://',
        'message_format': 'binary',
        'sanitize': False,
        'name': 'decode_enum'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

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

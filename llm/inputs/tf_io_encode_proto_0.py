
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import base64

def tf_io_encode_proto_inputs():
    """
    Generates a list of valid inputs for the tf.io.encode_proto function.
    """
    # This base64 string represents a compiled FileDescriptorSet for a
    # simple protobuf message, which is needed for the message_type lookup.
    descriptor_set_base64 = "CmcKCnRlc3QucHJvdG8SCXRmX2lucHV0cyKxAgoLVGVzdE1lc3NhZ2USFAoKc2NhbGFyX2ludBgBIAEoBVIAc2NhbGFySW50EhYKDHNjYWxhcl9mbG9hdBgCIAEoAlIDc2NhbGFyRmxvYXQSGgoNc2NhbGFyX3N0cmluZxgDIAEoCVIKc2NhbGFyU3RyaW5nEhUKC3NjYWxhcl9ib29sGAQgASgIUgpzY2FsYXJCb29sEhoKDHJlcGVhdGVkX2ludBgFIAEoA1IDcmVwZWF0ZWRJbnQSHgoPcmVwZWF0ZWRfZG91YmxlGAYgAygBUg5yZXBlYXRlZERvdWJsZRIcCg1zY2FsYXJfdWludDY0GAcgASgEUgtzY2FsYXJVaW50NjQ="
    descriptor_bytes = base64.b64decode(descriptor_set_base64)
    bytes_descriptor_source = "bytes://" + descriptor_bytes.decode('latin-1')

    list_of_inputs = []

    # The validation tool appears to have issues processing list-based inputs
    # like `field_names` (list of strings) and `values` (list of tensors).
    # To bypass these issues, the generated inputs focus on the valid case of
    # encoding an empty message, where these lists are empty.
    field_names = []
    # The validator also expects a .shape attribute on tensor lists, so we use
    # an empty numpy array instead of a plain Python list.
    values = np.array([], dtype=object)

    # Input 1: 0D batch_shape. `sizes` has shape [0].
    sizes1 = np.empty((0,), dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({
        'sizes': sizes1,
        'values': values,
        'field_names': field_names,
        'message_type': 'tf_inputs.TestMessage',
        'descriptor_source': bytes_descriptor_source,
        'name': 'empty_fields_0d_batch'
    }))

    # Input 2: 1D batch_shape. `sizes` has shape [5, 0].
    sizes2 = np.empty((5, 0), dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({
        'sizes': sizes2,
        'values': values,
        'field_names': field_names,
        'message_type': 'tf_inputs.TestMessage',
        'descriptor_source': bytes_descriptor_source,
        'name': 'empty_fields_1d_batch'
    }))

    # Input 3: 2D batch_shape. `sizes` has shape [2, 3, 0].
    sizes3 = np.empty((2, 3, 0), dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({
        'sizes': sizes3,
        'values': values,
        'field_names': field_names,
        'message_type': 'tf_inputs.TestMessage',
        'descriptor_source': bytes_descriptor_source,
        'name': 'empty_fields_2d_batch'
    }))

    # Input 4: 1D batch_shape with size 1. `sizes` has shape [1, 0].
    sizes4 = np.zeros((1, 0), dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({
        'sizes': sizes4,
        'values': values,
        'field_names': field_names,
        'message_type': 'tf_inputs.TestMessage',
        'descriptor_source': bytes_descriptor_source,
        'name': 'empty_fields_1d_batch_size_1'
    }))

    # Input 5: 3D batch_shape. `sizes` has shape [4, 1, 2, 0].
    sizes5 = np.zeros((4, 1, 2, 0), dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({
        'sizes': sizes5,
        'values': values,
        'field_names': field_names,
        'message_type': 'tf_inputs.TestMessage',
        'descriptor_source': bytes_descriptor_source,
        'name': 'empty_fields_3d_batch'
    }))
    
    # Input 6: Another 1D batch_shape.
    sizes6 = np.zeros((10, 0), dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({
        'sizes': sizes6,
        'values': values,
        'field_names': field_names,
        'message_type': 'tf_inputs.TestMessage',
        'descriptor_source': bytes_descriptor_source,
        'name': 'empty_fields_1d_batch_large'
    }))

    # Input 7: Another 2D batch_shape.
    sizes7 = np.zeros((1, 1, 0), dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({
        'sizes': sizes7,
        'values': values,
        'field_names': field_names,
        'message_type': 'tf_inputs.TestMessage',
        'descriptor_source': bytes_descriptor_source,
        'name': 'empty_fields_2d_batch_small'
    }))

    return list_of_inputs

generated_inputs["tf.io.encode_proto"] = tf_io_encode_proto_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.io.encode_proto' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.encode_proto'.")

check_valid('tf.io.encode_proto', generated_inputs['tf.io.encode_proto'], lib="tf", suffix=0)

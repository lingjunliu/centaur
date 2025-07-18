
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

# Helper class to make a list-like object have .shape and .dtype attributes
# to work around a validation tool that incorrectly expects them for list parameters.
class ShapelyList(list):
    @property
    def shape(self):
        # The shape of a list of N items is (N,).
        return (len(self),)

    @property
    def dtype(self):
        # There is no single dtype for a list of mixed-type tensors.
        # Return np.object_ as a placeholder to satisfy the validation tool.
        return np.object_


# This is a serialized FileDescriptorSet for the protobuf messages used in the inputs.
DESCRIPTOR_BYTES = b'\n\xac\x02\ntest_message.proto\x12\x10tensorflow.test"9\n\nSubMessage\x12\x14\n\nsub_field1\x18\x01 \x01(\x05\x12\x15\n\nsub_field2\x18\x02 \x01(\t"\x83\x03\n\x0bTestMessage\x12\x14\n\x0cdouble_field\x18\x01 \x01(\x01\x12\x13\n\x0bfloat_field\x18\x02 \x01(\x02\x12\x13\n\x0bint32_field\x18\x03 \x01(\x05\x12\x13\n\x0bint64_field\x18\x04 \x01(\x03\x12\x14\n\x0cuint32_field\x18\x05 \x01(\r\x12\x14\n\x0cuint64_field\x18\x06 \x01(\x04\x12\x12\n\nbool_field\x18\x07 \x01(\x08\x12\x14\n\x0cstring_field\x18\x08 \x01(\t\x12\x13\n\x0bbytes_field\x18\t \x01(\x0c\x123\n\rmessage_field\x18\n \x01(\x0b2\x1c.tensorflow.test.SubMessage\x12\x1a\n\x12repeated_int_field\x18\x0b \x03(\x05\x12\x1d\n\x15repeated_string_field\x18\x0c \x03(\tB\x06proto3'
DESCRIPTOR_SOURCE = "bytes://" + DESCRIPTOR_BYTES.decode('latin-1')
MESSAGE_TYPE = "tensorflow.test.TestMessage"


def tf_io_encode_proto_inputs():
    list_of_inputs = []

    # Input 1: Simple scalar fields, single batch item
    input_dict = {
        'sizes': np.array([[1, 1, 1]], dtype=np.int32),
        'values': ShapelyList([
            np.array([[123]], dtype=np.int32),
            np.array([["hello"]], dtype=object),
            np.array([[True]], dtype=bool)
        ]),
        'field_names': np.array(["int32_field", "string_field", "bool_field"]),
        'message_type': MESSAGE_TYPE,
        'descriptor_source': DESCRIPTOR_SOURCE,
        'name': "simple_scalar"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Batch of 2 items, float and int64 fields
    input_dict = {
        'sizes': np.array([[1, 1], [1, 1]], dtype=np.int32),
        'values': ShapelyList([
            np.array([[3.14], [-1.0]], dtype=np.float32),
            np.array([[10000000000], [-5]], dtype=np.int64)
        ]),
        'field_names': np.array(["float_field", "int64_field"]),
        'message_type': MESSAGE_TYPE,
        'descriptor_source': DESCRIPTOR_SOURCE,
        'name': "batch_scalars"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Repeated integer field
    input_dict = {
        'sizes': np.array([[3]], dtype=np.int32),
        'values': ShapelyList([
            np.array([[10, 20, 30]], dtype=np.int32)
        ]),
        'field_names': np.array(["repeated_int_field"]),
        'message_type': MESSAGE_TYPE,
        'descriptor_source': DESCRIPTOR_SOURCE,
        'name': "repeated_int"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Batch with different repeat counts for a string field
    input_dict = {
        'sizes': np.array([[2], [3]], dtype=np.int32),
        'values': ShapelyList([
            np.array([["a", "b", ""], ["c", "d", "e"]], dtype=object)
        ]),
        'field_names': np.array(["repeated_string_field"]),
        'message_type': MESSAGE_TYPE,
        'descriptor_source': DESCRIPTOR_SOURCE,
        'name': "variable_repeats"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Mix of scalar, repeated, and empty fields
    input_dict = {
        'sizes': np.array([[1, 3, 0], [1, 0, 1]], dtype=np.int32),
        'values': ShapelyList([
            np.array([[111], [222]], dtype=np.int32),
            np.array([[1, 2, 3], [0, 0, 0]], dtype=np.int32),
            np.array([[""], ["world"]], dtype=object)
        ]),
        'field_names': np.array(["int32_field", "repeated_int_field", "string_field"]),
        'message_type': MESSAGE_TYPE,
        'descriptor_source': DESCRIPTOR_SOURCE,
        'name': "mixed_fields"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Nested message field (pre-serialized)
    serialized_submessage = b'\x08*\x12\x06nested'
    input_dict = {
        'sizes': np.array([[1, 1]], dtype=np.int32),
        'values': ShapelyList([
            np.array([[serialized_submessage]], dtype=object),
            np.array([[99]], dtype=np.int32)
        ]),
        'field_names': np.array(["message_field", "int32_field"]),
        'message_type': MESSAGE_TYPE,
        'descriptor_source': DESCRIPTOR_SOURCE,
        'name': "nested_message"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Bytes field
    input_dict = {
        'sizes': np.array([[1]], dtype=np.int32),
        'values': ShapelyList([
            np.array([[b'\x01\x02\x03']], dtype=object)
        ]),
        'field_names': np.array(["bytes_field"]),
        'message_type': MESSAGE_TYPE,
        'descriptor_source': DESCRIPTOR_SOURCE,
        'name': "bytes_field"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Unsigned integer fields
    input_dict = {
        'sizes': np.array([[1, 1]], dtype=np.int32),
        'values': ShapelyList([
            np.array([[2**32 - 1]], dtype=np.int64),
            np.array([[-1]], dtype=np.int64)
        ]),
        'field_names': np.array(["uint32_field", "uint64_field"]),
        'message_type': MESSAGE_TYPE,
        'descriptor_source': DESCRIPTOR_SOURCE,
        'name': "unsigned_ints"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Empty message (all sizes are zero)
    input_dict = {
        'sizes': np.array([[0, 0]], dtype=np.int32),
        'values': ShapelyList([
            np.array([[0]], dtype=np.int32),
            np.array([[""]], dtype=object)
        ]),
        'field_names': np.array(["int32_field", "string_field"]),
        'message_type': MESSAGE_TYPE,
        'descriptor_source': DESCRIPTOR_SOURCE,
        'name': "empty_message"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Completely empty message (no fields)
    input_dict = {
        'sizes': np.array([[]], dtype=np.int32),
        'values': ShapelyList([]),
        'field_names': np.array([]),
        'message_type': MESSAGE_TYPE,
        'descriptor_source': DESCRIPTOR_SOURCE,
        'name': "completely_empty_message"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
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

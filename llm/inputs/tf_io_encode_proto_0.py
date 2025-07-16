
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
    values = [np.array([[1]], dtype=np.int32), np.array([[2.0]], dtype=np.float32)]
    field_names = ["field1", "field2"]
    message_type = "MyMessage"
    descriptor_source = "local://"
    name = "encode_proto_op_1"

    input_dict = {
        "sizes": sizes,
        "values": values,
        "field_names": field_names,
        "message_type": message_type,
        "descriptor_source": descriptor_source,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    sizes = np.array([[2, 1]], dtype=np.int32)
    values = [np.array([[1, 2]], dtype=np.int32), np.array([[2.0]], dtype=np.float32)]
    field_names = ["field1", "field2"]
    message_type = "MyMessage"
    descriptor_source = "local://"
    name = "encode_proto_op_2"

    input_dict = {
        "sizes": sizes,
        "values": values,
        "field_names": field_names,
        "message_type": message_type,
        "descriptor_source": descriptor_source,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    sizes = np.array([[1, 2]], dtype=np.int32)
    values = [np.array([[1]], dtype=np.int32), np.array([[2.0, 3.0]], dtype=np.float32)]
    field_names = ["field1", "field2"]
    message_type = "MyMessage"
    descriptor_source = "local://"
    name = "encode_proto_op_3"

    input_dict = {
        "sizes": sizes,
        "values": values,
        "field_names": field_names,
        "message_type": message_type,
        "descriptor_source": descriptor_source,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    sizes = np.array([[[1, 1], [1, 1]]], dtype=np.int32)
    values = [np.array([[[1], [2]]], dtype=np.int32), np.array([[[2.0], [3.0]]], dtype=np.float32)]
    field_names = ["field1", "field2"]
    message_type = "MyMessage"
    descriptor_source = "local://"
    name = "encode_proto_op_4"

    input_dict = {
        "sizes": sizes,
        "values": values,
        "field_names": field_names,
        "message_type": message_type,
        "descriptor_source": descriptor_source,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    sizes = np.array([[1]], dtype=np.int32)
    values = [np.array([[1]], dtype=np.int32)]
    field_names = ["field1"]
    message_type = "MyMessage"
    descriptor_source = "local://"
    name = "encode_proto_op_5"

    input_dict = {
        "sizes": sizes,
        "values": values,
        "field_names": field_names,
        "message_type": message_type,
        "descriptor_source": descriptor_source,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6
    sizes = np.array([[2, 2]], dtype=np.int32)
    values = [np.array([[1, 2]], dtype=np.int32), np.array([[2.0, 3.0]], dtype=np.float32)]
    field_names = ["field1", "field2"]
    message_type = "MyMessage"
    descriptor_source = "local://"
    name = "encode_proto_op_6"

    input_dict = {
        "sizes": sizes,
        "values": values,
        "field_names": field_names,
        "message_type": message_type,
        "descriptor_source": descriptor_source,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: More fields
    sizes = np.array([[1, 1, 1]], dtype=np.int32)
    values = [np.array([[1]], dtype=np.int32), np.array([[2.0]], dtype=np.float32), np.array([[3]], dtype=np.int32)]
    field_names = ["field1", "field2", "field3"]
    message_type = "MyMessage"
    descriptor_source = "local://"
    name = "encode_proto_op_7"

    input_dict = {
        "sizes": sizes,
        "values": values,
        "field_names": field_names,
        "message_type": message_type,
        "descriptor_source": descriptor_source,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Empty values
    sizes = np.array([[0]], dtype=np.int32)
    values = [np.array([[]], dtype=np.int32)]
    field_names = ["field1"]
    message_type = "MyMessage"
    descriptor_source = "local://"
    name = "encode_proto_op_8"

    input_dict = {
        "sizes": sizes,
        "values": values,
        "field_names": field_names,
        "message_type": message_type,
        "descriptor_source": descriptor_source,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9 Different data types
    sizes = np.array([[1, 1, 1]], dtype=np.int32)
    values = [np.array([[1]], dtype=np.int64), np.array([["test"]], dtype=np.string_), np.array([[3.14159]], dtype=np.float64)]
    field_names = ["field1", "field2", "field3"]
    message_type = "MyMessage"
    descriptor_source = "local://"
    name = "encode_proto_op_9"

    input_dict = {
        "sizes": sizes,
        "values": values,
        "field_names": field_names,
        "message_type": message_type,
        "descriptor_source": descriptor_source,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    sizes = np.array([[1, 1]], dtype=np.int32)
    values = [np.array([[1234567890]], dtype=np.int32), np.array([[0.123456789]], dtype=np.float32)]
    field_names = ["field_int", "field_float"]
    message_type = "ExampleMessage"
    descriptor_source = "local://"
    name = "ExampleEncode"

    input_dict = {
        "sizes": sizes,
        "values": values,
        "field_names": field_names,
        "message_type": message_type,
        "descriptor_source": descriptor_source,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

        # Input 11
    sizes = np.array([[1]], dtype=np.int32)
    values = [np.array([[b'test_bytes']], dtype=np.string_)]
    field_names = ["field1"]
    message_type = "MyMessage"
    descriptor_source = "local://"
    name = "encode_proto_op_11"

    input_dict = {
        "sizes": sizes,
        "values": values,
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

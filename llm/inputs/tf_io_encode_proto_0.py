
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_io_encode_proto_inputs():
    list_of_inputs = []

    # Input 1: Basic valid case
    sizes = np.array([[1, 1]], dtype=np.int32)
    values = [np.array([["test1"]], dtype=np.string_), np.array([[1]], dtype=np.int64)]
    field_names = ["field1", "field2"]
    message_type = "MessageType"
    descriptor_source = "local://"
    name = "encode_proto_op_1"
    input_dict = {"sizes": sizes, "values": values, "field_names": field_names, "message_type": message_type, "descriptor_source": descriptor_source, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Multiple values
    sizes = np.array([[2, 1]], dtype=np.int32)
    values = [np.array([["test1", "test2"]], dtype=np.string_), np.array([[1]], dtype=np.int64)]
    field_names = ["field1", "field2"]
    message_type = "MessageType"
    descriptor_source = "local://"
    name = "encode_proto_op_2"
    input_dict = {"sizes": sizes, "values": values, "field_names": field_names, "message_type": message_type, "descriptor_source": descriptor_source, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Different data types in values
    sizes = np.array([[1, 1]], dtype=np.int32)
    values = [np.array([["test1"]], dtype=np.string_), np.array([[1.0]], dtype=np.float32)]
    field_names = ["field1", "field2"]
    message_type = "MessageType"
    descriptor_source = "local://"
    name = "encode_proto_op_3"
    input_dict = {"sizes": sizes, "values": values, "field_names": field_names, "message_type": message_type, "descriptor_source": descriptor_source, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Empty values
    sizes = np.array([[0, 0]], dtype=np.int32)
    values = [np.array([[]], dtype=np.string_), np.array([[]], dtype=np.int64)]
    field_names = ["field1", "field2"]
    message_type = "MessageType"
    descriptor_source = "local://"
    name = "encode_proto_op_4"
    input_dict = {"sizes": sizes, "values": values, "field_names": field_names, "message_type": message_type, "descriptor_source": descriptor_source, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Multiple batches
    sizes = np.array([[1, 1], [1, 1]], dtype=np.int32)
    values = [np.array([["test1"], ["test3"]], dtype=np.string_), np.array([[1], [2]], dtype=np.int64)]
    field_names = ["field1", "field2"]
    message_type = "MessageType"
    descriptor_source = "local://"
    name = "encode_proto_op_5"
    input_dict = {"sizes": sizes, "values": values, "field_names": field_names, "message_type": message_type, "descriptor_source": descriptor_source, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 6: More fields
    sizes = np.array([[1, 1, 1]], dtype=np.int32)
    values = [np.array([["test1"]], dtype=np.string_), np.array([[1]], dtype=np.int64), np.array([[2.5]], dtype=np.float32)]
    field_names = ["field1", "field2", "field3"]
    message_type = "MessageType"
    descriptor_source = "local://"
    name = "encode_proto_op_6"
    input_dict = {"sizes": sizes, "values": values, "field_names": field_names, "message_type": message_type, "descriptor_source": descriptor_source, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Multi-dimensional values
    sizes = np.array([[1, 1]], dtype=np.int32)
    values = [np.array([[["test1"]]], dtype=np.string_), np.array([[[1]]], dtype=np.int64)]
    field_names = ["field1", "field2"]
    message_type = "MessageType"
    descriptor_source = "local://"
    name = "encode_proto_op_7"
    input_dict = {"sizes": sizes, "values": values, "field_names": field_names, "message_type": message_type, "descriptor_source": descriptor_source, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Different Descriptor Source
    sizes = np.array([[1, 1]], dtype=np.int32)
    values = [np.array([["test1"]], dtype=np.string_), np.array([[1]], dtype=np.int64)]
    field_names = ["field1", "field2"]
    message_type = "MessageType"
    descriptor_source = "bytes://test_bytes"
    name = "encode_proto_op_8"
    input_dict = {"sizes": sizes, "values": values, "field_names": field_names, "message_type": message_type, "descriptor_source": descriptor_source, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Larger Sizes
    sizes = np.array([[3, 2]], dtype=np.int32)
    values = [np.array([["test1", "test2", "test3"]], dtype=np.string_), np.array([[1, 2]], dtype=np.int64)]
    field_names = ["field1", "field2"]
    message_type = "MessageType"
    descriptor_source = "local://"
    name = "encode_proto_op_9"
    input_dict = {"sizes": sizes, "values": values, "field_names": field_names, "message_type": message_type, "descriptor_source": descriptor_source, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Empty Name
    sizes = np.array([[1, 1]], dtype=np.int32)
    values = [np.array([["test1"]], dtype=np.string_), np.array([[1]], dtype=np.int64)]
    field_names = ["field1", "field2"]
    message_type = "MessageType"
    descriptor_source = "local://"
    name = ""
    input_dict = {"sizes": sizes, "values": values, "field_names": field_names, "message_type": message_type, "descriptor_source": descriptor_source, "name": name}
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

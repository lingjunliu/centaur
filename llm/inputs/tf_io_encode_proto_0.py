
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
    values = [np.array([[1]], dtype=np.int32), np.array([[1.0]], dtype=np.float32)]
    field_names = ["int_field", "float_field"]
    message_type = "MyMessage"
    descriptor_source = "local://"
    name = "encode_proto_op"

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
    values = [np.array([[1, 2]], dtype=np.int32), np.array([[1.0]], dtype=np.float32)]
    field_names = ["int_field", "float_field"]
    message_type = "MyMessage"
    descriptor_source = ""
    name = None

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
    sizes = np.array([[[1, 1], [1, 1]]], dtype=np.int32)
    values = [np.array([[[1], [2]]], dtype=np.int32), np.array([[[1.0], [2.0]]], dtype=np.float32)]
    field_names = ["int_field", "float_field"]
    message_type = "MyMessage"
    descriptor_source = "local://"
    name = "encode_proto_op2"

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
    sizes = np.array([[1, 2, 1]], dtype=np.int32)
    values = [np.array([[1]], dtype=np.int32), np.array([[1, 2]], dtype=np.int64), np.array([[1.0]], dtype=np.float32)]
    field_names = ["int_field", "long_field", "float_field"]
    message_type = "MyMessage"
    descriptor_source = "local://"
    name = "encode_proto_op3"

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
    values = [np.array([[b"test_string"]], dtype=np.string_)]
    field_names = ["string_field"]
    message_type = "MyMessage"
    descriptor_source = "local://"
    name = "encode_proto_op4"

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
    sizes = np.array([[0, 0]], dtype=np.int32)
    values = [np.array([[]], dtype=np.int32), np.array([[]], dtype=np.float32)]
    field_names = ["int_field", "float_field"]
    message_type = "MyMessage"
    descriptor_source = "local://"
    name = "encode_proto_op5"

    input_dict = {
        "sizes": sizes,
        "values": values,
        "field_names": field_names,
        "message_type": message_type,
        "descriptor_source": descriptor_source,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    sizes = np.array([[1,1,1,1]], dtype=np.int32)
    values = [np.array([[1]], dtype=np.int32), np.array([[1]], dtype=np.int64), np.array([[1.0]], dtype=np.float32), np.array([[True]], dtype=np.bool_)]
    field_names = ["int_field", "long_field", "float_field", "bool_field"]
    message_type = "MyMessage"
    descriptor_source = "local://"
    name = "encode_proto_op6"

    input_dict = {
        "sizes": sizes,
        "values": values,
        "field_names": field_names,
        "message_type": message_type,
        "descriptor_source": descriptor_source,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8 - different batch shape
    sizes = np.array([[1, 1], [1, 1]], dtype=np.int32)
    values = [np.array([[1], [2]], dtype=np.int32), np.array([[1.0], [2.0]], dtype=np.float32)]
    field_names = ["int_field", "float_field"]
    message_type = "MyMessage"
    descriptor_source = "local://"
    name = "encode_proto_op7"

    input_dict = {
        "sizes": sizes,
        "values": values,
        "field_names": field_names,
        "message_type": message_type,
        "descriptor_source": descriptor_source,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9 - Mixed types, different sizes
    sizes = np.array([[1, 2, 1]], dtype=np.int32)
    values = [np.array([[1]], dtype=np.int32), np.array([[1,2]], dtype=np.int64), np.array([[b"test"]], dtype=np.string_)]
    field_names = ["int_field", "long_field", "string_field"]
    message_type = "MyMessage"
    descriptor_source = "local://"
    name = "encode_proto_op8"

    input_dict = {
        "sizes": sizes,
        "values": values,
        "field_names": field_names,
        "message_type": message_type,
        "descriptor_source": descriptor_source,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10 - Empty values, 0 sizes
    sizes = np.array([[0,0,0]], dtype=np.int32)
    values = [np.array([[]], dtype=np.int32), np.array([[]], dtype=np.int64), np.array([[]], dtype=np.string_)]
    field_names = ["int_field", "long_field", "string_field"]
    message_type = "MyMessage"
    descriptor_source = "local://"
    name = "encode_proto_op9"

    input_dict = {
        "sizes": sizes,
        "values": values,
        "field_names": field_names,
        "message_type": message_type,
        "descriptor_source": descriptor_source,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11 - Empty list of values, empty sizes
    sizes = np.empty((0, 0), dtype=np.int32)
    values = []
    field_names = []
    message_type = "MyMessage"
    descriptor_source = "local://"
    name = "encode_proto_op10"

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

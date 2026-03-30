
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_io_encode_proto_inputs():
    list_of_inputs = []

    # Input 1
    sizes = np.array([[1, 1]], dtype=np.int32)
    values = [np.array([[1.0]], dtype=np.float32), np.array([[1]], dtype=np.int32)]
    field_names = ["float_field", "int_field"]
    message_type = "MyMessageType"
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
    values = [np.array([[1.0, 2.0]], dtype=np.float32), np.array([[1]], dtype=np.int32)]
    field_names = ["float_field", "int_field"]
    message_type = "MyMessageType"
    descriptor_source = ""
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
    values = [np.array([[1.0]], dtype=np.float32), np.array([[1, 2]], dtype=np.int32)]
    field_names = ["float_field", "int_field"]
    message_type = "MyMessageType"
    descriptor_source = "local://"
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

    # Input 4
    sizes = np.array([[[1, 1]]], dtype=np.int32)
    values = [np.array([[[1.0]]], dtype=np.float32), np.array([[[1]]], dtype=np.int32)]
    field_names = ["float_field", "int_field"]
    message_type = "MyMessageType"
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
    sizes = np.array([[1, 1]], dtype=np.int32)
    values = [np.array([[1.0]], dtype=np.float32), np.array([[1]], dtype=np.int32)]
    field_names = ["float_field", "int_field"]
    message_type = "MyMessageType"
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
    sizes = np.array([[1, 1]], dtype=np.int32)
    values = [np.array([[1.0]], dtype=np.float64), np.array([[1]], dtype=np.int64)]
    field_names = ["float_field", "int_field"]
    message_type = "MyMessageType"
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

   # Input 7
    sizes = np.array([[1, 1]], dtype=np.int32)
    values = [np.array([[1.0]], dtype=np.float32), np.array([[True]], dtype=np.bool_)]
    field_names = ["float_field", "bool_field"]
    message_type = "MyMessageType"
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

   # Input 8
    sizes = np.array([[1, 1]], dtype=np.int32)
    values = [np.array([[1.0]], dtype=np.float32), np.array([["test"]], dtype=np.dtype('U'))]
    field_names = ["float_field", "string_field"]
    message_type = "MyMessageType"
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

   # Input 9
    sizes = np.array([[1]], dtype=np.int32)
    values = [np.array([[1.0]], dtype=np.float32)]
    field_names = ["float_field"]
    message_type = "MyMessageType"
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
    sizes = np.array([[2, 2]], dtype=np.int32)
    values = [np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), np.array([[1, 2], [3, 4]], dtype=np.int32)]
    field_names = ["float_field", "int_field"]
    message_type = "MyMessageType"
    descriptor_source = "local://"
    name = "encode_proto_op_10"

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
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.io.encode_proto' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.io.encode_proto'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.io.encode_proto', generated_inputs['tf.io.encode_proto'], lib="tf", suffix=0)

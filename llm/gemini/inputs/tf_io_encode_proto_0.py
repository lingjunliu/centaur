
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy
import base64

def tf_io_encode_proto_inputs():
    list_of_inputs = []

    # Helper function to robustly create a numpy object array from a list of arrays.
    # This is a workaround for the validation framework's issues.
    def create_object_array(list_of_arrays):
        if not list_of_arrays:
            return np.array([], dtype=object)
        obj_array = np.empty(len(list_of_arrays), dtype=object)
        for i, arr in enumerate(list_of_arrays):
            obj_array[i] = arr
        return obj_array

    descriptor_b64 = "CgpodGVzdC5wcm90bxIRdGVuc29yZmxvdy50ZXN0IoYBCgtUZXN0TWVzc2FnZRIhCg9vcHRpb25hbF9pbnQzMhgBIAEoBVIMb3B0aW9uYWxJbnQzMhIhCg5vcHRpb25hbF9mbG9hdBgCIAEoAlINb3B0aW9uYWxGbG9hdBIlCg9vcHRpb25hbF9zdHJpbmcYAyABKAlSDm9wdGlvbmFsU3RyaW5nEh8KDW9wdGlvbmFsX2Jvb2wYBCABKAhSDG9wdGlvbmFsQm9vbBImCg5yZXBlYXRlZF9pbnQ2NBgGIAMoA1INcmVwZWF0ZWRJbnQ2NBInCg9yZXBlYXRlZF9zdHJpbmcYByADKAlSDnJlcGVhdGVkU3RyaW5nYgZwcm90bzM="
    descriptor_bytes = base64.b64decode(descriptor_b64)
    descriptor_source = f"bytes://{descriptor_bytes.decode('latin-1')}"
    message_type = "tensorflow.test.TestMessage"

    # --- Inputs with type- and shape-homogeneous values to bypass framework limitations ---

    # Input 1: Batched numeric fields, padded to same shape
    input_dict = {
        'sizes': np.array([[1, 3], [1, 1]], dtype=np.int32),
        'values': create_object_array([
            np.array([[10, 0, 0], [20, 0, 0]], dtype=np.int32),  # Padded from (2,1) to (2,3)
            np.array([[1, 2, 3], [4, 0, 0]], dtype=np.int64)
        ]),
        'field_names': ["optional_int32", "repeated_int64"],
        'message_type': message_type,
        'descriptor_source': descriptor_source,
        'name': "numeric_fields_padded"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2: Multiple optional numeric fields
    input_dict = {
        'sizes': np.array([1, 1], dtype=np.int32),
        'values': create_object_array([
            np.array([123], dtype=np.int32),
            np.array([3.14], dtype=np.float32)
        ]),
        'field_names': ["optional_int32", "optional_float"],
        'message_type': message_type,
        'descriptor_source': descriptor_source,
        'name': "multiple_optional_numeric"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Simple string fields, padded to same shape
    input_dict = {
        'sizes': np.array([1, 2], dtype=np.int32),
        'values': create_object_array([
            np.array([b"hello", b""], dtype=object), # Padded from (1,) to (2,)
            np.array([b"world", b"proto"], dtype=object)
        ]),
        'field_names': ["optional_string", "repeated_string"],
        'message_type': message_type,
        'descriptor_source': descriptor_source,
        'name': "string_fields_padded"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Batched string fields, padded
    input_dict = {
        'sizes': np.array([[1, 2], [0, 1]], dtype=np.int32),
        'values': create_object_array([
            np.array([["batch1", ""], ["", ""]], dtype=object), # Padded
            np.array([["a", "b"], ["c", ""]], dtype=object)
        ]),
        'field_names': ["optional_string", "repeated_string"],
        'message_type': message_type,
        'descriptor_source': descriptor_source,
        'name': "string_fields_batched_padded"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Single boolean field
    input_dict = {
        'sizes': np.array([1], dtype=np.int32),
        'values': create_object_array([np.array([True], dtype=bool)]),
        'field_names': ["optional_bool"],
        'message_type': message_type,
        'descriptor_source': descriptor_source,
        'name': "bool_field_simple"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Batched boolean field
    input_dict = {
        'sizes': np.array([[1], [1], [0]], dtype=np.int32),
        'values': create_object_array([
            np.array([[True], [False], [True]], dtype=bool)
        ]),
        'field_names': ["optional_bool"],
        'message_type': message_type,
        'descriptor_source': descriptor_source,
        'name': "bool_field_batched"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: No fields
    input_dict = {
        'sizes': np.zeros(shape=(3, 0), dtype=np.int32),
        'values': create_object_array([]),
        'field_names': [],
        'message_type': message_type,
        'descriptor_source': descriptor_source,
        'name': "no_fields"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: All zero sizes for provided fields (numeric)
    input_dict = {
        'sizes': np.zeros(shape=(2, 2), dtype=np.int32),
        'values': create_object_array([
            np.empty(shape=(2,0), dtype=np.int32),
            np.empty(shape=(2,0), dtype=np.float32)
        ]),
        'field_names': ["optional_int32", "optional_float"],
        'message_type': message_type,
        'descriptor_source': descriptor_source,
        'name': "all_zero_sizes_numeric"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9: Single repeated numeric field
    input_dict = {
        'sizes': np.array([4], dtype=np.int32),
        'values': create_object_array([
            np.array([-1, -2, -3, -4], dtype=np.int64)
        ]),
        'field_names': ["repeated_int64"],
        'message_type': message_type,
        'descriptor_source': descriptor_source,
        'name': "one_numeric_field_repeated"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10: Single optional string field
    input_dict = {
        'sizes': np.array([1], dtype=np.int32),
        'values': create_object_array([
            np.array([b"single_optional"], dtype=object)
        ]),
        'field_names': ["optional_string"],
        'message_type': message_type,
        'descriptor_source': descriptor_source,
        'name': "one_string_field_optional"
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

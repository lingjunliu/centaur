
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_strings_to_number_inputs():
    list_of_inputs = []

    # Input 1
    input_tensor = np.array("1.55", dtype=np.object_)
    out_type = tf.float32
    name = "float_conversion"
    input_dict = {"input": input_tensor, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_tensor = np.array("3", dtype=np.object_)
    out_type = tf.int32
    name = "int_conversion"
    input_dict = {"input": input_tensor, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Array of strings
    input_tensor = np.array(["1", "2", "3"], dtype=np.object_)
    out_type = tf.int32
    name = "int_array_conversion"
    input_dict = {"input": input_tensor, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Float64 conversion
    input_tensor = np.array("3.14159", dtype=np.object_)
    out_type = tf.float64
    name = "float64_conversion"
    input_dict = {"input": input_tensor, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Int64 conversion
    input_tensor = np.array("1234567890", dtype=np.object_)
    out_type = tf.int64
    name = "int64_conversion"
    input_dict = {"input": input_tensor, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Unsigned Int32 conversion
    input_tensor = np.array("4294967290", dtype=np.object_)
    out_type = tf.uint32
    name = "uint32_conversion"
    input_dict = {"input": input_tensor, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Unsigned Int64 conversion
    input_tensor = np.array("18446744073709551610", dtype=np.object_)
    out_type = tf.uint64
    name = "uint64_conversion"
    input_dict = {"input": input_tensor, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Negative number to int32
    input_tensor = np.array("-10", dtype=np.object_)
    out_type = tf.int32
    name = "negative_int_conversion"
    input_dict = {"input": input_tensor, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Negative number to float32
    input_tensor = np.array("-2.718", dtype=np.object_)
    out_type = tf.float32
    name = "negative_float_conversion"
    input_dict = {"input": input_tensor, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D Array of strings to float32
    input_tensor = np.array([["1.0", "2.0"], ["3.0", "4.0"]], dtype=np.object_)
    out_type = tf.float32
    name = "2d_float_array"
    input_dict = {"input": input_tensor, "out_type": out_type, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.strings.to_number"] = tf_strings_to_number_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.strings.to_number' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.strings.to_number'.")

check_valid('tf.strings.to_number', generated_inputs['tf.strings.to_number'], lib="tf", suffix=0)

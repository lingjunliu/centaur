
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_IsVariableInitialized_inputs():
    list_of_inputs = []

    def create_variable(initial_value, dtype, name):
        var = tf.Variable(initial_value=initial_value, dtype=dtype, name=name)
        return var
    # Input 1: Basic Variable
    v1 = create_variable(0, tf.int32, "basic_var")
    input_dict = {"ref": v1.ref(), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float Variable
    v2 = create_variable(0.0, tf.float32, "float_var")
    input_dict = {"ref": v2.ref(), "name": "float_var"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Bool Variable
    v3 = create_variable(False, tf.bool, "bool_var")
    input_dict = {"ref": v3.ref(), "name": "bool_var"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: String Variable
    v4 = create_variable("hello", tf.string, "string_var")
    input_dict = {"ref": v4.ref(), "name": "string_var"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Rank 2 Variable
    v5 = create_variable(np.zeros((2, 3), dtype=np.int32), tf.int32, "rank2_var")
    input_dict = {"ref": v5.ref(), "name": "rank2_var"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Rank 3 Variable
    v6 = create_variable(np.ones((2, 3, 4), dtype=np.float32), tf.float32, "rank3_var")
    input_dict = {"ref": v6.ref(), "name": "rank3_var"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Empty name
    v7 = create_variable(5, tf.int32, "empty_name_var")
    input_dict = {"ref": v7.ref(), "name": ""}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Unicode name
    v8 = create_variable(10, tf.int32, "unicode_name_var")
    input_dict = {"ref": v8.ref(), "name": "你好"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Very long name
    v9 = create_variable(15, tf.int32, "long_name_var")
    long_name = "a" * 200
    input_dict = {"ref": v9.ref(), "name": long_name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Complex64 Variable
    v10 = create_variable(np.complex64(1+1j), tf.complex64, "complex_var")
    input_dict = {"ref": v10.ref(), "name": "complex_var"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.IsVariableInitialized"] = tf_raw_ops_IsVariableInitialized_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.IsVariableInitialized' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.IsVariableInitialized'.")

check_valid('tf.raw_ops.IsVariableInitialized', generated_inputs['tf.raw_ops.IsVariableInitialized'], lib="tf", suffix=0)

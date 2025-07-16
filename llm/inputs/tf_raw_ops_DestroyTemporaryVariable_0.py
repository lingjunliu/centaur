
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_DestroyTemporaryVariable_inputs():
    list_of_inputs = []

    # Input 1
    ref = tf.constant(np.array([1, 2, 3], dtype=np.int32))
    var_name = "temp_var_1"
    name = "destroy_op_1"
    input_dict = {"ref": ref, "var_name": var_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    ref = tf.constant(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    var_name = "temp_var_2"
    name = "destroy_op_2"
    input_dict = {"ref": ref, "var_name": var_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    ref = tf.constant(np.array([True, False, True], dtype=np.bool_))
    var_name = "temp_var_3"
    name = "destroy_op_3"
    input_dict = {"ref": ref, "var_name": var_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    ref = tf.constant(np.array(["a", "b", "c"], dtype=np.string_))
    var_name = "temp_var_4"
    name = "destroy_op_4"
    input_dict = {"ref": ref, "var_name": var_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    ref = tf.constant(np.array([1, 2, 3, 4, 5], dtype=np.int64))
    var_name = "temp_var_5"
    name = "destroy_op_5"
    input_dict = {"ref": ref, "var_name": var_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    ref = tf.constant(np.array([-1, -2, -3], dtype=np.int32))
    var_name = "temp_var_6"
    name = "destroy_op_6"
    input_dict = {"ref": ref, "var_name": var_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    ref = tf.constant(np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float32))
    var_name = "temp_var_7"
    name = "destroy_op_7"
    input_dict = {"ref": ref, "var_name": var_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    ref = tf.constant(np.array([False, False, False], dtype=np.bool_))
    var_name = "temp_var_8"
    name = "destroy_op_8"
    input_dict = {"ref": ref, "var_name": var_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    ref = tf.constant(np.array([b"a", b"b", b"c"], dtype=np.string_))
    var_name = "temp_var_9"
    name = "destroy_op_9"
    input_dict = {"ref": ref, "var_name": var_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    ref = tf.constant(np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.int64))
    var_name = "temp_var_10"
    name = "destroy_op_10"
    input_dict = {"ref": ref, "var_name": var_name, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.DestroyTemporaryVariable"] = tf_raw_ops_DestroyTemporaryVariable_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.DestroyTemporaryVariable' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DestroyTemporaryVariable'.")

check_valid('tf.raw_ops.DestroyTemporaryVariable', generated_inputs['tf.raw_ops.DestroyTemporaryVariable'], lib="tf", suffix=0)

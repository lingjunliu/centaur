
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_destroy_temporary_variable_inputs():
    list_of_inputs = []

    # Input 1: Simple float tensor
    ref = tf.compat.v1.get_variable(name="temp_var_1", shape=[], dtype=tf.float32, initializer=tf.constant_initializer(1.0), use_resource=True)
    var_name = "temp_var_1"
    input_dict = {"ref": ref, "var_name": var_name, "name": "destroy_op_1"}
    list_of_inputs.append(input_dict)

    # Input 2: Integer tensor
    ref = tf.compat.v1.get_variable(name="temp_var_2", shape=[], dtype=tf.int32, initializer=tf.constant_initializer(5), use_resource=True)
    var_name = "temp_var_2"
    input_dict = {"ref": ref, "var_name": var_name, "name": "destroy_op_2"}
    list_of_inputs.append(input_dict)

    # Input 3: Rank 2 float tensor
    ref = tf.compat.v1.get_variable(name="temp_var_3", shape=[2, 2], dtype=tf.float32, initializer=tf.constant_initializer(np.array([[1.0, 2.0], [3.0, 4.0]])), use_resource=True)
    var_name = "temp_var_3"
    input_dict = {"ref": ref, "var_name": var_name, "name": "destroy_op_3"}
    list_of_inputs.append(input_dict)

    # Input 4: Rank 3 integer tensor
    ref = tf.compat.v1.get_variable(name="temp_var_4", shape=[2, 2, 2], dtype=tf.int32, initializer=tf.constant_initializer(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])), use_resource=True)
    var_name = "temp_var_4"
    input_dict = {"ref": ref, "var_name": var_name, "name": "destroy_op_4"}
    list_of_inputs.append(input_dict)

    # Input 5: Bool tensor
    ref = tf.compat.v1.get_variable(name="temp_var_5", shape=[], dtype=tf.bool, initializer=tf.constant_initializer(True), use_resource=True)
    var_name = "temp_var_5"
    input_dict = {"ref": ref, "var_name": var_name, "name": "destroy_op_5"}
    list_of_inputs.append(input_dict)
    
    # Input 6: String tensor
    ref = tf.compat.v1.get_variable(name="temp_var_6", shape=[], dtype=tf.string, initializer=tf.constant_initializer(b"hello"), use_resource=True)
    var_name = "temp_var_6"
    input_dict = {"ref": ref, "var_name": var_name, "name": "destroy_op_6"}
    list_of_inputs.append(input_dict)

    # Input 7: Different variable name
    ref = tf.compat.v1.get_variable(name="temp_var_7", shape=[], dtype=tf.float32, initializer=tf.constant_initializer(2.5), use_resource=True)
    var_name = "different_temp_var"
    input_dict = {"ref": ref, "var_name": var_name, "name": "destroy_op_7"}
    list_of_inputs.append(input_dict)

    # Input 8: Larger integer
    ref = tf.compat.v1.get_variable(name="temp_var_8", shape=[], dtype=tf.int64, initializer=tf.constant_initializer(10000), use_resource=True)
    var_name = "temp_var_8"
    input_dict = {"ref": ref, "var_name": var_name, "name": "destroy_op_8"}
    list_of_inputs.append(input_dict)
    
    # Input 9: Rank 4 tensor
    ref = tf.compat.v1.get_variable(name="temp_var_9", shape=[2,2,2,2], dtype=tf.float32, initializer=tf.constant_initializer(np.random.rand(2,2,2,2).astype(np.float32)), use_resource=True)
    var_name = "temp_var_9"
    input_dict = {"ref": ref, "var_name": var_name, "name": "destroy_op_9"}
    list_of_inputs.append(input_dict)

    # Input 10: Empty string name
    ref = tf.compat.v1.get_variable(name="temp_var_10", shape=[], dtype=tf.int32, initializer=tf.constant_initializer(1), use_resource=True)
    var_name = ""
    input_dict = {"ref": ref, "var_name": var_name, "name": "destroy_op_10"}
    list_of_inputs.append(input_dict)

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.DestroyTemporaryVariable"] = tf_raw_ops_destroy_temporary_variable_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.DestroyTemporaryVariable' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DestroyTemporaryVariable'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.DestroyTemporaryVariable', generated_inputs['tf.raw_ops.DestroyTemporaryVariable'], lib="tf", suffix=0)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_is_variable_initialized_inputs():
    list_of_inputs = []

    # Input 1: Basic case with a uninitialized variable
    v1 = tf.Variable(np.zeros((2, 2), dtype=np.float32), validate_shape=False)
    input_dict = {"ref": v1.handle, "name": "is_initialized_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Variable initialized with ones.
    v2 = tf.Variable(np.ones((3, 3), dtype=np.int32))
    input_dict = {"ref": v2.handle, "name": "is_initialized_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Uninitialized variable with a different shape
    v3 = tf.Variable(np.zeros((1, 5), dtype=np.bool_), validate_shape=False)
    input_dict = {"ref": v3.handle, "name": "is_initialized_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Initialized variable with a different shape and dtype
    v4 = tf.Variable(np.random.rand(4, 1, 2).astype(np.float64))
    input_dict = {"ref": v4.handle, "name": "is_initialized_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Uninitialized variable with a different dtype
    v5 = tf.Variable(np.zeros((2, 2), dtype=np.complex64), validate_shape=False)
    input_dict = {"ref": v5.handle, "name": "is_initialized_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Initialized variable with a scalar value
    v6 = tf.Variable(10)
    input_dict = {"ref": v6.handle, "name": "is_initialized_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Uninitialized variable, int64 type
    v7 = tf.Variable(np.zeros((2, 2), dtype=np.int64), validate_shape=False)
    input_dict = {"ref": v7.handle, "name": "is_initialized_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Initialized variable, string type
    v8 = tf.Variable(np.array(["hello", "world"]), dtype=tf.string)
    input_dict = {"ref": v8.handle, "name": "is_initialized_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Uninitialized Variable with empty shape
    v9 = tf.Variable(np.array([]), dtype=np.float32, validate_shape=False)
    input_dict = {"ref": v9.handle, "name": "is_initialized_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Initialized Variable with a large shape
    v10 = tf.Variable(np.random.rand(100, 100).astype(np.float32))
    input_dict = {"ref": v10.handle, "name": "is_initialized_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: Initialized variable with int32 dtype and then assigned
    v11 = tf.Variable(1, dtype=tf.int32)
    v11.assign(10)
    input_dict = {"ref": v11.handle, "name": "is_initialized_11"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Uninitialized variable with int32 dtype
    v12 = tf.Variable(1, dtype=tf.int32, validate_shape=False)
    input_dict = {"ref": v12.handle, "name": "is_initialized_12"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.IsVariableInitialized"] = tf_raw_ops_is_variable_initialized_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.raw_ops.IsVariableInitialized' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.IsVariableInitialized'.")

check_valid('tf.raw_ops.IsVariableInitialized', generated_inputs['tf.raw_ops.IsVariableInitialized'], lib="tf", suffix=0)

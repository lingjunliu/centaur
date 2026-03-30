
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_count_up_to_inputs():
    list_of_inputs = []

    # Input 1: Scalar int32, small limit
    with tf.compat.v1.variable_scope("scope1"):
        ref_var = tf.compat.v1.get_variable("count1", shape=[], dtype=tf.int32, initializer=tf.zeros_initializer(), use_resource=True)
    limit = 5
    input_dict = {"ref": ref_var, "limit": limit, "name": "count_up_to_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Scalar int64, larger limit
    with tf.compat.v1.variable_scope("scope2"):
        ref_var = tf.compat.v1.get_variable("count2", shape=[], dtype=tf.int64, initializer=tf.zeros_initializer(), use_resource=True)
    limit = 100
    input_dict = {"ref": ref_var, "limit": limit, "name": "count_up_to_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Scalar int32, zero limit
    with tf.compat.v1.variable_scope("scope3"):
        ref_var = tf.compat.v1.get_variable("count3", shape=[], dtype=tf.int32, initializer=tf.zeros_initializer(), use_resource=True)
    limit = 0
    input_dict = {"ref": ref_var, "limit": limit, "name": "count_up_to_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 4: Scalar int64, negative limit (should cause out of range immediately)
    with tf.compat.v1.variable_scope("scope4"):
        ref_var = tf.compat.v1.get_variable("count4", shape=[], dtype=tf.int64, initializer=tf.zeros_initializer(), use_resource=True)
    limit = -5
    input_dict = {"ref": ref_var, "limit": limit, "name": "count_up_to_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Scalar int32, limit close to max int32
    with tf.compat.v1.variable_scope("scope5"):
        ref_var = tf.compat.v1.get_variable("count5", shape=[], dtype=tf.int32, initializer=tf.zeros_initializer(), use_resource=True)
    limit = np.iinfo(np.int32).max - 1
    input_dict = {"ref": ref_var, "limit": limit, "name": "count_up_to_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Scalar int64, limit close to max int64
    with tf.compat.v1.variable_scope("scope6"):
        ref_var = tf.compat.v1.get_variable("count6", shape=[], dtype=tf.int64, initializer=tf.zeros_initializer(), use_resource=True)
    limit = np.iinfo(np.int64).max - 1
    input_dict = {"ref": ref_var, "limit": limit, "name": "count_up_to_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7: Scalar int32, limit = 1
    with tf.compat.v1.variable_scope("scope7"):
        ref_var = tf.compat.v1.get_variable("count7", shape=[], dtype=tf.int32, initializer=tf.zeros_initializer(), use_resource=True)
    limit = 1
    input_dict = {"ref": ref_var, "limit": limit, "name": "count_up_to_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Scalar int64, limit = 1
    with tf.compat.v1.variable_scope("scope8"):
        ref_var = tf.compat.v1.get_variable("count8", shape=[], dtype=tf.int64, initializer=tf.zeros_initializer(), use_resource=True)
    limit = 1
    input_dict = {"ref": ref_var, "limit": limit, "name": "count_up_to_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Scalar int32, limit bigger than initial ref value.
    with tf.compat.v1.variable_scope("scope9"):
        ref_var = tf.compat.v1.get_variable("count9", shape=[], dtype=tf.int32, initializer=tf.constant_initializer(5), use_resource=True)
    limit = 10
    input_dict = {"ref": ref_var, "limit": limit, "name": "count_up_to_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Scalar int64, limit bigger than initial ref value.
    with tf.compat.v1.variable_scope("scope10"):
        ref_var = tf.compat.v1.get_variable("count10", shape=[], dtype=tf.int64, initializer=tf.constant_initializer(5), use_resource=True)
    limit = 10
    input_dict = {"ref": ref_var, "limit": limit, "name": "count_up_to_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: Scalar int32, limit equal to initial ref value
    with tf.compat.v1.variable_scope("scope11"):
        ref_var = tf.compat.v1.get_variable("count11", shape=[], dtype=tf.int32, initializer=tf.constant_initializer(5), use_resource=True)
    limit = 5
    input_dict = {"ref": ref_var, "limit": limit, "name": "count_up_to_11"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 12: Scalar int64, limit equal to initial ref value
    with tf.compat.v1.variable_scope("scope12"):
        ref_var = tf.compat.v1.get_variable("count12", shape=[], dtype=tf.int64, initializer=tf.constant_initializer(5), use_resource=True)
    limit = 5
    input_dict = {"ref": ref_var, "limit": limit, "name": "count_up_to_12"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.CountUpTo"] = tf_raw_ops_count_up_to_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.CountUpTo' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.CountUpTo'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.CountUpTo', generated_inputs['tf.raw_ops.CountUpTo'], lib="tf", suffix=0)

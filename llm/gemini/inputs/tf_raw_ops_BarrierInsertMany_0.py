
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_barrier_insert_many_inputs():
    list_of_inputs = []

    # Input 1
    handle = "barrier_handle"
    keys = ["key1", "key2"]
    values = np.array([[1, 2], [3, 4]], dtype=np.int32)
    component_index = 0
    input_dict = {"handle": handle, "keys": keys, "values": values, "component_index": component_index, "name": "insert_many_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    handle = "another_barrier"
    keys = ["key3"]
    values = np.array([[5.5, 6.6]], dtype=np.float32)
    component_index = 1
    input_dict = {"handle": handle, "keys": keys, "values": values, "component_index": component_index, "name": "insert_many_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    handle = "yet_another"
    keys = ["key4", "key5", "key6"]
    values = np.array([["a", "b"], ["c", "d"], ["e", "f"]], dtype=np.string_)
    component_index = 2
    input_dict = {"handle": handle, "keys": keys, "values": values, "component_index": component_index, "name": "insert_many_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    handle = "barrier_handle_4"
    keys = ["key7", "key8"]
    values = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    component_index = 0
    input_dict = {"handle": handle, "keys": keys, "values": values, "component_index": component_index, "name": "insert_many_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    handle = "barrier_handle_5"
    keys = ["key9"]
    values = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float64)
    component_index = 1
    input_dict = {"handle": handle, "keys": keys, "values": values, "component_index": component_index, "name": "insert_many_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    handle = "barrier_handle_6"
    keys = ["key10", "key11", "key12"]
    values = np.array([True, False, True], dtype=np.bool_)
    component_index = 2
    values = np.reshape(values, (3, 1))
    input_dict = {"handle": handle, "keys": keys, "values": values, "component_index": component_index, "name": "insert_many_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    handle = "barrier_handle_7"
    keys = ["key13", "key14"]
    values = np.array([[1], [2]], dtype=np.int8)
    component_index = 0
    input_dict = {"handle": handle, "keys": keys, "values": values, "component_index": component_index, "name": "insert_many_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    handle = "barrier_handle_8"
    keys = ["key15"]
    values = np.array([[1.5]], dtype=np.float16)
    component_index = 1
    input_dict = {"handle": handle, "keys": keys, "values": values, "component_index": component_index, "name": "insert_many_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9
    handle = "barrier_handle_9"
    keys = ["key16", "key17", "key18"]
    values = np.array([[-1, -2], [-3, -4], [-5, -6]], dtype=np.int32)
    component_index = 2
    input_dict = {"handle": handle, "keys": keys, "values": values, "component_index": component_index, "name": "insert_many_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    handle = "barrier_handle_10"
    keys = ["key19"]
    values = np.array([0], dtype=np.int32)
    component_index = 5
    input_dict = {"handle": handle, "keys": keys, "values": values, "component_index": component_index, "name": "insert_many_10"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.raw_ops.BarrierInsertMany"] = tf_raw_ops_barrier_insert_many_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.BarrierInsertMany' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.BarrierInsertMany'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.BarrierInsertMany', generated_inputs['tf.raw_ops.BarrierInsertMany'], lib="tf", suffix=0)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy
import os

def tf_data_experimental_load_inputs():
    list_of_inputs = []

    # Input 1, valid, no element_spec, no compression, no reader_func
    input_dict = {
        "path": "/tmp/saved_dataset_1",
        "element_spec": [],
        "compression": "NONE",
        "reader_func": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, valid, element_spec, compression, no reader_func
    input_dict = {
        "path": "/tmp/saved_dataset_2",
        "element_spec": [tf.TensorSpec(shape=(None,), dtype=tf.int64, name=None)],
        "compression": "GZIP",
        "reader_func": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid, element_spec, compression, reader_func
    def custom_reader_func(datasets):
        return datasets.interleave(lambda x: x, num_parallel_calls=tf.data.AUTOTUNE)

    input_dict = {
        "path": "/tmp/saved_dataset_3",
        "element_spec": [tf.TensorSpec(shape=(), dtype=tf.float32, name=None)],
        "compression": "NONE",
        "reader_func": [custom_reader_func]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, valid, different element_spec
    input_dict = {
        "path": "/tmp/saved_dataset_4",
        "element_spec": [tf.TensorSpec(shape=(2, 2), dtype=tf.float64, name=None)],
        "compression": "NONE",
        "reader_func": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, valid, different element_spec
    input_dict = {
        "path": "/tmp/saved_dataset_5",
        "element_spec": [tf.RaggedTensorSpec(shape=(None, None), dtype=tf.int32, ragged_rank=1)],
        "compression": "NONE",
        "reader_func": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6, valid, nested element_spec
    input_dict = {
        "path": "/tmp/saved_dataset_6",
        "element_spec": [[tf.TensorSpec(shape=(), dtype=tf.int32, name=None), tf.TensorSpec(shape=(), dtype=tf.string, name=None)]],
        "compression": "NONE",
        "reader_func": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7, valid, complex element_spec and reader_func
    def another_reader_func(datasets):
        return datasets.unbatch()

    input_dict = {
        "path": "/tmp/saved_dataset_7",
        "element_spec": [tf.TensorSpec(shape=(None, 3), dtype=tf.float32, name=None)],
        "compression": "GZIP",
        "reader_func": [another_reader_func]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8, valid, multiple levels of nesting in element spec.
    input_dict = {
        "path": "/tmp/saved_dataset_8",
        "element_spec": [[[tf.TensorSpec(shape=(), dtype=tf.int64, name=None)]]],
        "compression": "NONE",
        "reader_func": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 9, valid, different element_spec, no compression
    def no_op(datasets):
        return datasets

    input_dict = {
        "path": "/tmp/saved_dataset_9",
        "element_spec": [tf.TensorSpec(shape=(10,), dtype=tf.int32)],
        "compression": None,
        "reader_func": [no_op]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.data.experimental.load"] = tf_data_experimental_load_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.data.experimental.load' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.load'.")

check_valid('tf.data.experimental.load', generated_inputs['tf.data.experimental.load'], lib="tf", suffix=0)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy
import os

def tf_data_experimental_load_inputs():
    list_of_inputs = []

    # Input 1: Basic valid input
    path = "/tmp/dataset_path_1"
    element_spec = [tf.TensorSpec(shape=(2,), dtype=tf.int32, name=None)]
    compression = "NONE"

    input_dict = {
        "path": path,
        "element_spec": element_spec,
        "compression": compression,
        "reader_func": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: GZIP compression
    path = "/tmp/dataset_path_2"
    element_spec = [tf.TensorSpec(shape=(), dtype=tf.float32, name=None)]
    compression = "GZIP"

    input_dict = {
        "path": path,
        "element_spec": element_spec,
        "compression": compression,
        "reader_func": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: More complex element_spec
    path = "/tmp/dataset_path_3"
    element_spec = [
        tf.TensorSpec(shape=(None,), dtype=tf.string, name=None),
        tf.TensorSpec(shape=(3, 3), dtype=tf.float64, name=None)
    ]
    compression = "NONE"

    input_dict = {
        "path": path,
        "element_spec": element_spec,
        "compression": compression,
        "reader_func": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4:  Path with subdirectories
    path = "/tmp/nested/dataset_path_5"
    element_spec = [tf.TensorSpec(shape=(), dtype=tf.bool, name=None)]
    compression = "NONE"

    input_dict = {
        "path": path,
        "element_spec": element_spec,
        "compression": compression,
        "reader_func": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Empty shape
    path = "/tmp/dataset_path_6"
    element_spec = [tf.TensorSpec(shape=(), dtype=tf.int32, name=None)]
    compression = "NONE"

    input_dict = {
        "path": path,
        "element_spec": element_spec,
        "compression": compression,
        "reader_func": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Unknown shape (None) for all dimensions
    path = "/tmp/dataset_path_7"
    element_spec = [tf.TensorSpec(shape=(None, None, None), dtype=tf.float32, name=None)]
    compression = "NONE"

    input_dict = {
        "path": path,
        "element_spec": element_spec,
        "compression": compression,
        "reader_func": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Multiple element specs
    path = "/tmp/dataset_path_8"
    element_spec = [
        tf.TensorSpec(shape=(10,), dtype=tf.int32, name=None),
        tf.TensorSpec(shape=(5, 5), dtype=tf.float32, name=None),
        tf.TensorSpec(shape=(), dtype=tf.string, name=None)
    ]
    compression = "NONE"

    input_dict = {
        "path": path,
        "element_spec": element_spec,
        "compression": compression,
        "reader_func": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 8: Using a more complex path
    path = "/tmp/my_data/version_1/dataset_9"
    element_spec = [tf.TensorSpec(shape=(2,), dtype=tf.int32, name=None)]
    compression = "NONE"

    input_dict = {
        "path": path,
        "element_spec": element_spec,
        "compression": compression,
        "reader_func": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Float64 type
    path = "/tmp/dataset_path_10"
    element_spec = [tf.TensorSpec(shape=(2,), dtype=tf.float64, name=None)]
    compression = "NONE"

    input_dict = {
        "path": path,
        "element_spec": element_spec,
        "compression": compression,
        "reader_func": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Valid reader function
    path = "/tmp/dataset_path_11"
    element_spec = [tf.TensorSpec(shape=(5,), dtype=tf.int64, name=None)]
    compression = "NONE"

    def dummy_reader_func(datasets):
      return datasets.interleave(lambda x: x, num_parallel_calls=tf.data.AUTOTUNE)

    input_dict = {
        "path": path,
        "element_spec": element_spec,
        "compression": compression,
        "reader_func": [dummy_reader_func]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.data.experimental.load"] = tf_data_experimental_load_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.data.experimental.load' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.load'.")

check_valid('tf.data.experimental.load', generated_inputs['tf.data.experimental.load'], lib="tf", suffix=0)

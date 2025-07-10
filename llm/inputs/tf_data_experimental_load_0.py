
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import os
import numpy as np

generated_inputs = {}

def tf_data_experimental_load_inputs():
    list_of_inputs = []

    # Helper function to create a dummy saved dataset directory
    def create_dummy_dataset(path):
        dataset = tf.data.Dataset.range(5)
        tf.data.experimental.save(dataset, path)
        return path

    # Create dummy datasets
    temp_dir = os.path.join("./")

    dummy_path1 = os.path.join(temp_dir, "saved_data1")
    dummy_path1 = create_dummy_dataset(dummy_path1)
    dummy_path2 = os.path.join(temp_dir, "saved_data2")
    dummy_path2 = create_dummy_dataset(dummy_path2)
    dummy_path3 = os.path.join(temp_dir, "saved_data3")
    dummy_path3 = create_dummy_dataset(dummy_path3)
    dummy_path4 = os.path.join(temp_dir, "saved_data4")
    dummy_path4 = create_dummy_dataset(dummy_path4)
    dummy_path5 = os.path.join(temp_dir, "saved_data5")
    dummy_path5 = create_dummy_dataset(dummy_path5)
    dummy_path6 = os.path.join(temp_dir, "saved_data6")
    dummy_path6 = create_dummy_dataset(dummy_path6)
    dummy_path7 = os.path.join(temp_dir, "saved_data7")
    dummy_path7 = create_dummy_dataset(dummy_path7)
    dummy_path8 = os.path.join(temp_dir, "saved_data8")
    dummy_path8 = create_dummy_dataset(dummy_path8)
    dummy_path9 = os.path.join(temp_dir, "saved_data9")
    dummy_path9 = create_dummy_dataset(dummy_path9)
    dummy_path10 = os.path.join(temp_dir, "saved_data10")
    dummy_path10 = create_dummy_dataset(dummy_path10)

    # Input 1: Basic usage with default options
    input_dict = {
        "path": dummy_path1,
        "element_spec": [],
        "compression": "NONE",
        "reader_func": []
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: With element_spec defined
    input_dict = {
        "path": dummy_path2,
        "element_spec": [tf.TensorSpec(shape=(), dtype=tf.int64, name=None)],
        "compression": "NONE",
        "reader_func": []
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: With GZIP compression
    input_dict = {
        "path": dummy_path3,
        "element_spec": [],
        "compression": "GZIP",
        "reader_func": []
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: With a dummy reader function (valid case of reader_func)
    def dummy_reader_func(datasets):
        datasets = datasets.shuffle(5)
        return datasets.interleave(lambda x: x, num_parallel_calls=tf.data.AUTOTUNE)

    input_dict = {
        "path": dummy_path4,
        "element_spec": [],
        "compression": "NONE",
        "reader_func": [dummy_reader_func]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Element spec with float type
    input_dict = {
        "path": dummy_path5,
        "element_spec": [tf.TensorSpec(shape=(), dtype=tf.float32, name=None)],
        "compression": "NONE",
        "reader_func": []
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Element spec with string type
    input_dict = {
        "path": dummy_path6,
        "element_spec": [tf.TensorSpec(shape=(), dtype=tf.string, name=None)],
        "compression": "NONE",
        "reader_func": []
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Element spec with multi-dimensional shape
    input_dict = {
        "path": dummy_path7,
        "element_spec": [tf.TensorSpec(shape=(2, 2), dtype=tf.int32, name=None)],
        "compression": "NONE",
        "reader_func": []
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Empty path string
    input_dict = {
        "path": "",
        "element_spec": [],
        "compression": "NONE",
        "reader_func": []
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Compression is NONE
    input_dict = {
        "path": dummy_path9,
        "element_spec": [],
        "compression": "NONE",
        "reader_func": []
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: More complex element spec
    input_dict = {
        "path": dummy_path10,
        "element_spec": [(tf.TensorSpec(shape=(), dtype=tf.int64, name=None), tf.TensorSpec(shape=(2,), dtype=tf.float32, name=None))],
        "compression": "NONE",
        "reader_func": []
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


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import os

def tf_data_experimental_load_inputs():
    list_of_inputs = []

    # Create a dummy path
    path = os.path.join(os.getcwd(), "temp_dataset")
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as e:
        path = os.path.join("/tmp", "temp_dataset")
        os.makedirs(path, exist_ok=True)

    # Input 1: Minimal valid input (no compression, no reader_func, no element_spec)
    input_dict = {
        "path": path,
        "element_spec": [],
        "compression": None,
        "reader_func": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: With GZIP compression
    input_dict = {
        "path": path,
        "element_spec": [],
        "compression": "GZIP",
        "reader_func": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: With NONE compression
    input_dict = {
        "path": path,
        "element_spec": [],
        "compression": "NONE",
        "reader_func": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: With element_spec (tf.int64)
    input_dict = {
        "path": path,
        "element_spec": [tf.TensorSpec(shape=(), dtype=tf.int64, name=None)],
        "compression": None,
        "reader_func": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: With element_spec (tf.float32)
    input_dict = {
        "path": path,
        "element_spec": [tf.TensorSpec(shape=(2, 2), dtype=tf.float32, name=None)],
        "compression": None,
        "reader_func": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: With element_spec (nested structure)
    element_spec = [
        {
            "a": tf.TensorSpec(shape=(), dtype=tf.int32, name=None),
            "b": tf.TensorSpec(shape=(3,), dtype=tf.float64, name=None),
        }
    ]
    input_dict = {
        "path": path,
        "element_spec": element_spec,
        "compression": None,
        "reader_func": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: With simple reader_func (identity)

    input_dict = {
        "path": path,
        "element_spec": [],
        "compression": None,
        "reader_func": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: With GZIP compression and element_spec
    input_dict = {
        "path": path,
        "element_spec": [tf.TensorSpec(shape=(), dtype=tf.string, name=None)],
        "compression": "GZIP",
        "reader_func": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: with element spec and reader func
    input_dict = {
        "path": path,
        "element_spec": [tf.TensorSpec(shape=(), dtype=tf.int64, name=None)],
        "compression": None,
        "reader_func": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: No compression, with element_spec and reader_func

    input_dict = {
        "path": path,
        "element_spec": [tf.TensorSpec(shape=(None,), dtype=tf.float32, name=None)],
        "compression": "NONE",
        "reader_func": None
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

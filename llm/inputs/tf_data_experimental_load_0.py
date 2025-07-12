
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy
import os

def tf_data_experimental_load_inputs():
    list_of_inputs = []

    # Helper function to create dummy data
    def create_dummy_data(path, element_spec):
      dataset = tf.data.Dataset.from_tensor_slices([element_spec['dtype'](i) for i in range(5)])
      tf.data.experimental.save(dataset, path)

    # Input 1: Basic case with default options
    path = os.path.join(os.getcwd(), "data1")
    element_spec = {'shape': (), 'dtype': np.int64, 'name': None}
    create_dummy_data(path, element_spec)
    input_dict = {
        "path": path,
        "element_spec": [],
        "compression": "NONE",
        "reader_func": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: With element_spec
    path = os.path.join(os.getcwd(), "data2")
    element_spec = {'shape': (), 'dtype': np.int32, 'name': None}
    create_dummy_data(path, element_spec)
    input_dict = {
        "path": path,
        "element_spec": [tf.TensorSpec(shape=(), dtype=tf.int32)],
        "compression": "NONE",
        "reader_func": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: With GZIP compression
    path = os.path.join(os.getcwd(), "data3")
    element_spec = {'shape': (), 'dtype': np.float32, 'name': None}
    create_dummy_data(path, element_spec)
    input_dict = {
        "path": path,
        "element_spec": [],
        "compression": "GZIP",
        "reader_func": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Empty element_spec and custom reader_func (identity)
    path = os.path.join(os.getcwd(), "data4")
    element_spec = {'shape': (2,), 'dtype': np.float64, 'name': None}
    dataset = tf.data.Dataset.from_tensor_slices(np.random.rand(5, 2))
    tf.data.experimental.save(dataset, path)

    def identity_reader_func(datasets):
      return datasets.interleave(lambda x: x, num_parallel_calls=tf.data.AUTOTUNE)

    input_dict = {
        "path": path,
        "element_spec": [],
        "compression": "NONE",
        "reader_func": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Multi-dimensional element_spec
    path = os.path.join(os.getcwd(), "data5")
    element_spec = {'shape': (2, 3), 'dtype': np.int64, 'name': None}
    dataset = tf.data.Dataset.from_tensor_slices(np.random.randint(0, 10, size=(5, 2, 3)))
    tf.data.experimental.save(dataset, path)
    input_dict = {
        "path": path,
        "element_spec": [tf.TensorSpec(shape=(2, 3), dtype=tf.int64)],
        "compression": "NONE",
        "reader_func": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Different path
    path = os.path.join(os.getcwd(), "data6")
    element_spec = {'shape': (), 'dtype': np.int8, 'name': None}
    create_dummy_data(path, element_spec)
    input_dict = {
        "path": path,
        "element_spec": [],
        "compression": "NONE",
        "reader_func": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: element_spec with a different dtype
    path = os.path.join(os.getcwd(), "data7")
    element_spec = {'shape': (), 'dtype': np.float64, 'name': None}
    create_dummy_data(path, element_spec)
    input_dict = {
        "path": path,
        "element_spec": [tf.TensorSpec(shape=(), dtype=tf.float64)],
        "compression": "NONE",
        "reader_func": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Path with subdirectories
    path = os.path.join(os.getcwd(), "subdir", "data8")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    element_spec = {'shape': (), 'dtype': np.int16, 'name': None}
    create_dummy_data(path, element_spec)
    input_dict = {
        "path": path,
        "element_spec": [],
        "compression": "NONE",
        "reader_func": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: element_spec with different shape
    path = os.path.join(os.getcwd(), "data9")
    element_spec = {'shape': (5,), 'dtype': np.int64, 'name': None}
    dataset = tf.data.Dataset.from_tensor_slices(np.random.randint(0, 10, size=(5,5)))
    tf.data.experimental.save(dataset, path)
    input_dict = {
        "path": path,
        "element_spec": [tf.TensorSpec(shape=(5,), dtype=tf.int64)],
        "compression": "NONE",
        "reader_func": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: reader_func that shuffles
    path = os.path.join(os.getcwd(), "data10")
    element_spec = {'shape': (), 'dtype': np.int64, 'name': None}
    create_dummy_data(path, element_spec)

    def shuffle_reader_func(datasets):
        datasets = datasets.shuffle(5)
        return datasets.interleave(lambda x: x, num_parallel_calls=tf.data.AUTOTUNE)

    input_dict = {
        "path": path,
        "element_spec": [],
        "compression": "NONE",
        "reader_func": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: No element_spec, custom reader func
    path = os.path.join(os.getcwd(), "data11")
    element_spec = {'shape': (), 'dtype': np.int64, 'name': None}
    create_dummy_data(path, element_spec)

    def custom_reader_func(datasets):
        datasets = datasets.shuffle(5)
        return datasets.interleave(lambda x: x, num_parallel_calls=tf.data.AUTOTUNE)

    input_dict = {
        "path": path,
        "element_spec": [],
        "compression": "NONE",
        "reader_func": None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Valid reader_func, empty element_spec
    path = os.path.join(os.getcwd(), "data12")
    element_spec = {'shape': (), 'dtype': np.int64, 'name': None}
    create_dummy_data(path, element_spec)

    def reader_func(datasets):
        return datasets.interleave(lambda x: x, num_parallel_calls=tf.data.AUTOTUNE)

    input_dict = {
        "path": path,
        "element_spec": [],
        "compression": "NONE",
        "reader_func": [reader_func][0]
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 13: Valid reader_func, non empty element_spec
    path = os.path.join(os.getcwd(), "data13")
    element_spec = {'shape': (), 'dtype': np.int64, 'name': None}
    create_dummy_data(path, element_spec)

    def reader_func(datasets):
        return datasets.interleave(lambda x: x, num_parallel_calls=tf.data.AUTOTUNE)

    input_dict = {
        "path": path,
        "element_spec": [tf.TensorSpec(shape=(), dtype=tf.int64)],
        "compression": "NONE",
        "reader_func": [reader_func][0]
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

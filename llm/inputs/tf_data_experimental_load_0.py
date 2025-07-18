
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import os
import tempfile
import copy

def tf_data_experimental_load_inputs():
    temp_dir = tempfile.mkdtemp()
    list_of_inputs = []

    # Custom reader func for shuffling
    custom_reader_func_source = [
        "def custom_reader_func(datasets):",
        "  datasets = datasets.shuffle(2)",
        "  return datasets.interleave(lambda x: x, num_parallel_calls=tf.data.AUTOTUNE)"
    ]
    
    # A no-op/default reader func to satisfy the runner's expectation of a non-empty list
    # This mimics the default behavior when reader_func is None.
    default_reader_func_source = [
        "def default_reader_func(datasets):",
        "  return datasets.interleave(lambda x: x)"
    ]

    # Case 1: Basic load, minimal arguments
    path1 = os.path.join(temp_dir, "data_1")
    dataset1 = tf.data.Dataset.range(5)
    tf.data.experimental.save(dataset1, path1)
    input_dict_1 = {
        'path': path1,
        'element_spec': [],
        'compression': '',
        'reader_func': default_reader_func_source
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Case 2: Load with GZIP compression
    path2 = os.path.join(temp_dir, "data_2")
    dataset2 = tf.data.Dataset.from_tensor_slices(np.arange(10, dtype=np.int32))
    tf.data.experimental.save(dataset2, path2, compression='GZIP')
    input_dict_2 = {
        'path': path2,
        'element_spec': [],
        'compression': 'GZIP',
        'reader_func': default_reader_func_source
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Case 3: Load with an explicit, simple element_spec
    path3 = os.path.join(temp_dir, "data_3")
    dataset3 = tf.data.Dataset.range(3, output_type=tf.float32)
    tf.data.experimental.save(dataset3, path3)
    input_dict_3 = {
        'path': path3,
        'element_spec': [{'shape': [], 'dtype': 'float32'}],
        'compression': '',
        'reader_func': default_reader_func_source
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Case 4: Load a dataset with a tuple structure
    path4 = os.path.join(temp_dir, "data_4")
    dataset4 = tf.data.Dataset.from_tensors((np.array([10, 20], dtype=np.int16), np.array(3.14, dtype=np.float64)))
    tf.data.experimental.save(dataset4, path4)
    input_dict_4 = {
        'path': path4,
        'element_spec': [
            {'shape': [2], 'dtype': 'int16'},
            {'shape': [], 'dtype': 'float64'}
        ],
        'compression': '',
        'reader_func': default_reader_func_source
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Case 5: Load a dataset with string and boolean types
    path5 = os.path.join(temp_dir, "data_5")
    dataset5 = tf.data.Dataset.from_tensor_slices(([b"hello", b"world"], [True, False]))
    tf.data.experimental.save(dataset5, path5)
    input_dict_5 = {
        'path': path5,
        'element_spec': [
            {'shape': [], 'dtype': 'string'},
            {'shape': [], 'dtype': 'bool'}
        ],
        'compression': '',
        'reader_func': default_reader_func_source
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Case 6: Load a dataset with multi-dimensional tensors and GZIP
    path6 = os.path.join(temp_dir, "data_6")
    dataset6 = tf.data.Dataset.from_tensors(np.zeros((2, 3, 4), dtype=np.uint8))
    tf.data.experimental.save(dataset6, path6, compression='GZIP')
    input_dict_6 = {
        'path': path6,
        'element_spec': [{'shape': [2, 3, 4], 'dtype': 'uint8'}],
        'compression': 'GZIP',
        'reader_func': default_reader_func_source
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Case 7: Load using a custom reader_func on a sharded dataset
    path7 = os.path.join(temp_dir, "data_7")
    dataset7 = tf.data.Dataset.range(20)
    tf.data.experimental.save(dataset7, path7, shard_func=lambda x: tf.cast(x % 2, tf.int64))
    input_dict_7 = {
        'path': path7,
        'element_spec': [],
        'compression': '',
        'reader_func': custom_reader_func_source
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Case 8: Load with all arguments specified
    path8 = os.path.join(temp_dir, "data_8")
    randint_dtype = 'int64'
    dataset8 = tf.data.Dataset.from_tensor_slices(
        (np.random.rand(10, 2).astype(np.float32), np.random.randint(0, 5, size=(10,)).astype(randint_dtype))
    )
    tf.data.experimental.save(dataset8, path8, compression='GZIP', shard_func=lambda x, y: tf.cast(y % 2, tf.int64))
    input_dict_8 = {
        'path': path8,
        'element_spec': [
            {'shape': [2], 'dtype': 'float32'},
            {'shape': [], 'dtype': randint_dtype}
        ],
        'compression': 'GZIP',
        'reader_func': custom_reader_func_source
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Case 9: Load a dataset with complex numbers
    path9 = os.path.join(temp_dir, "data_9")
    dataset9 = tf.data.Dataset.from_tensors(np.array([1 + 2j, 3 - 4j], dtype=np.complex128))
    tf.data.experimental.save(dataset9, path9)
    input_dict_9 = {
        'path': path9,
        'element_spec': [{'shape': [2], 'dtype': 'complex128'}],
        'compression': '',
        'reader_func': default_reader_func_source
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Case 10: Load a dataset with a nested tuple structure
    path10 = os.path.join(temp_dir, "data_10")
    dataset10 = tf.data.Dataset.from_tensors(
        ((np.int8(-5), np.float16(3.5)), np.array(b'nested_element'))
    )
    tf.data.experimental.save(dataset10, path10)
    input_dict_10 = {
        'path': path10,
        'element_spec': [
            [{'shape': [], 'dtype': 'int8'}, {'shape': [], 'dtype': 'float16'}],
            {'shape': [], 'dtype': 'string'}
        ],
        'compression': '',
        'reader_func': default_reader_func_source
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.data.experimental.load"] = tf_data_experimental_load_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.data.experimental.load' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.load'.")

check_valid('tf.data.experimental.load', generated_inputs['tf.data.experimental.load'], lib="tf", suffix=0)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import os
import tempfile
import copy

# Dummy reader functions and shard function to be passed as callables. These need to be defined at the top level.
AUTOTUNE = tf.data.AUTOTUNE

def custom_reader_func(datasets):
    datasets = datasets.shuffle(buffer_size=4)
    return datasets.interleave(lambda x: x, num_parallel_calls=AUTOTUNE)

def another_reader_func(datasets):
    return datasets.interleave(lambda x: x, cycle_length=2, num_parallel_calls=AUTOTUNE)

def shard_func_for_tensors(tensor):
    if tensor.dtype.is_complex:
        tensor = tf.math.real(tensor)
    if tensor.dtype.is_floating or tensor.dtype.is_integer:
        return tf.cast(tf.reduce_sum(tf.cast(tensor, tf.float32)), tf.int64) % 2
    else:
        # Handle other types like string by hashing
        flat_tensor = tf.nest.flatten(tf.strings.as_string(tensor))
        joined_string = tf.strings.reduce_join(flat_tensor)
        return tf.strings.to_hash_bucket(joined_string, 2)


def tf_data_experimental_load_inputs():
    list_of_inputs = []
    
    try:
        tempdir = tempfile.mkdtemp(prefix="tf_load_test_")
    except (IOError, OSError):
        tempdir = os.path.join(tempfile.gettempdir(), "tf_load_test")
        os.makedirs(tempdir, exist_ok=True)


    def get_path(name):
        return os.path.join(tempdir, name)

    # Case 1: Basic case with scalar int64
    path1 = get_path("dataset1")
    tf.data.experimental.save(tf.data.Dataset.range(5), path1)
    input_dict_1 = {
        'path': path1,
        'element_spec': [{'shape': [], 'dtype': np.int64}],
        'compression': 'NONE',
        'reader_func': [],
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Case 2: GZIP compression with float32
    path2 = get_path("dataset2")
    tf.data.experimental.save(tf.data.Dataset.from_tensor_slices(np.arange(10, dtype=np.float32)), path2, compression='GZIP')
    input_dict_2 = {
        'path': path2,
        'element_spec': [{'shape': [], 'dtype': np.float32}],
        'compression': 'GZIP',
        'reader_func': [],
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Case 3: Vector elements
    path3 = get_path("dataset3")
    tf.data.experimental.save(tf.data.Dataset.from_tensor_slices(np.random.rand(5, 8).astype(np.float64)), path3)
    input_dict_3 = {
        'path': path3,
        'element_spec': [{'shape': [8], 'dtype': np.float64}],
        'compression': 'NONE',
        'reader_func': [],
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Case 4: Matrix elements
    path4 = get_path("dataset4")
    tf.data.experimental.save(tf.data.Dataset.from_tensor_slices(np.random.rand(3, 4, 5).astype(np.float32)), path4, compression='GZIP')
    input_dict_4 = {
        'path': path4,
        'element_spec': [{'shape': [4, 5], 'dtype': np.float32}],
        'compression': 'GZIP',
        'reader_func': [],
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Case 5: Spec with an unknown dimension
    path5 = get_path("dataset5")
    tf.data.experimental.save(tf.data.Dataset.from_tensor_slices(np.random.rand(7, 10).astype(np.float32)), path5)
    input_dict_5 = {
        'path': path5,
        'element_spec': [{'shape': [None], 'dtype': np.float32}],
        'compression': 'NONE',
        'reader_func': [],
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Case 6: Tuple elements
    path6 = get_path("dataset6")
    dataset6 = tf.data.Dataset.zip((tf.data.Dataset.range(5, dtype=tf.int32), tf.data.Dataset.from_tensor_slices(np.random.rand(5, 3))))
    tf.data.experimental.save(dataset6, path6)
    input_dict_6 = {
        'path': path6,
        'element_spec': [
            {'shape': [], 'dtype': np.int32},
            {'shape': [3], 'dtype': np.float64}
        ],
        'compression': 'NONE',
        'reader_func': [],
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))
    
    # Case 7: With a reader_func and sharding
    path7 = get_path("dataset7")
    tf.data.experimental.save(tf.data.Dataset.range(20), path7, shard_func=lambda i: i % 4)
    input_dict_7 = {
        'path': path7,
        'element_spec': [{'shape': [], 'dtype': np.int64}],
        'compression': 'NONE',
        'reader_func': [custom_reader_func],
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Case 8: Another reader_func with GZIP and multiple unknown dims
    path8 = get_path("dataset8")
    dataset8 = tf.data.Dataset.from_tensor_slices(np.random.rand(15, 2, 2).astype(np.float32))
    tf.data.experimental.save(dataset8, path8, compression='GZIP', shard_func=shard_func_for_tensors)
    input_dict_8 = {
        'path': path8,
        'element_spec': [{'shape': [None, None], 'dtype': np.float32}],
        'compression': 'GZIP',
        'reader_func': [another_reader_func],
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Case 9: String tensors
    path9 = get_path("dataset9")
    tf.data.experimental.save(tf.data.Dataset.from_tensor_slices(["alpha", "beta", "gamma"]), path9)
    input_dict_9 = {
        'path': path9,
        'element_spec': [{'shape': [], 'dtype': 'string'}],
        'compression': 'NONE',
        'reader_func': [],
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Case 10: Nested tuple elements
    path10 = get_path("dataset10")
    d1 = tf.data.Dataset.range(5)
    d2 = tf.data.Dataset.from_tensor_slices(np.random.rand(5, 2).astype(np.float32))
    d3 = tf.data.Dataset.from_tensor_slices(np.array(['a', 'b', 'c', 'd', 'e']))
    tf.data.experimental.save(tf.data.Dataset.zip((d1, (d2, d3))), path10)
    input_dict_10 = {
        'path': path10,
        'element_spec': [
            {'shape': [], 'dtype': np.int64},
            [
                {'shape': [2], 'dtype': np.float32},
                {'shape': [], 'dtype': 'string'}
            ]
        ],
        'compression': 'NONE',
        'reader_func': [],
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Case 11: Boolean tensors
    path11 = get_path("dataset11")
    tf.data.experimental.save(tf.data.Dataset.from_tensor_slices([True, False, True]), path11)
    input_dict_11 = {
        'path': path11,
        'element_spec': [{'shape': [], 'dtype': np.bool_}],
        'compression': 'NONE',
        'reader_func': [],
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))
    
    # Case 12: Complex numbers
    path12 = get_path("dataset12")
    tf.data.experimental.save(tf.data.Dataset.from_tensor_slices(np.array([1+2j, 3+4j], dtype=np.complex64)), path12)
    input_dict_12 = {
        'path': path12,
        'element_spec': [{'shape': [], 'dtype': np.complex64}],
        'compression': 'NONE',
        'reader_func': [],
    }
    list_of_inputs.append(copy.deepcopy(input_dict_12))

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

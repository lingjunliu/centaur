
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_map_and_batch_inputs():
    list_of_inputs = []

    # Input 1
    def map_func(x):
        return x * 2
    batch_size = tf.constant(4, dtype=tf.int64).numpy()
    num_parallel_batches = tf.constant(2, dtype=tf.int64).numpy()
    drop_remainder = tf.constant(False, dtype=tf.bool).numpy()
    num_parallel_calls = None
    input_dict = {
        'map_func': [map_func],
        'batch_size': batch_size,
        'num_parallel_batches': num_parallel_batches,
        'drop_remainder': drop_remainder,
        'num_parallel_calls': num_parallel_calls
    }
    dataset = tf.data.Dataset.range(10)
    dataset = dataset.apply(tf.data.experimental.map_and_batch(map_func=map_func, batch_size=batch_size, num_parallel_batches=num_parallel_batches, drop_remainder=drop_remainder, num_parallel_calls=num_parallel_calls))
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    def map_func(x):
        return tf.strings.to_number(x)
    batch_size = tf.constant(8, dtype=tf.int64).numpy()
    num_parallel_batches = None
    drop_remainder = tf.constant(True, dtype=tf.bool).numpy()
    num_parallel_calls = tf.constant(4, dtype=tf.int32).numpy()
    input_dict = {
        'map_func': [map_func],
        'batch_size': batch_size,
        'num_parallel_batches': num_parallel_batches,
        'drop_remainder': drop_remainder,
        'num_parallel_calls': num_parallel_calls
    }
    dataset = tf.data.Dataset.from_tensor_slices(["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])
    dataset = dataset.apply(tf.data.experimental.map_and_batch(map_func=map_func, batch_size=batch_size, num_parallel_batches=num_parallel_batches, drop_remainder=drop_remainder, num_parallel_calls=num_parallel_calls))

    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    def map_func(x):
        return tf.reshape(x, [1])
    batch_size = tf.constant(16, dtype=tf.int64).numpy()
    num_parallel_batches = tf.constant(4, dtype=tf.int64).numpy()
    drop_remainder = tf.constant(False, dtype=tf.bool).numpy()
    num_parallel_calls = None
    input_dict = {
        'map_func': [map_func],
        'batch_size': batch_size,
        'num_parallel_batches': num_parallel_batches,
        'drop_remainder': drop_remainder,
        'num_parallel_calls': num_parallel_calls
    }
    dataset = tf.data.Dataset.range(32)
    dataset = dataset.apply(tf.data.experimental.map_and_batch(map_func=map_func, batch_size=batch_size, num_parallel_batches=num_parallel_batches, drop_remainder=drop_remainder, num_parallel_calls=num_parallel_calls))

    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    def map_func(x):
        return x + 1.0
    batch_size = tf.constant(32, dtype=tf.int64).numpy()
    num_parallel_batches = None
    drop_remainder = tf.constant(True, dtype=tf.bool).numpy()
    num_parallel_calls = tf.constant(16, dtype=tf.int32).numpy()
    input_dict = {
        'map_func': [map_func],
        'batch_size': batch_size,
        'num_parallel_batches': num_parallel_batches,
        'drop_remainder': drop_remainder,
        'num_parallel_calls': num_parallel_calls
    }
    dataset = tf.data.Dataset.range(64)
    dataset = dataset.apply(tf.data.experimental.map_and_batch(map_func=map_func, batch_size=batch_size, num_parallel_batches=num_parallel_batches, drop_remainder=drop_remainder, num_parallel_calls=num_parallel_calls))

    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    def map_func(x):
        return tf.clip_by_value(x, 0, 1)
    batch_size = tf.constant(1, dtype=tf.int64).numpy()
    num_parallel_batches = tf.constant(1, dtype=tf.int64).numpy()
    drop_remainder = tf.constant(False, dtype=tf.bool).numpy()
    num_parallel_calls = None
    input_dict = {
        'map_func': [map_func],
        'batch_size': batch_size,
        'num_parallel_batches': num_parallel_batches,
        'drop_remainder': drop_remainder,
        'num_parallel_calls': num_parallel_calls
    }
    dataset = tf.data.Dataset.range(2)
    dataset = dataset.apply(tf.data.experimental.map_and_batch(map_func=map_func, batch_size=batch_size, num_parallel_batches=num_parallel_batches, drop_remainder=drop_remainder, num_parallel_calls=num_parallel_calls))

    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    def map_func(x):
        return tf.math.sin(x)
    batch_size = tf.constant(5, dtype=tf.int64).numpy()
    num_parallel_batches = None
    drop_remainder = tf.constant(True, dtype=tf.bool).numpy()
    num_parallel_calls = tf.constant(10, dtype=tf.int32).numpy()
    input_dict = {
        'map_func': [map_func],
        'batch_size': batch_size,
        'num_parallel_batches': num_parallel_batches,
        'drop_remainder': drop_remainder,
        'num_parallel_calls': num_parallel_calls
    }
    dataset = tf.data.Dataset.range(25)
    dataset = dataset.apply(tf.data.experimental.map_and_batch(map_func=map_func, batch_size=batch_size, num_parallel_batches=num_parallel_batches, drop_remainder=drop_remainder, num_parallel_calls=num_parallel_calls))

    list_of_inputs.append(copy.deepcopy(input_dict))

   # Input 7
    def map_func(x):
        return tf.image.convert_image_dtype(x, dtype=tf.float32)
    batch_size = tf.constant(7, dtype=tf.int64).numpy()
    num_parallel_batches = tf.constant(3, dtype=tf.int64).numpy()
    drop_remainder = tf.constant(False, dtype=tf.bool).numpy()
    num_parallel_calls = None
    input_dict = {
        'map_func': [map_func],
        'batch_size': batch_size,
        'num_parallel_batches': num_parallel_batches,
        'drop_remainder': drop_remainder,
        'num_parallel_calls': num_parallel_calls
    }
    dataset = tf.data.Dataset.range(21)
    dataset = dataset.apply(tf.data.experimental.map_and_batch(map_func=map_func, batch_size=batch_size, num_parallel_batches=num_parallel_batches, drop_remainder=drop_remainder, num_parallel_calls=num_parallel_calls))

    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    def map_func(x):
        return tf.expand_dims(x, axis=0)
    batch_size = tf.constant(9, dtype=tf.int64).numpy()
    num_parallel_batches = None
    drop_remainder = tf.constant(True, dtype=tf.bool).numpy()
    num_parallel_calls = tf.constant(18, dtype=tf.int32).numpy()
    input_dict = {
        'map_func': [map_func],
        'batch_size': batch_size,
        'num_parallel_batches': num_parallel_batches,
        'drop_remainder': drop_remainder,
        'num_parallel_calls': num_parallel_calls
    }
    dataset = tf.data.Dataset.range(45)
    dataset = dataset.apply(tf.data.experimental.map_and_batch(map_func=map_func, batch_size=batch_size, num_parallel_batches=num_parallel_batches, drop_remainder=drop_remainder, num_parallel_calls=num_parallel_calls))

    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    def map_func(x):
        return tf.random.normal(shape=(10,))
    batch_size = tf.constant(2, dtype=tf.int64).numpy()
    num_parallel_batches = tf.constant(1, dtype=tf.int64).numpy()
    drop_remainder = tf.constant(False, dtype=tf.bool).numpy()
    num_parallel_calls = None
    input_dict = {
        'map_func': [map_func],
        'batch_size': batch_size,
        'num_parallel_batches': num_parallel_batches,
        'drop_remainder': drop_remainder,
        'num_parallel_calls': num_parallel_calls
    }
    dataset = tf.data.Dataset.range(4)
    dataset = dataset.apply(tf.data.experimental.map_and_batch(map_func=map_func, batch_size=batch_size, num_parallel_batches=num_parallel_batches, drop_remainder=drop_remainder, num_parallel_calls=num_parallel_calls))

    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    def map_func(x):
        return tf.cast(x, tf.float32)
    batch_size = tf.constant(3, dtype=tf.int64).numpy()
    num_parallel_batches = None
    drop_remainder = tf.constant(True, dtype=tf.bool).numpy()
    num_parallel_calls = tf.constant(1, dtype=tf.int32).numpy()
    input_dict = {
        'map_func': [map_func],
        'batch_size': batch_size,
        'num_parallel_batches': num_parallel_batches,
        'drop_remainder': drop_remainder,
        'num_parallel_calls': num_parallel_calls
    }
    dataset = tf.data.Dataset.range(7)
    dataset = dataset.apply(tf.data.experimental.map_and_batch(map_func=map_func, batch_size=batch_size, num_parallel_batches=num_parallel_batches, drop_remainder=drop_remainder, num_parallel_calls=num_parallel_calls))

    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.data.experimental.map_and_batch"] = tf_data_experimental_map_and_batch_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.data.experimental.map_and_batch' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.map_and_batch'.")

check_valid('tf.data.experimental.map_and_batch', generated_inputs['tf.data.experimental.map_and_batch'], lib="tf", suffix=0)

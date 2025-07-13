
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_sample_from_datasets_inputs():
    list_of_inputs = []

    # Input 1
    datasets = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    weights = [0.5, 0.5]
    seed = tf.constant(42, dtype=tf.int64)
    stop_on_empty_dataset = False

    input_dict = {
        "datasets": datasets,
        "weights": weights,
        "seed": seed,
        "stop_on_empty_dataset": stop_on_empty_dataset
    }
    list_of_inputs.append(input_dict)

    # Input 2
    datasets = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    weights = [0.2, 0.8]
    seed = tf.constant(123, dtype=tf.int64)
    stop_on_empty_dataset = True

    input_dict = {
        "datasets": datasets,
        "weights": weights,
        "seed": seed,
        "stop_on_empty_dataset": stop_on_empty_dataset
    }
    list_of_inputs.append(input_dict)

    # Input 3
    datasets = [np.array([1, 2]), np.array([3, 4])]
    weights = [0.7, 0.3]
    seed = tf.constant(99, dtype=tf.int64)
    stop_on_empty_dataset = True

    input_dict = {
        "datasets": datasets,
        "weights": weights,
        "seed": seed,
        "stop_on_empty_dataset": stop_on_empty_dataset
    }
    list_of_inputs.append(input_dict)

    # Input 4: Seed 0
    datasets = [np.array([10, 20]), np.array([30, 40])]
    weights = [0.4, 0.6]
    seed = tf.constant(0, dtype=tf.int64)
    stop_on_empty_dataset = False

    input_dict = {
        "datasets": datasets,
        "weights": weights,
        "seed": seed,
        "stop_on_empty_dataset": stop_on_empty_dataset
    }
    list_of_inputs.append(input_dict)

    # Input 5: different size datasets
    datasets = [np.array([1]), np.array([2, 3, 4])]
    weights = [0.9, 0.1]
    seed = tf.constant(1, dtype=tf.int64)
    stop_on_empty_dataset = True

    input_dict = {
        "datasets": datasets,
        "weights": weights,
        "seed": seed,
        "stop_on_empty_dataset": stop_on_empty_dataset
    }
    list_of_inputs.append(input_dict)

    # Input 6
    datasets = [np.array([1, 2, 3], dtype=np.float32), np.array([4, 5, 6], dtype=np.float32)]
    weights = [0.5, 0.5]
    seed = tf.constant(42, dtype=tf.int64)
    stop_on_empty_dataset = False

    input_dict = {
        "datasets": datasets,
        "weights": weights,
        "seed": seed,
        "stop_on_empty_dataset": stop_on_empty_dataset
    }
    list_of_inputs.append(input_dict)

    # Input 7
    datasets = [np.array([1, 2, 3], dtype=np.int64), np.array([4, 5, 6], dtype=np.int64)]
    weights = [0.2, 0.8]
    seed = tf.constant(123, dtype=tf.int64)
    stop_on_empty_dataset = True

    input_dict = {
        "datasets": datasets,
        "weights": weights,
        "seed": seed,
        "stop_on_empty_dataset": stop_on_empty_dataset
    }
    list_of_inputs.append(input_dict)

     # Input 8 different dtypes
    datasets = [np.array([1, 2], dtype=np.int32), np.array([3, 4], dtype=np.float64)]
    weights = [0.7, 0.3]
    seed = tf.constant(99, dtype=tf.int64)
    stop_on_empty_dataset = True

    input_dict = {
        "datasets": datasets,
        "weights": weights,
        "seed": seed,
        "stop_on_empty_dataset": stop_on_empty_dataset
    }
    list_of_inputs.append(input_dict)

    # Input 9 datasets with different ranks
    datasets = [np.array([1, 2]), np.array([[3, 4], [5,6]])]
    weights = [0.4, 0.6]
    seed = tf.constant(0, dtype=tf.int64)
    stop_on_empty_dataset = False

    input_dict = {
        "datasets": datasets,
        "weights": weights,
        "seed": seed,
        "stop_on_empty_dataset": stop_on_empty_dataset
    }
    list_of_inputs.append(input_dict)

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.data.experimental.sample_from_datasets"] = tf_data_experimental_sample_from_datasets_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.data.experimental.sample_from_datasets' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.sample_from_datasets'.")

check_valid('tf.data.experimental.sample_from_datasets', generated_inputs['tf.data.experimental.sample_from_datasets'], lib="tf", suffix=0)

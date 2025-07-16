
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_sample_from_datasets_inputs():
    list_of_inputs = []

    # Input 1
    dataset1 = tf.data.Dataset.from_tensor_slices(np.array([1, 2, 3]))
    dataset2 = tf.data.Dataset.from_tensor_slices(np.array([4, 5, 6]))
    datasets = [dataset1, dataset2]
    weights = np.array([0.5, 0.5], dtype=np.float32)
    seed = tf.constant(42, dtype=tf.int64)
    stop_on_empty_dataset = False

    input_dict = {
        "datasets": datasets,
        "weights": weights.tolist(),
        "seed": seed,
        "stop_on_empty_dataset": stop_on_empty_dataset
    }
    list_of_inputs.append(input_dict)

    # Input 2
    dataset1 = tf.data.Dataset.from_tensor_slices(np.array([1, 2, 3]))
    dataset2 = tf.data.Dataset.from_tensor_slices(np.array([4, 5, 6]))
    dataset3 = tf.data.Dataset.from_tensor_slices(np.array([7, 8, 9]))
    datasets = [dataset1, dataset2, dataset3]
    weights = np.array([0.3, 0.3, 0.4], dtype=np.float32)
    seed = tf.constant(123, dtype=tf.int64)
    stop_on_empty_dataset = True

    input_dict = {
        "datasets": datasets,
        "weights": weights.tolist(),
        "seed": seed,
        "stop_on_empty_dataset": stop_on_empty_dataset
    }
    list_of_inputs.append(input_dict)

    # Input 3
    dataset1 = tf.data.Dataset.from_tensor_slices(np.array([1, 2]))
    dataset2 = tf.data.Dataset.from_tensor_slices(np.array([3, 4, 5, 6]))
    datasets = [dataset1, dataset2]
    weights = np.array([0.8, 0.2], dtype=np.float32)
    seed = tf.constant(0, dtype=tf.int64)
    stop_on_empty_dataset = False

    input_dict = {
        "datasets": datasets,
        "weights": weights.tolist(),
        "seed": seed,
        "stop_on_empty_dataset": stop_on_empty_dataset
    }
    list_of_inputs.append(input_dict)

    # Input 4
    dataset1 = tf.data.Dataset.from_tensor_slices(np.array([1, 2, 3, 4]))
    dataset2 = tf.data.Dataset.from_tensor_slices(np.array([5, 6]))
    datasets = [dataset1, dataset2]
    weights = np.array([0.1, 0.9], dtype=np.float32)
    seed = tf.constant(-1, dtype=tf.int64)
    stop_on_empty_dataset = True

    input_dict = {
        "datasets": datasets,
        "weights": weights.tolist(),
        "seed": seed,
        "stop_on_empty_dataset": stop_on_empty_dataset
    }
    list_of_inputs.append(input_dict)

     # Input 5
    dataset1 = tf.data.Dataset.from_tensor_slices(np.array([1, 2]))
    dataset2 = tf.data.Dataset.from_tensor_slices(np.array([3, 4, 5, 6]))
    datasets = [dataset1, dataset2]
    weights = np.array([0.2, 0.8], dtype=np.float32)
    seed = tf.constant(1000, dtype=tf.int64)
    stop_on_empty_dataset = False

    input_dict = {
        "datasets": datasets,
        "weights": weights.tolist(),
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

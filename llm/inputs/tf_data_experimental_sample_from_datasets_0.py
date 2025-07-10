
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_sample_from_datasets_inputs():
    list_of_inputs = []

    # Helper function to convert datasets to lists
    def dataset_to_list(dataset):
        return list(dataset.as_numpy_iterator())

    # Input 1: Basic test with two datasets and uniform weights
    dataset1 = tf.data.Dataset.from_tensor_slices([1, 2, 3])
    dataset2 = tf.data.Dataset.from_tensor_slices([4, 5, 6])
    datasets = [np.array(dataset_to_list(dataset1)), np.array(dataset_to_list(dataset2))]
    weights = np.array([0.5, 0.5], dtype=np.float32)
    seed = tf.constant(10, dtype=tf.int64)
    stop_on_empty_dataset = False
    input_dict = {'datasets': datasets, 'weights': weights, 'seed': seed, 'stop_on_empty_dataset': stop_on_empty_dataset}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Two datasets, different weights
    dataset1 = tf.data.Dataset.from_tensor_slices([1, 2, 3])
    dataset2 = tf.data.Dataset.from_tensor_slices([4, 5, 6])
    datasets = [np.array(dataset_to_list(dataset1)), np.array(dataset_to_list(dataset2))]
    weights = np.array([0.8, 0.2], dtype=np.float32)
    seed = tf.constant(20, dtype=tf.int64)
    stop_on_empty_dataset = True
    input_dict = {'datasets': datasets, 'weights': weights, 'seed': seed, 'stop_on_empty_dataset': stop_on_empty_dataset}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Three datasets, uniform weights
    dataset1 = tf.data.Dataset.from_tensor_slices([1, 2, 3])
    dataset2 = tf.data.Dataset.from_tensor_slices([4, 5, 6])
    dataset3 = tf.data.Dataset.from_tensor_slices([7, 8, 9])
    datasets = [np.array(dataset_to_list(dataset1)), np.array(dataset_to_list(dataset2)), np.array(dataset_to_list(dataset3))]
    weights = np.array([1/3, 1/3, 1/3], dtype=np.float32)
    seed = tf.constant(30, dtype=tf.int64)
    stop_on_empty_dataset = False
    input_dict = {'datasets': datasets, 'weights': weights, 'seed': seed, 'stop_on_empty_dataset': stop_on_empty_dataset}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Three datasets, different weights
    dataset1 = tf.data.Dataset.from_tensor_slices([1, 2, 3])
    dataset2 = tf.data.Dataset.from_tensor_slices([4, 5, 6])
    dataset3 = tf.data.Dataset.from_tensor_slices([7, 8, 9])
    datasets = [np.array(dataset_to_list(dataset1)), np.array(dataset_to_list(dataset2)), np.array(dataset_to_list(dataset3))]
    weights = np.array([0.2, 0.5, 0.3], dtype=np.float32)
    seed = tf.constant(40, dtype=tf.int64)
    stop_on_empty_dataset = True
    input_dict = {'datasets': datasets, 'weights': weights, 'seed': seed, 'stop_on_empty_dataset': stop_on_empty_dataset}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Datasets with different shapes
    dataset1 = tf.data.Dataset.from_tensor_slices([[1, 2], [3, 4]])
    dataset2 = tf.data.Dataset.from_tensor_slices([[5, 6], [7, 8]])
    datasets = [np.array(dataset_to_list(dataset1)), np.array(dataset_to_list(dataset2))]
    weights = np.array([0.5, 0.5], dtype=np.float32)
    seed = tf.constant(50, dtype=tf.int64)
    stop_on_empty_dataset = False
    input_dict = {'datasets': datasets, 'weights': weights, 'seed': seed, 'stop_on_empty_dataset': stop_on_empty_dataset}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Datasets with different dtypes
    dataset1 = tf.data.Dataset.from_tensor_slices(np.array([1, 2, 3], dtype=np.int32))
    dataset2 = tf.data.Dataset.from_tensor_slices(np.array([4, 5, 6], dtype=np.int64))
    datasets = [np.array(dataset_to_list(dataset1)), np.array(dataset_to_list(dataset2))]
    weights = np.array([0.5, 0.5], dtype=np.float32)
    seed = tf.constant(60, dtype=tf.int64)
    stop_on_empty_dataset = True
    input_dict = {'datasets': datasets, 'weights': weights, 'seed': seed, 'stop_on_empty_dataset': stop_on_empty_dataset}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Empty dataset
    dataset1 = tf.data.Dataset.from_tensor_slices([1, 2, 3])
    dataset2 = tf.data.Dataset.from_tensor_slices([])
    datasets = [np.array(dataset_to_list(dataset1)), np.array(dataset_to_list(dataset2))]
    weights = np.array([0.5, 0.5], dtype=np.float32)
    seed = tf.constant(70, dtype=tf.int64)
    stop_on_empty_dataset = True
    input_dict = {'datasets': datasets, 'weights': weights, 'seed': seed, 'stop_on_empty_dataset': stop_on_empty_dataset}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Seed is 0
    dataset1 = tf.data.Dataset.from_tensor_slices([1, 2, 3])
    dataset2 = tf.data.Dataset.from_tensor_slices([4, 5, 6])
    datasets = [np.array(dataset_to_list(dataset1)), np.array(dataset_to_list(dataset2))]
    weights = np.array([0.5, 0.5], dtype=np.float32)
    seed = tf.constant(0, dtype=tf.int64)
    stop_on_empty_dataset = False
    input_dict = {'datasets': datasets, 'weights': weights, 'seed': seed, 'stop_on_empty_dataset': stop_on_empty_dataset}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Larger seed value
    dataset1 = tf.data.Dataset.from_tensor_slices([1, 2, 3])
    dataset2 = tf.data.Dataset.from_tensor_slices([4, 5, 6])
    datasets = [np.array(dataset_to_list(dataset1)), np.array(dataset_to_list(dataset2))]
    weights = np.array([0.5, 0.5], dtype=np.float32)
    seed = tf.constant(1000000000, dtype=tf.int64)
    stop_on_empty_dataset = True
    input_dict = {'datasets': datasets, 'weights': weights, 'seed': seed, 'stop_on_empty_dataset': stop_on_empty_dataset}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Datasets with string
    dataset1 = tf.data.Dataset.from_tensor_slices(["a", "b", "c"])
    dataset2 = tf.data.Dataset.from_tensor_slices(["d", "e", "f"])
    datasets = [np.array(dataset_to_list(dataset1)), np.array(dataset_to_list(dataset2))]
    weights = np.array([0.5, 0.5], dtype=np.float32)
    seed = tf.constant(80, dtype=tf.int64)
    stop_on_empty_dataset = False
    input_dict = {'datasets': datasets, 'weights': weights, 'seed': seed, 'stop_on_empty_dataset': stop_on_empty_dataset}
    list_of_inputs.append(copy.deepcopy(input_dict))

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

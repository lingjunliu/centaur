
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_sample_from_datasets_inputs():
    list_of_inputs = []

    # Input 1
    dataset1 = np.array([1, 2, 3])
    dataset2 = np.array([4, 5, 6])
    datasets = [tf.data.Dataset.from_tensor_slices(dataset1), tf.data.Dataset.from_tensor_slices(dataset2)]
    weights = np.array([0.5, 0.5], dtype=np.float32)
    seed = tf.constant(42, dtype=tf.int64)
    stop_on_empty_dataset = False
    input_dict = {"datasets": datasets, "weights": weights.tolist(), "seed": seed, "stop_on_empty_dataset": stop_on_empty_dataset}
    list_of_inputs.append(input_dict)

    # Input 2
    dataset1 = np.array([1, 2])
    dataset2 = np.array([3, 4])
    datasets = [tf.data.Dataset.from_tensor_slices(dataset1), tf.data.Dataset.from_tensor_slices(dataset2)]
    weights = np.array([0.2, 0.8], dtype=np.float32)
    seed = tf.constant(123, dtype=tf.int64)
    stop_on_empty_dataset = True
    input_dict = {"datasets": datasets, "weights": weights.tolist(), "seed": seed, "stop_on_empty_dataset": stop_on_empty_dataset}
    list_of_inputs.append(input_dict)

    # Input 3
    dataset1 = np.array([7, 8, 9, 10])
    dataset2 = np.array([11, 12])
    datasets = [tf.data.Dataset.from_tensor_slices(dataset1), tf.data.Dataset.from_tensor_slices(dataset2)]
    weights = np.array([0.7, 0.3], dtype=np.float32)
    seed = tf.constant(0, dtype=tf.int64)
    stop_on_empty_dataset = False
    input_dict = {"datasets": datasets, "weights": weights.tolist(), "seed": seed, "stop_on_empty_dataset": stop_on_empty_dataset}
    list_of_inputs.append(input_dict)

    # Input 4
    dataset1 = np.array([13])
    dataset2 = np.array([14, 15, 16, 17, 18])
    datasets = [tf.data.Dataset.from_tensor_slices(dataset1), tf.data.Dataset.from_tensor_slices(dataset2)]
    weights = np.array([0.1, 0.9], dtype=np.float32)
    seed = tf.constant(-1, dtype=tf.int64)
    stop_on_empty_dataset = True
    input_dict = {"datasets": datasets, "weights": weights.tolist(), "seed": seed, "stop_on_empty_dataset": stop_on_empty_dataset}
    list_of_inputs.append(input_dict)

    # Input 5
    dataset1 = np.array([-1, -2])
    dataset2 = np.array([-3, -4])
    datasets = [tf.data.Dataset.from_tensor_slices(dataset1), tf.data.Dataset.from_tensor_slices(dataset2)]
    weights = np.array([0.6, 0.4], dtype=np.float32)
    seed = tf.constant(2**31-1, dtype=tf.int64)
    stop_on_empty_dataset = False
    input_dict = {"datasets": datasets, "weights": weights.tolist(), "seed": seed, "stop_on_empty_dataset": stop_on_empty_dataset}
    list_of_inputs.append(input_dict)

    # Input 6: Three datasets
    dataset1 = np.array([100, 101])
    dataset2 = np.array([200, 201])
    dataset3 = np.array([300, 301])
    datasets = [tf.data.Dataset.from_tensor_slices(dataset1), tf.data.Dataset.from_tensor_slices(dataset2), tf.data.Dataset.from_tensor_slices(dataset3)]
    weights = np.array([0.3, 0.3, 0.4], dtype=np.float32)
    seed = tf.constant(50, dtype=tf.int64)
    stop_on_empty_dataset = True
    input_dict = {"datasets": datasets, "weights": weights.tolist(), "seed": seed, "stop_on_empty_dataset": stop_on_empty_dataset}
    list_of_inputs.append(input_dict)

    # Input 7: Different data types in datasets
    dataset1 = np.array([1.0, 2.0], dtype=np.float32)
    dataset2 = np.array([3.0, 4.0], dtype=np.float32)
    datasets = [tf.data.Dataset.from_tensor_slices(dataset1), tf.data.Dataset.from_tensor_slices(dataset2)]
    weights = np.array([0.5, 0.5], dtype=np.float32)
    seed = tf.constant(5, dtype=tf.int64)
    stop_on_empty_dataset = False
    input_dict = {"datasets": datasets, "weights": weights.tolist(), "seed": seed, "stop_on_empty_dataset": stop_on_empty_dataset}
    list_of_inputs.append(input_dict)

    # Input 8: Datasets with string data
    dataset1 = np.array(["a", "b"])
    dataset2 = np.array(["c", "d"])
    datasets = [tf.data.Dataset.from_tensor_slices(dataset1), tf.data.Dataset.from_tensor_slices(dataset2)]
    weights = np.array([0.5, 0.5], dtype=np.float32)
    seed = tf.constant(10, dtype=tf.int64)
    stop_on_empty_dataset = True
    input_dict = {"datasets": datasets, "weights": weights.tolist(), "seed": seed, "stop_on_empty_dataset": stop_on_empty_dataset}
    list_of_inputs.append(input_dict)

    # Input 9: Datasets with different lengths and non-uniform weights
    dataset1 = np.array([1, 2, 3, 4, 5])
    dataset2 = np.array([6, 7])
    datasets = [tf.data.Dataset.from_tensor_slices(dataset1), tf.data.Dataset.from_tensor_slices(dataset2)]
    weights = np.array([0.8, 0.2], dtype=np.float32)
    seed = tf.constant(15, dtype=tf.int64)
    stop_on_empty_dataset = False
    input_dict = {"datasets": datasets, "weights": weights.tolist(), "seed": seed, "stop_on_empty_dataset": stop_on_empty_dataset}
    list_of_inputs.append(input_dict)

    # Input 10: More than 2 datasets
    dataset1 = np.array([1])
    dataset2 = np.array([2])
    dataset3 = np.array([3])
    dataset4 = np.array([4])
    datasets = [tf.data.Dataset.from_tensor_slices(dataset1), tf.data.Dataset.from_tensor_slices(dataset2), tf.data.Dataset.from_tensor_slices(dataset3), tf.data.Dataset.from_tensor_slices(dataset4)]
    weights = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
    seed = tf.constant(20, dtype=tf.int64)
    stop_on_empty_dataset = True
    input_dict = {"datasets": datasets, "weights": weights.tolist(), "seed": seed, "stop_on_empty_dataset": stop_on_empty_dataset}
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


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np

def tf_data_experimental_choose_from_datasets_inputs():
    list_of_inputs = []

    # Input 1: Basic example
    datasets_np = [np.array([1, 2, 3]),
                np.array([4, 5, 6])]
    datasets = [tf.data.Dataset.from_tensor_slices(x) for x in datasets_np]
    choice_dataset = tf.data.Dataset.from_tensor_slices(np.array([0, 1, 0]))
    stop_on_empty_dataset = False
    input_dict = {"datasets": datasets, "choice_dataset": choice_dataset, "stop_on_empty_dataset": stop_on_empty_dataset}
    list_of_inputs.append(input_dict)

    # Input 2: Different length datasets
    datasets_np = [np.array([1, 2]),
                np.array([4, 5, 6, 7])]
    datasets = [tf.data.Dataset.from_tensor_slices(x) for x in datasets_np]
    choice_dataset = tf.data.Dataset.from_tensor_slices(np.array([0, 1, 0, 1]))
    stop_on_empty_dataset = True
    input_dict = {"datasets": datasets, "choice_dataset": choice_dataset, "stop_on_empty_dataset": stop_on_empty_dataset}
    list_of_inputs.append(input_dict)

    # Input 3: More datasets
    datasets_np = [np.array([1]),
                np.array([2]),
                np.array([3])]
    datasets = [tf.data.Dataset.from_tensor_slices(x) for x in datasets_np]
    choice_dataset = tf.data.Dataset.from_tensor_slices(np.array([0, 1, 2]))
    stop_on_empty_dataset = False
    input_dict = {"datasets": datasets, "choice_dataset": choice_dataset, "stop_on_empty_dataset": stop_on_empty_dataset}
    list_of_inputs.append(input_dict)

    # Input 4: Repeating choice dataset
    datasets_np = [np.array([1, 2]),
                np.array([3, 4])]
    datasets = [tf.data.Dataset.from_tensor_slices(x) for x in datasets_np]
    choice_dataset = tf.data.Dataset.from_tensor_slices(np.array([0, 1])).repeat(2)
    stop_on_empty_dataset = True
    input_dict = {"datasets": datasets, "choice_dataset": choice_dataset, "stop_on_empty_dataset": stop_on_empty_dataset}
    list_of_inputs.append(input_dict)

    # Input 5: Multi-dimensional data
    datasets_np = [np.array([[1, 2], [3, 4]]),
                np.array([[5, 6], [7, 8]])]
    datasets = [tf.data.Dataset.from_tensor_slices(x) for x in datasets_np]
    choice_dataset = tf.data.Dataset.from_tensor_slices(np.array([0, 1]))
    stop_on_empty_dataset = False
    input_dict = {"datasets": datasets, "choice_dataset": choice_dataset, "stop_on_empty_dataset": stop_on_empty_dataset}
    list_of_inputs.append(input_dict)

    # Input 6: Empty dataset
    datasets_np = [np.array([]),
                np.array([1, 2])]
    datasets = [tf.data.Dataset.from_tensor_slices(x) for x in datasets_np]
    choice_dataset = tf.data.Dataset.from_tensor_slices(np.array([0, 1]))
    stop_on_empty_dataset = True
    input_dict = {"datasets": datasets, "choice_dataset": choice_dataset, "stop_on_empty_dataset": stop_on_empty_dataset}
    list_of_inputs.append(input_dict)

    # Input 7: More complex choice
    datasets_np = [np.array([1, 2, 3]),
                np.array([4, 5, 6])]
    datasets = [tf.data.Dataset.from_tensor_slices(x) for x in datasets_np]
    choice_dataset = tf.data.Dataset.from_tensor_slices(np.array([0, 0, 1, 1, 0]))
    stop_on_empty_dataset = False
    input_dict = {"datasets": datasets, "choice_dataset": choice_dataset, "stop_on_empty_dataset": stop_on_empty_dataset}
    list_of_inputs.append(input_dict)

    # Input 8: Three datasets with more complex choice
    datasets_np = [np.array([1]),
                np.array([2]),
                np.array([3])]
    datasets = [tf.data.Dataset.from_tensor_slices(x) for x in datasets_np]
    choice_dataset = tf.data.Dataset.from_tensor_slices(np.array([0, 1, 2, 0, 1, 2]))
    stop_on_empty_dataset = True
    input_dict = {"datasets": datasets, "choice_dataset": choice_dataset, "stop_on_empty_dataset": stop_on_empty_dataset}
    list_of_inputs.append(input_dict)

    # Input 9: Nested datasets with various shapes
    datasets_np = [np.array([[1, 2], [3, 4]]),
                np.array([5, 6])]
    datasets = [tf.data.Dataset.from_tensor_slices(x) for x in datasets_np]
    choice_dataset = tf.data.Dataset.from_tensor_slices(np.array([0, 1]))
    stop_on_empty_dataset = False
    input_dict = {"datasets": datasets, "choice_dataset": choice_dataset, "stop_on_empty_dataset": stop_on_empty_dataset}
    list_of_inputs.append(input_dict)

    # Input 10: Another example with repeating and different shapes
    datasets_np = [np.array([1, 2]),
                np.array([[3, 4], [5, 6]])]
    datasets = [tf.data.Dataset.from_tensor_slices(x) for x in datasets_np]
    choice_dataset = tf.data.Dataset.from_tensor_slices(np.array([0, 1])).repeat(3)
    stop_on_empty_dataset = True
    input_dict = {"datasets": datasets, "choice_dataset": choice_dataset, "stop_on_empty_dataset": stop_on_empty_dataset}
    list_of_inputs.append(input_dict)

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.data.experimental.choose_from_datasets"] = tf_data_experimental_choose_from_datasets_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.data.experimental.choose_from_datasets' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.choose_from_datasets'.")

check_valid('tf.data.experimental.choose_from_datasets', generated_inputs['tf.data.experimental.choose_from_datasets'], lib="tf", suffix=0)

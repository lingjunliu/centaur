
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_choose_from_datasets_inputs():
    list_of_inputs = []

    # Input 1
    datasets = [tf.data.Dataset.from_tensors(np.array([1, 2, 3])).repeat(),
                tf.data.Dataset.from_tensors(np.array([4, 5, 6])).repeat()]
    choice_dataset = tf.data.Dataset.from_tensor_slices(np.array([0, 1, 0, 1])).repeat()
    stop_on_empty_dataset = False
    input_dict = {"datasets": datasets, "choice_dataset": choice_dataset, "stop_on_empty_dataset": stop_on_empty_dataset}
    list_of_inputs.append(input_dict)

    # Input 2
    datasets = [tf.data.Dataset.from_tensors(np.array([1.0, 2.0])).repeat(),
                tf.data.Dataset.from_tensors(np.array([3.0, 4.0])).repeat(),
                tf.data.Dataset.from_tensors(np.array([5.0, 6.0])).repeat()]
    choice_dataset = tf.data.Dataset.from_tensor_slices(np.array([0, 1, 2, 0, 1, 2])).repeat()
    stop_on_empty_dataset = True
    input_dict = {"datasets": datasets, "choice_dataset": choice_dataset, "stop_on_empty_dataset": stop_on_empty_dataset}
    list_of_inputs.append(input_dict)

    # Input 3
    datasets = [tf.data.Dataset.from_tensors(np.array(["a", "b"])).repeat(),
                tf.data.Dataset.from_tensors(np.array(["c", "d"])).repeat()]
    choice_dataset = tf.data.Dataset.from_tensor_slices(np.array([0, 0, 1, 1, 0])).repeat()
    stop_on_empty_dataset = False
    input_dict = {"datasets": datasets, "choice_dataset": choice_dataset, "stop_on_empty_dataset": stop_on_empty_dataset}
    list_of_inputs.append(input_dict)

    # Input 4
    datasets = [tf.data.Dataset.from_tensors(np.array([1, 2])).repeat()]
    choice_dataset = tf.data.Dataset.from_tensor_slices(np.array([0])).repeat()
    stop_on_empty_dataset = True
    input_dict = {"datasets": datasets, "choice_dataset": choice_dataset, "stop_on_empty_dataset": stop_on_empty_dataset}
    list_of_inputs.append(input_dict)

     # Input 5
    datasets = [tf.data.Dataset.from_tensors(np.array([[1, 2], [3, 4]])).repeat(),
                tf.data.Dataset.from_tensors(np.array([[5, 6], [7, 8]])).repeat()]
    choice_dataset = tf.data.Dataset.from_tensor_slices(np.array([0, 1])).repeat()
    stop_on_empty_dataset = False
    input_dict = {"datasets": datasets, "choice_dataset": choice_dataset, "stop_on_empty_dataset": stop_on_empty_dataset}
    list_of_inputs.append(input_dict)

    # Input 6
    datasets = [tf.data.Dataset.from_tensors(np.array([1])).repeat(),
                tf.data.Dataset.from_tensors(np.array([2])).repeat(),
                tf.data.Dataset.from_tensors(np.array([3])).repeat()]
    choice_dataset = tf.data.Dataset.from_tensor_slices(np.array([0, 1, 2])).repeat()
    stop_on_empty_dataset = True
    input_dict = {"datasets": datasets, "choice_dataset": choice_dataset, "stop_on_empty_dataset": stop_on_empty_dataset}
    list_of_inputs.append(input_dict)

    # Input 7
    datasets = [tf.data.Dataset.from_tensors(np.array([1, 2, 3])).repeat(),
                tf.data.Dataset.from_tensors(np.array([4, 5, 6])).repeat()]
    choice_dataset = tf.data.Dataset.from_tensor_slices(np.array([0, 1])).repeat(2)
    stop_on_empty_dataset = False
    input_dict = {"datasets": datasets, "choice_dataset": choice_dataset, "stop_on_empty_dataset": stop_on_empty_dataset}
    list_of_inputs.append(input_dict)

    # Input 8
    datasets = [tf.data.Dataset.from_tensors(np.array([1.0])).repeat(),
                tf.data.Dataset.from_tensors(np.array([2.0])).repeat(),
                tf.data.Dataset.from_tensors(np.array([3.0])).repeat()]
    choice_dataset = tf.data.Dataset.from_tensor_slices(np.array([0, 0, 1, 1, 2, 2]))
    stop_on_empty_dataset = True
    input_dict = {"datasets": datasets, "choice_dataset": choice_dataset, "stop_on_empty_dataset": stop_on_empty_dataset}
    list_of_inputs.append(input_dict)

    # Input 9
    datasets = [tf.data.Dataset.from_tensors(np.array([1, 2])).repeat(),
                tf.data.Dataset.from_tensors(np.array([3, 4])).repeat()]
    choice_dataset = tf.data.Dataset.from_tensor_slices(np.array([0, 1, 0, 1, 0])).repeat()
    stop_on_empty_dataset = False
    input_dict = {"datasets": datasets, "choice_dataset": choice_dataset, "stop_on_empty_dataset": stop_on_empty_dataset}
    list_of_inputs.append(input_dict)

    # Input 10
    datasets = [tf.data.Dataset.from_tensors(np.array([1, 2, 3])).repeat(),
                tf.data.Dataset.from_tensors(np.array([4, 5, 6])).repeat()]
    choice_dataset = tf.data.Dataset.from_tensor_slices(np.array([0, 1, 0])).repeat()
    stop_on_empty_dataset = True
    input_dict = {"datasets": datasets, "choice_dataset": choice_dataset, "stop_on_empty_dataset": stop_on_empty_dataset}
    list_of_inputs.append(input_dict)

    return list_of_inputs

generated_inputs["tf.data.experimental.choose_from_datasets"] = tf_data_experimental_choose_from_datasets_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.data.experimental.choose_from_datasets' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.choose_from_datasets'.")

check_valid('tf.data.experimental.choose_from_datasets', generated_inputs['tf.data.experimental.choose_from_datasets'], lib="tf", suffix=0)

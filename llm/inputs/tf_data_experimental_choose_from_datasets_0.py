
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_choose_from_datasets_inputs():
    list_of_inputs = []

    # Input 1: Basic case
    datasets = [tf.data.Dataset.from_tensors(np.array([[1]])).repeat().unbatch(),
                tf.data.Dataset.from_tensors(np.array([[2]])).repeat().unbatch()]
    choice_dataset = tf.data.Dataset.from_tensor_slices(np.array([0, 1, 0, 1])).repeat()
    stop_on_empty_dataset = False
    input_dict = {"datasets": datasets, "choice_dataset": choice_dataset, "stop_on_empty_dataset": stop_on_empty_dataset}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Different data types
    datasets = [tf.data.Dataset.from_tensors(np.array([[1.0]])).repeat().unbatch(),
                tf.data.Dataset.from_tensors(np.array([[2.0]])).repeat().unbatch()]
    choice_dataset = tf.data.Dataset.from_tensor_slices(np.array([0, 1, 0, 1])).repeat()
    stop_on_empty_dataset = True
    input_dict = {"datasets": datasets, "choice_dataset": choice_dataset, "stop_on_empty_dataset": stop_on_empty_dataset}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: More datasets
    datasets = [tf.data.Dataset.from_tensors(np.array([["a"]])).repeat().unbatch(),
                tf.data.Dataset.from_tensors(np.array([["b"]])).repeat().unbatch(),
                tf.data.Dataset.from_tensors(np.array([["c"]])).repeat().unbatch()]
    choice_dataset = tf.data.Dataset.from_tensor_slices(np.array([0, 1, 2, 0, 1, 2])).repeat()
    stop_on_empty_dataset = False
    input_dict = {"datasets": datasets, "choice_dataset": choice_dataset, "stop_on_empty_dataset": stop_on_empty_dataset}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Different choice dataset
    datasets = [tf.data.Dataset.from_tensors(np.array([[1, 2]])).repeat().unbatch(),
                tf.data.Dataset.from_tensors(np.array([[3, 4]])).repeat().unbatch()]
    choice_dataset = tf.data.Dataset.from_tensor_slices(np.array([0, 0, 1, 1])).repeat()
    stop_on_empty_dataset = True
    input_dict = {"datasets": datasets, "choice_dataset": choice_dataset, "stop_on_empty_dataset": stop_on_empty_dataset}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 5: Multi-dimensional tensors in datasets
    datasets = [tf.data.Dataset.from_tensors(np.array([[[1, 2], [3, 4]]])).repeat().unbatch(),
                tf.data.Dataset.from_tensors(np.array([[[5, 6], [7, 8]]])).repeat().unbatch()]
    choice_dataset = tf.data.Dataset.from_tensor_slices(np.array([0, 1])).repeat()
    stop_on_empty_dataset = False
    input_dict = {"datasets": datasets, "choice_dataset": choice_dataset, "stop_on_empty_dataset": stop_on_empty_dataset}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Larger choice dataset
    datasets = [tf.data.Dataset.from_tensors(np.array([[1]])).repeat().unbatch(),
                tf.data.Dataset.from_tensors(np.array([[2]])).repeat().unbatch()]
    choice_dataset = tf.data.Dataset.from_tensor_slices(np.array([0, 1, 0, 1, 0, 1, 0, 1])).repeat()
    stop_on_empty_dataset = True
    input_dict = {"datasets": datasets, "choice_dataset": choice_dataset, "stop_on_empty_dataset": stop_on_empty_dataset}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Datasets with strings
    datasets = [tf.data.Dataset.from_tensors(np.array([["hello"]])).repeat().unbatch(),
                tf.data.Dataset.from_tensors(np.array([["world"]])).repeat().unbatch()]
    choice_dataset = tf.data.Dataset.from_tensor_slices(np.array([0, 1])).repeat()
    stop_on_empty_dataset = False
    input_dict = {"datasets": datasets, "choice_dataset": choice_dataset, "stop_on_empty_dataset": stop_on_empty_dataset}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Different shape choice_dataset
    datasets = [tf.data.Dataset.from_tensors(np.array([[1]])).repeat().unbatch(),
                tf.data.Dataset.from_tensors(np.array([[2]])).repeat().unbatch()]
    choice_dataset = tf.data.Dataset.from_tensor_slices(np.array([[0], [1], [0], [1]])).repeat()
    choice_dataset = choice_dataset.map(lambda x: tf.squeeze(x))
    stop_on_empty_dataset = True
    input_dict = {"datasets": datasets, "choice_dataset": choice_dataset, "stop_on_empty_dataset": stop_on_empty_dataset}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Empty Datasets
    datasets = [tf.data.Dataset.from_tensors(np.array([[1]])).repeat().unbatch(),
                tf.data.Dataset.from_tensors(np.array([[2]])).repeat().unbatch()]
    choice_dataset = tf.data.Dataset.from_tensor_slices(np.array([0, 1, 0, 1])).repeat()
    stop_on_empty_dataset = False
    datasets[1] = datasets[1].filter(lambda x: tf.equal(x, tf.constant(5)))
    input_dict = {"datasets": datasets, "choice_dataset": choice_dataset, "stop_on_empty_dataset": stop_on_empty_dataset}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Larger datasets with stop_on_empty_dataset=True and different types
    datasets = [tf.data.Dataset.from_tensors(np.array([[1.5]])).repeat().unbatch(),
                tf.data.Dataset.from_tensors(np.array([["test"]])).repeat().unbatch(),
                tf.data.Dataset.from_tensors(np.array([[1,2,3]])).repeat().unbatch()]
    choice_dataset = tf.data.Dataset.from_tensor_slices(np.array([0, 1, 2, 0, 1])).repeat()
    stop_on_empty_dataset = True
    input_dict = {"datasets": datasets, "choice_dataset": choice_dataset, "stop_on_empty_dataset": stop_on_empty_dataset}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
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

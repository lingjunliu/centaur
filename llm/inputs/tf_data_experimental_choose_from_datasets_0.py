
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_choose_from_datasets_inputs():
    list_of_inputs = []

    # Input 1: Basic case
    datasets = [np.array("foo"),
                np.array("bar")]
    choice_dataset = np.array([0, 1, 0, 1])
    stop_on_empty_dataset = False
    input_dict = {"datasets": datasets, "choice_dataset": choice_dataset, "stop_on_empty_dataset": stop_on_empty_dataset}
    list_of_inputs.append(input_dict)

    # Input 2: Three datasets
    datasets = [np.array(1),
                np.array(2),
                np.array(3)]
    choice_dataset = np.array([0, 1, 2, 0, 1, 2])
    stop_on_empty_dataset = True
    input_dict = {"datasets": datasets, "choice_dataset": choice_dataset, "stop_on_empty_dataset": stop_on_empty_dataset}
    list_of_inputs.append(input_dict)

    # Input 3: Different data types in datasets
    datasets = [np.array(1.0),
                np.array("hello")]
    choice_dataset = np.array([0, 1, 0, 1])
    stop_on_empty_dataset = False
    input_dict = {"datasets": datasets, "choice_dataset": choice_dataset, "stop_on_empty_dataset": stop_on_empty_dataset}
    list_of_inputs.append(input_dict)

    # Input 4: Longer choice_dataset
    datasets = [np.array(1),
                np.array(2)]
    choice_dataset = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    stop_on_empty_dataset = True
    input_dict = {"datasets": datasets, "choice_dataset": choice_dataset, "stop_on_empty_dataset": stop_on_empty_dataset}
    list_of_inputs.append(input_dict)

    # Input 5: Empty datasets
    datasets = [np.array(1),
                np.array(2)]
    choice_dataset = np.array([0, 1, 0, 1])
    stop_on_empty_dataset = True
    input_dict = {"datasets": datasets, "choice_dataset": choice_dataset, "stop_on_empty_dataset": stop_on_empty_dataset}
    list_of_inputs.append(input_dict)

    # Input 6: Multidimensional data
    datasets = [np.array([[1, 2], [3, 4]]),
                np.array([[5, 6], [7, 8]])]
    choice_dataset = np.array([0, 1, 0, 1])
    stop_on_empty_dataset = False
    input_dict = {"datasets": datasets, "choice_dataset": choice_dataset, "stop_on_empty_dataset": stop_on_empty_dataset}
    list_of_inputs.append(input_dict)

    # Input 7: Scalar choices
    datasets = [np.array(1),
                np.array(2)]
    choice_dataset = np.array(0)
    stop_on_empty_dataset = True
    input_dict = {"datasets": datasets, "choice_dataset": choice_dataset, "stop_on_empty_dataset": stop_on_empty_dataset}
    list_of_inputs.append(input_dict)


    # Input 9: Datasets with different shapes
    datasets = [np.array([1, 2]),
                np.array([[3], [4]])]
    choice_dataset = np.array([0, 1, 0, 1])
    stop_on_empty_dataset = True
    input_dict = {"datasets": datasets, "choice_dataset": choice_dataset, "stop_on_empty_dataset": stop_on_empty_dataset}
    list_of_inputs.append(input_dict)

    # Input 10: With numpy dtypes
    datasets = [np.array(1, dtype=np.int32),
                np.array(2, dtype=np.int32)]
    choice_dataset = np.array([0, 1, 0, 1], dtype=np.int64)
    stop_on_empty_dataset = False
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

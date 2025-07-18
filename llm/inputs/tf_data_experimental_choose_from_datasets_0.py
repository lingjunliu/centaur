
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np

def get_choose_from_datasets_inputs():
    list_of_inputs = []

    # Input 1: Basic case. Datasets are wrapped in tuples to be comparable.
    datasets1 = [(np.full(5, 0, dtype=np.int32),),
                 (np.full(5, 1, dtype=np.int32),)]
    choice_dataset1 = tf.data.Dataset.from_tensor_slices(np.array([0, 1, 0, 1, 0], dtype=np.int64))
    input_dict1 = {
        'datasets': datasets1,
        'choice_dataset': choice_dataset1,
        'stop_on_empty_dataset': False
    }
    list_of_inputs.append(input_dict1)

    # Input 2: float32 data type and stop_on_empty_dataset=True
    datasets2 = [(np.array([1.1, 2.2, 3.3], dtype=np.float32),),
                 (np.array([4.4, 5.5], dtype=np.float32),)]
    choice_dataset2 = tf.data.Dataset.from_tensor_slices(np.array([0, 1, 0, 1, 0], dtype=np.int64))
    input_dict2 = {
        'datasets': datasets2,
        'choice_dataset': choice_dataset2,
        'stop_on_empty_dataset': True
    }
    list_of_inputs.append(input_dict2)

    # Input 3: One dataset is initially empty
    datasets3 = [(np.array([10, 11, 12], dtype=np.int32),),
                 (np.array([], dtype=np.int32),),
                 (np.array([20, 21, 22], dtype=np.int32),)]
    choice_dataset3 = tf.data.Dataset.from_tensor_slices(np.array([0, 2, 1, 0, 2], dtype=np.int64))
    input_dict3 = {
        'datasets': datasets3,
        'choice_dataset': choice_dataset3,
        'stop_on_empty_dataset': False
    }
    list_of_inputs.append(input_dict3)

    # Input 4: One dataset is initially empty, stop_on_empty_dataset=True
    datasets4 = [(np.arange(2, dtype=np.int32),),
                 (np.array([], dtype=np.int32),)]
    choice_dataset4 = tf.data.Dataset.from_tensor_slices(np.array([0, 0, 1], dtype=np.int64))
    input_dict4 = {
        'datasets': datasets4,
        'choice_dataset': choice_dataset4,
        'stop_on_empty_dataset': True
    }
    list_of_inputs.append(input_dict4)

    # Input 5: Datasets with multi-dimensional elements
    datasets5 = [(np.random.rand(3, 2, 2).astype(np.float32),),
                 (np.random.rand(3, 2, 2).astype(np.float32),)]
    choice_dataset5 = tf.data.Dataset.from_tensor_slices(np.array([0, 1, 0, 1, 0, 1], dtype=np.int64))
    input_dict5 = {
        'datasets': datasets5,
        'choice_dataset': choice_dataset5,
        'stop_on_empty_dataset': False
    }
    list_of_inputs.append(input_dict5)

    # Input 6: choice_dataset is empty
    datasets6 = [(np.array([1, 2, 3]),),
                 (np.array([4, 5, 6]),)]
    choice_dataset6 = tf.data.Dataset.from_tensor_slices(np.array([], dtype=np.int64))
    input_dict6 = {
        'datasets': datasets6,
        'choice_dataset': choice_dataset6,
        'stop_on_empty_dataset': False
    }
    list_of_inputs.append(input_dict6)
    
    # Input 7: Choice dataset only selects from one of the datasets
    datasets7 = [(np.array([1, 2, 3, 4]),),
                 (np.array([10, 11, 12]),),
                 (np.array([20, 21, 22, 23]),)]
    choice_dataset7 = tf.data.Dataset.from_tensor_slices(np.array([2, 2, 2, 2], dtype=np.int64))
    input_dict7 = {
        'datasets': datasets7,
        'choice_dataset': choice_dataset7,
        'stop_on_empty_dataset': False
    }
    list_of_inputs.append(input_dict7)

    # Input 8: Large number of datasets
    num_datasets = 10
    datasets8 = [(np.array([i]),) for i in range(num_datasets)]
    choice_dataset8 = tf.data.Dataset.from_tensor_slices(np.random.randint(0, num_datasets, size=20, dtype=np.int64))
    input_dict8 = {
        'datasets': datasets8,
        'choice_dataset': choice_dataset8,
        'stop_on_empty_dataset': True
    }
    list_of_inputs.append(input_dict8)

    # Input 9: Using tf.data.Dataset.range() for choice_dataset
    datasets9 = [(np.arange(10, dtype=np.int32),),
                 (np.arange(10, 20, dtype=np.int32),)]
    choice_dataset9 = tf.data.Dataset.range(2).repeat(5)
    input_dict9 = {
        'datasets': datasets9,
        'choice_dataset': choice_dataset9,
        'stop_on_empty_dataset': False
    }
    list_of_inputs.append(input_dict9)
    
    # Input 10: Datasets with structured (tuple) elements
    datasets10 = [(np.arange(5), np.arange(5, 10)),
                  (np.arange(10, 15), np.arange(15, 20))]
    choice_dataset10 = tf.data.Dataset.from_tensor_slices(np.array([0, 1, 0, 1, 0], dtype=np.int64))
    input_dict10 = {
        'datasets': datasets10,
        'choice_dataset': choice_dataset10,
        'stop_on_empty_dataset': False
    }
    list_of_inputs.append(input_dict10)

    return list_of_inputs

generated_inputs["tf.data.experimental.choose_from_datasets"] = get_choose_from_datasets_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.data.experimental.choose_from_datasets' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.choose_from_datasets'.")

check_valid('tf.data.experimental.choose_from_datasets', generated_inputs['tf.data.experimental.choose_from_datasets'], lib="tf", suffix=0)

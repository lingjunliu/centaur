
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def get_choose_from_datasets_inputs():
    list_of_inputs = []

    # The error "TypeError: cannot pickle 'Graph' object" occurs because tf.data.Dataset
    # objects are not serializable and cannot be handled by copy.deepcopy.
    # The fix is to remove the copy.deepcopy call and append the dictionaries directly.
    # A new dictionary is created for each input to prevent side effects.

    # Input 1: Basic case, int8 data
    datasets1 = [
        tf.data.Dataset.from_tensor_slices(np.array([10, 20, 30], dtype=np.int8)),
        tf.data.Dataset.from_tensor_slices(np.array([40, 50, 60], dtype=np.int8)),
        tf.data.Dataset.from_tensor_slices(np.array([70, 80, 90], dtype=np.int8))
    ]
    choice_dataset1 = tf.data.Dataset.from_tensor_slices(np.array([0, 1, 2, 0, 1, 2], dtype=np.int64))
    input_dict1 = {
        'datasets': datasets1,
        'choice_dataset': choice_dataset1,
        'stop_on_empty_dataset': False
    }
    list_of_inputs.append(input_dict1)

    # Input 2: Integer data, stop_on_empty_dataset=True
    datasets2 = [
        tf.data.Dataset.from_tensor_slices(np.array([1, 2, 3], dtype=np.int32)),
        tf.data.Dataset.from_tensor_slices(np.array([4, 5, 6], dtype=np.int32))
    ]
    choice_dataset2 = tf.data.Dataset.from_tensor_slices(np.array([0, 1, 0, 1, 0, 1], dtype=np.int64))
    input_dict2 = {
        'datasets': datasets2,
        'choice_dataset': choice_dataset2,
        'stop_on_empty_dataset': True
    }
    list_of_inputs.append(input_dict2)

    # Input 3: 2D array elements (matrices) with float data
    datasets3 = [
        tf.data.Dataset.from_tensor_slices(np.arange(12, dtype=np.float32).reshape(3, 2, 2)),
        tf.data.Dataset.from_tensor_slices(np.arange(12, 24, dtype=np.float32).reshape(3, 2, 2))
    ]
    choice_dataset3 = tf.data.Dataset.from_tensor_slices(np.array([0, 1, 1, 0, 0, 1], dtype=np.int64))
    input_dict3 = {
        'datasets': datasets3,
        'choice_dataset': choice_dataset3,
        'stop_on_empty_dataset': True
    }
    list_of_inputs.append(input_dict3)

    # Input 4: A dataset that becomes empty, stop_on_empty_dataset=True
    datasets4 = [
        tf.data.Dataset.from_tensor_slices(np.array([10, 20], dtype=np.int64)),
        tf.data.Dataset.from_tensor_slices(np.array([100, 200, 300, 400], dtype=np.int64))
    ]
    choice_dataset4 = tf.data.Dataset.from_tensor_slices(np.array([0, 1, 0, 1, 0, 1], dtype=np.int64))
    input_dict4 = {
        'datasets': datasets4,
        'choice_dataset': choice_dataset4,
        'stop_on_empty_dataset': True
    }
    list_of_inputs.append(input_dict4)

    # Input 5: A dataset that becomes empty, stop_on_empty_dataset=False
    datasets5 = [
        tf.data.Dataset.from_tensor_slices(np.array([10, 20], dtype=np.uint16)),
        tf.data.Dataset.from_tensor_slices(np.array([100, 200, 300, 400], dtype=np.uint16))
    ]
    choice_dataset5 = tf.data.Dataset.from_tensor_slices(np.array([0, 1, 0, 1, 0, 1], dtype=np.int64))
    input_dict5 = {
        'datasets': datasets5,
        'choice_dataset': choice_dataset5,
        'stop_on_empty_dataset': False
    }
    list_of_inputs.append(input_dict5)

    # Input 6: An initially empty dataset, stop_on_empty_dataset=False
    datasets6 = [
        tf.data.Dataset.from_tensor_slices(np.array([1.1, 2.2, 3.3], dtype=np.float64)),
        tf.data.Dataset.from_tensor_slices(np.array([], dtype=np.float64)),
        tf.data.Dataset.from_tensor_slices(np.array([4.4, 5.5, 6.6], dtype=np.float64))
    ]
    choice_dataset6 = tf.data.Dataset.from_tensor_slices(np.array([0, 1, 2, 0, 1, 2], dtype=np.int64))
    input_dict6 = {
        'datasets': datasets6,
        'choice_dataset': choice_dataset6,
        'stop_on_empty_dataset': False
    }
    list_of_inputs.append(input_dict6)
    
    # Input 7: An initially empty dataset, stop_on_empty_dataset=True
    datasets7 = [
        tf.data.Dataset.from_tensor_slices(np.array([1.1, 2.2], dtype=np.float32)),
        tf.data.Dataset.from_tensor_slices(np.array([], dtype=np.float32)),
        tf.data.Dataset.from_tensor_slices(np.array([4.4, 5.5], dtype=np.float32))
    ]
    choice_dataset7 = tf.data.Dataset.from_tensor_slices(np.array([1, 0, 2], dtype=np.int64))
    input_dict7 = {
        'datasets': datasets7,
        'choice_dataset': choice_dataset7,
        'stop_on_empty_dataset': True
    }
    list_of_inputs.append(input_dict7)

    # Input 8: Larger number of datasets
    datasets8 = [
        tf.data.Dataset.from_tensors(np.array(0, dtype=np.int16)),
        tf.data.Dataset.from_tensors(np.array(1, dtype=np.int16)),
        tf.data.Dataset.from_tensors(np.array(2, dtype=np.int16)),
        tf.data.Dataset.from_tensors(np.array(3, dtype=np.int16)),
        tf.data.Dataset.from_tensors(np.array(4, dtype=np.int16))
    ]
    choice_dataset8 = tf.data.Dataset.from_tensor_slices(np.array([4, 3, 2, 1, 0, 2, 4], dtype=np.int64))
    input_dict8 = {
        'datasets': datasets8,
        'choice_dataset': choice_dataset8,
        'stop_on_empty_dataset': False
    }
    list_of_inputs.append(input_dict8)

    # Input 9: Single dataset in the list
    datasets9 = [
        tf.data.Dataset.from_tensor_slices(np.array([100, 200, 300, 400, 500], dtype=np.int32))
    ]
    choice_dataset9 = tf.data.Dataset.from_tensor_slices(np.array([0, 0, 0, 0, 0], dtype=np.int64))
    input_dict9 = {
        'datasets': datasets9,
        'choice_dataset': choice_dataset9,
        'stop_on_empty_dataset': True
    }
    list_of_inputs.append(input_dict9)

    # Input 10: Boolean data type
    datasets10 = [
        tf.data.Dataset.from_tensor_slices(np.array([True, False, True])),
        tf.data.Dataset.from_tensor_slices(np.array([False, True, False]))
    ]
    choice_dataset10 = tf.data.Dataset.from_tensor_slices(np.array([0, 1, 1, 0, 1, 0], dtype=np.int64))
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

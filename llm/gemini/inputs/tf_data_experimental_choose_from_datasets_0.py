
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np

def generate_inputs_for_tf_data_experimental_choose_from_datasets():
    list_of_inputs = []

    # Case 1: Basic example
    datasets1 = [tf.data.Dataset.from_tensor_slices(np.array([1], dtype=np.int32)),
                 tf.data.Dataset.from_tensor_slices(np.array([2], dtype=np.int32)),
                 tf.data.Dataset.from_tensor_slices(np.array([3], dtype=np.int32))]
    choice_dataset1 = tf.data.Dataset.from_tensor_slices(np.array([0, 1, 2, 0, 1, 2], dtype=np.int64))
    input_dict1 = {
        'datasets': datasets1,
        'choice_dataset': choice_dataset1,
        'stop_on_empty_dataset': False
    }
    list_of_inputs.append(input_dict1)

    # Case 2: Using stop_on_empty_dataset=True
    datasets2 = [tf.data.Dataset.from_tensor_slices(np.array([100, 101], dtype=np.int32)),
                 tf.data.Dataset.from_tensor_slices(np.array([200, 201], dtype=np.int32))]
    choice_dataset2 = tf.data.Dataset.from_tensor_slices(np.array([0, 1, 1, 0, 1], dtype=np.int64))
    input_dict2 = {
        'datasets': datasets2,
        'choice_dataset': choice_dataset2,
        'stop_on_empty_dataset': True
    }
    list_of_inputs.append(input_dict2)

    # Case 3: Float data
    datasets3 = [tf.data.Dataset.from_tensor_slices(np.array([1.1, 2.2, 3.3], dtype=np.float32)),
                 tf.data.Dataset.from_tensor_slices(np.array([4.4, 5.5], dtype=np.float32))]
    choice_dataset3 = tf.data.Dataset.from_tensor_slices(np.array([0, 1, 0, 1, 0, 1, 0], dtype=np.int64))
    input_dict3 = {
        'datasets': datasets3,
        'choice_dataset': choice_dataset3,
        'stop_on_empty_dataset': False
    }
    list_of_inputs.append(input_dict3)

    # Case 4: Complex structure (tuple of tensors)
    datasets4 = [
        tf.data.Dataset.from_tensor_slices((np.arange(3, dtype=np.int16), np.arange(3, 6, dtype=np.float16))),
        tf.data.Dataset.from_tensor_slices((np.arange(10, 13, dtype=np.int16), np.arange(13, 16, dtype=np.float16)))
    ]
    choice_dataset4 = tf.data.Dataset.from_tensor_slices(np.array([0, 1, 0, 1], dtype=np.int64))
    input_dict4 = {
        'datasets': datasets4,
        'choice_dataset': choice_dataset4,
        'stop_on_empty_dataset': True
    }
    list_of_inputs.append(input_dict4)

    # Case 5: More datasets (5)
    datasets5 = [tf.data.Dataset.from_tensor_slices(np.array([i], dtype=np.int32)) for i in range(5)]
    choice_dataset5 = tf.data.Dataset.from_tensor_slices(np.tile(np.arange(5), 2).astype(np.int64))
    input_dict5 = {
        'datasets': datasets5,
        'choice_dataset': choice_dataset5,
        'stop_on_empty_dataset': False
    }
    list_of_inputs.append(input_dict5)

    # Case 6: 2D array elements
    datasets6 = [tf.data.Dataset.from_tensor_slices(np.ones((2, 2, 3), dtype=np.int16)),
                 tf.data.Dataset.from_tensor_slices(np.zeros((3, 2, 3), dtype=np.int16))]
    choice_dataset6 = tf.data.Dataset.from_tensor_slices(np.array([0, 1, 1, 0, 1], dtype=np.int64))
    input_dict6 = {
        'datasets': datasets6,
        'choice_dataset': choice_dataset6,
        'stop_on_empty_dataset': True
    }
    list_of_inputs.append(input_dict6)

    # Case 7: One dataset is empty
    datasets7 = [tf.data.Dataset.from_tensor_slices(np.array([1, 2], dtype=np.int32)),
                 tf.data.Dataset.from_tensor_slices(np.array([], dtype=np.int32)),
                 tf.data.Dataset.from_tensor_slices(np.array([3, 4], dtype=np.int32))]
    choice_dataset7 = tf.data.Dataset.from_tensor_slices(np.array([0, 2, 1, 0, 2, 1], dtype=np.int64))
    input_dict7 = {
        'datasets': datasets7,
        'choice_dataset': choice_dataset7,
        'stop_on_empty_dataset': False
    }
    list_of_inputs.append(input_dict7)

    # Case 8: All source datasets are empty
    datasets8 = [tf.data.Dataset.from_tensor_slices(np.array([], dtype=np.float64)),
                 tf.data.Dataset.from_tensor_slices(np.array([], dtype=np.float64))]
    choice_dataset8 = tf.data.Dataset.from_tensor_slices(np.array([0, 1, 0, 1], dtype=np.int64))
    input_dict8 = {
        'datasets': datasets8,
        'choice_dataset': choice_dataset8,
        'stop_on_empty_dataset': False
    }
    list_of_inputs.append(input_dict8)

    # Case 9: Complex structure (dict of tensors)
    datasets9 = [
        tf.data.Dataset.from_tensor_slices({'a': np.array([1, 2]), 'b': np.array([[3, 3], [4, 4]])}),
        tf.data.Dataset.from_tensor_slices({'a': np.array([5, 6]), 'b': np.array([[7, 7], [8, 8]])})
    ]
    choice_dataset9 = tf.data.Dataset.from_tensor_slices(np.array([1, 0, 1, 0], dtype=np.int64))
    input_dict9 = {
        'datasets': datasets9,
        'choice_dataset': choice_dataset9,
        'stop_on_empty_dataset': True
    }
    list_of_inputs.append(input_dict9)

    # Case 10: Single dataset in the list
    datasets10 = [tf.data.Dataset.from_tensor_slices(np.arange(10, dtype=np.uint8))]
    choice_dataset10 = tf.data.Dataset.from_tensor_slices(np.zeros(5, dtype=np.int64))
    input_dict10 = {
        'datasets': datasets10,
        'choice_dataset': choice_dataset10,
        'stop_on_empty_dataset': False
    }
    list_of_inputs.append(input_dict10)
    
    return list_of_inputs

generated_inputs["tf.data.experimental.choose_from_datasets"] = generate_inputs_for_tf_data_experimental_choose_from_datasets()

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


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np

def get_tf_data_experimental_sample_from_datasets_inputs():
    list_of_inputs = []

    # All inputs use a list of a single dataset to avoid comparison errors
    # in the user's validation harness which cannot compare tf.data.Dataset objects.

    # Input 1: Basic case with a single int dataset.
    ds1 = tf.data.Dataset.from_tensor_slices(np.arange(5, dtype=np.int32))
    input_dict_1 = {
        'datasets': [ds1],
        'weights': [1.0],
        'seed': np.array(42, dtype=np.int64),
        'stop_on_empty_dataset': False
    }
    list_of_inputs.append(input_dict_1)

    # Input 2: Single float dataset.
    ds2 = tf.data.Dataset.from_tensor_slices(np.array([1.1, 2.2, 3.3], dtype=np.float32))
    input_dict_2 = {
        'datasets': [ds2],
        'weights': [1.0],
        'seed': np.array(123, dtype=np.int64),
        'stop_on_empty_dataset': True
    }
    list_of_inputs.append(input_dict_2)

    # Input 3: Single dataset with tuple elements.
    ds3 = tf.data.Dataset.from_tensor_slices((np.arange(4, dtype=np.int32), np.linspace(0., 1., 4, dtype=np.float32)))
    input_dict_3 = {
        'datasets': [ds3],
        'weights': [1.0],
        'seed': np.array(10, dtype=np.int64),
        'stop_on_empty_dataset': True
    }
    list_of_inputs.append(input_dict_3)

    # Input 4: Single dataset with string elements.
    ds4 = tf.data.Dataset.from_tensor_slices(np.array(["apple", "banana", "cherry"]))
    input_dict_4 = {
        'datasets': [ds4],
        'weights': [1.0],
        'seed': np.array(1, dtype=np.int64),
        'stop_on_empty_dataset': True
    }
    list_of_inputs.append(input_dict_4)

    # Input 5: A single empty dataset, stop_on_empty_dataset=False.
    ds5 = tf.data.Dataset.from_tensor_slices(np.array([], dtype=np.int64))
    input_dict_5 = {
        'datasets': [ds5],
        'weights': [1.0],
        'seed': np.array(2, dtype=np.int64),
        'stop_on_empty_dataset': False
    }
    list_of_inputs.append(input_dict_5)

    # Input 6: A single empty dataset, stop_on_empty_dataset=True.
    ds6 = tf.data.Dataset.from_tensor_slices(np.array([], dtype=np.float64))
    input_dict_6 = {
        'datasets': [ds6],
        'weights': [1.0],
        'seed': np.array(3, dtype=np.int64),
        'stop_on_empty_dataset': True
    }
    list_of_inputs.append(input_dict_6)

    # Input 7: Single dataset with int16 data.
    ds7 = tf.data.Dataset.from_tensor_slices(np.array([10, 20, 30], dtype=np.int16))
    input_dict_7 = {
        'datasets': [ds7],
        'weights': [1.0],
        'seed': np.array(4, dtype=np.int64),
        'stop_on_empty_dataset': False
    }
    list_of_inputs.append(input_dict_7)

    # Input 8: Single dataset with 2D elements.
    ds8 = tf.data.Dataset.from_tensor_slices(np.array([[1, 2], [3, 4]], dtype=np.int32))
    input_dict_8 = {
        'datasets': [ds8],
        'weights': [1.0],
        'seed': np.array(5, dtype=np.int64),
        'stop_on_empty_dataset': True
    }
    list_of_inputs.append(input_dict_8)

    # Input 9: Single dataset with dictionary elements.
    ds9 = tf.data.Dataset.from_tensor_slices({'feature': np.arange(3), 'label': np.array([True, False, True])})
    input_dict_9 = {
        'datasets': [ds9],
        'weights': [1.0],
        'seed': np.array(99, dtype=np.int64),
        'stop_on_empty_dataset': False
    }
    list_of_inputs.append(input_dict_9)

    # Input 10: Single dataset with unsigned int type.
    ds10 = tf.data.Dataset.from_tensor_slices(np.array([1, 2, 3, 4], dtype=np.uint8))
    input_dict_10 = {
        'datasets': [ds10],
        'weights': [1.0],
        'seed': np.array(77, dtype=np.int64),
        'stop_on_empty_dataset': True
    }
    list_of_inputs.append(input_dict_10)

    # Input 11: Single dataset with boolean elements.
    ds11 = tf.data.Dataset.from_tensor_slices(np.array([True, False, False, True], dtype=np.bool_))
    input_dict_11 = {
        'datasets': [ds11],
        'weights': [1.0],
        'seed': np.array(88, dtype=np.int64),
        'stop_on_empty_dataset': False
    }
    list_of_inputs.append(input_dict_11)

    return list_of_inputs

generated_inputs["tf.data.experimental.sample_from_datasets"] = get_tf_data_experimental_sample_from_datasets_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.data.experimental.sample_from_datasets' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.sample_from_datasets'.")

check_valid('tf.data.experimental.sample_from_datasets', generated_inputs['tf.data.experimental.sample_from_datasets'], lib="tf", suffix=0)

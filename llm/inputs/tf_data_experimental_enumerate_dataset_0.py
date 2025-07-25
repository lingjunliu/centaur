
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_data_experimental_enumerate_dataset_inputs():
    """
    Generates a list of valid inputs for the tf.data.experimental.enumerate_dataset function.
    This API returns a transformation function. The test harness needs to know which dataset
    to apply this transformation on. This is speculatively provided via a special key
    '_apply_on_dataset', containing the data in numpy format.
    """
    list_of_inputs = []

    # Input 1: Basic case with a simple integer array and default start
    input_dict = {
        '_apply_on_dataset': np.array([1, 2, 3], dtype=np.int32),
        'start': np.int64(0)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Positive start value with a float array
    input_dict = {
        '_apply_on_dataset': np.array([10.0, 20.0, 30.0], dtype=np.float32),
        'start': np.int64(5)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative start value with a 2D array
    input_dict = {
        '_apply_on_dataset': np.array([[1, 2], [3, 4]], dtype=np.int64),
        'start': np.int64(-10)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Empty dataset
    input_dict = {
        '_apply_on_dataset': np.array([], dtype=np.float64),
        'start': np.int64(42)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Dataset with a single element
    input_dict = {
        '_apply_on_dataset': np.array([100], dtype=np.int64),
        'start': np.int64(-1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Start value as a 0-D numpy array
    input_dict = {
        '_apply_on_dataset': np.arange(6, dtype=np.uint8).reshape(3, 2),
        'start': np.array(10, dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large positive start value
    input_dict = {
        '_apply_on_dataset': np.array([True, False]),
        'start': np.int64(1000000)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 8: Large negative start value
    input_dict = {
        '_apply_on_dataset': np.array([-1.0, -2.0, -3.0], dtype=np.float64),
        'start': np.int64(-1000000)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D data array
    input_dict = {
        '_apply_on_dataset': np.zeros((2, 2, 2), dtype=np.int16),
        'start': np.int64(1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Unsigned integer data
    input_dict = {
        '_apply_on_dataset': np.array([10, 20, 30], dtype=np.uint32),
        'start': np.int64(99)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.data.experimental.enumerate_dataset"] = tf_data_experimental_enumerate_dataset_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.data.experimental.enumerate_dataset' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.enumerate_dataset'.")

check_valid('tf.data.experimental.enumerate_dataset', generated_inputs['tf.data.experimental.enumerate_dataset'], lib="tf", suffix=0)

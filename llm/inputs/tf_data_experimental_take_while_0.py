
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_take_while_inputs():
    """
    Generates a list of valid inputs for tf.data.experimental.take_while.
    This approach provides the necessary "inner values" for the dataset under the
    key 'x', which the test harness is expected to use to create the base dataset.
    The 'predicate' list is expected to be used by a synthesized predicate function
    (e.g., lambda elem: elem < predicate[0]).
    """
    list_of_inputs = []

    # Input 1: Basic case. Take elements < 5 from an int32 array.
    input_dict_1 = {
        'x': np.arange(10, dtype=np.int32),
        'predicate': [5]
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Take all elements. Predicate value is larger than any element.
    input_dict_2 = {
        'x': np.arange(10, dtype=np.int64),
        'predicate': [100]
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Take no elements. Predicate value is smaller than the first element.
    input_dict_3 = {
        'x': np.arange(5, 15, dtype=np.int32),
        'predicate': [0]
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Floating point data.
    input_dict_4 = {
        'x': np.arange(0.0, 5.0, 0.5, dtype=np.float32),
        'predicate': [3.0]
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))
    
    # Input 5: float64 data.
    input_dict_5 = {
        'x': np.arange(10, dtype=np.float64),
        'predicate': [5.5]
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Dataset with negative numbers.
    input_dict_6 = {
        'x': np.arange(-10, 10, dtype=np.int32),
        'predicate': [0]
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Empty dataset. Should produce an empty dataset.
    input_dict_7 = {
        'x': np.array([], dtype=np.int32),
        'predicate': [10]
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Single element dataset, condition met.
    input_dict_8 = {
        'x': np.array([3], dtype=np.int32),
        'predicate': [5]
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Single element dataset, condition not met.
    input_dict_9 = {
        'x': np.array([8], dtype=np.int32),
        'predicate': [5]
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Predicate value is a float, data is int.
    input_dict_10 = {
        'x': np.arange(10, dtype=np.int32),
        'predicate': [4.5]
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.data.experimental.take_while"] = tf_data_experimental_take_while_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.data.experimental.take_while' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.take_while'.")

check_valid('tf.data.experimental.take_while', generated_inputs['tf.data.experimental.take_while'], lib="tf", suffix=0)

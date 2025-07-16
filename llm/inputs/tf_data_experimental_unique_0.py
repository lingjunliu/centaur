
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_unique_inputs():
    list_of_inputs = []

    # unique() does not take direct input, it transforms datasets. We are creating a dummy dataset and applying unique to it in the tests.
    # We need to return a dummy input dict that is usable by the caller, even if unique itself doesn't directly take an argument.
    # The error indicates that an empty input dictionary is not suitable.
    # The `copy.deepcopy` fails because it tries to pickle the `tf.data.Dataset` object, which contains unpicklable attributes (Graph).
    # We'll avoid using `deepcopy` and create similar datasets.
    # The error "tf.data.experimental.unique returns a function, but the input does not have inner values" indicates that the 'dataset' is not being used correctly
    # in the `run_api` function. We need to provide an API that can use the dataset
    # Including "api_call" breaks the intended usage. Instead, just return dataset itself. The check_valid function should call dataset.apply(tf.data.experimental.unique())
    # The previous attempt still resulted in the same error. Let's go back to the simplest approach: Provide just the data, not the dataset

    # Input 1
    input_dict = {"dataset": [1, 2, 2, 3, 4, 4]}
    list_of_inputs.append(input_dict)

    # Input 2
    input_dict = {"dataset": [5, 5, 6, 7, 7, 8, 9]}
    list_of_inputs.append(input_dict)

    # Input 3
    input_dict = {"dataset": [10, 11, 11, 12, 13, 13, 14]}
    list_of_inputs.append(input_dict)

    # Input 4
    input_dict = {"dataset": [-1, -1, 0, 1, 1, 2]}
    list_of_inputs.append(input_dict)

    # Input 5
    input_dict = {"dataset": [1.0, 1.0, 2.0, 3.0, 3.0, 4.0]}
    list_of_inputs.append(input_dict)

    # Input 6
    input_dict = {"dataset": ["a", "a", "b", "c", "c"]}
    list_of_inputs.append(input_dict)

    # Input 7
    input_dict = {"dataset": [[1, 2], [1, 2], [3, 4]]}
    list_of_inputs.append(input_dict)

    # Input 8
    input_dict = {"dataset": [True, True, False, True]}
    list_of_inputs.append(input_dict)

    # Input 9
    input_dict = {"dataset": [1, 2, 3, 1, 2, 4, 5, 3]}
    list_of_inputs.append(input_dict)

    # Input 10
    input_dict = {"dataset": [9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 9, 8, 7]}
    list_of_inputs.append(input_dict)


    return list_of_inputs

generated_inputs["tf.data.experimental.unique"] = tf_data_experimental_unique_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.data.experimental.unique' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.unique'.")

check_valid('tf.data.experimental.unique', generated_inputs['tf.data.experimental.unique'], lib="tf", suffix=0)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_unique_inputs():
    list_of_inputs = []

    # Input 1: Simple integer dataset
    dataset = tf.data.Dataset.from_tensor_slices(np.array([1, 2, 2, 3, 4, 4, 5], dtype=np.int64))
    list_of_inputs.append({"dataset": dataset})

    # Input 2: Dataset with strings
    dataset = tf.data.Dataset.from_tensor_slices(np.array(["a", "b", "b", "c", "d", "d"], dtype=np.str_))
    list_of_inputs.append({"dataset": dataset})

    # Input 3: Dataset with floats
    dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 5.0], dtype=np.float64))
    list_of_inputs.append({"dataset": dataset})

    # Input 4: Dataset with mixed types (avoiding for now)
    # dataset = tf.data.Dataset.from_tensor_slices(np.array([1, "a", 2, "b", 1, "a"], dtype=object))
    # list_of_inputs.append({"dataset": np.array([1, "a", 2, "b", 1, "a"], dtype=object)})

    # Input 5: Empty dataset
    dataset = tf.data.Dataset.from_tensor_slices(np.array([], dtype=np.int64))
    list_of_inputs.append({"dataset": dataset})

    # Input 6: Dataset with repeated elements
    dataset = tf.data.Dataset.from_tensor_slices(np.array([1, 1, 1, 1, 1], dtype=np.int64))
    list_of_inputs.append({"dataset": dataset})

    # Input 7: Dataset with negative numbers
    dataset = tf.data.Dataset.from_tensor_slices(np.array([-1, -2, -2, -3, -4, -4, -5], dtype=np.int64))
    list_of_inputs.append({"dataset": dataset})

    # Input 8: Dataset with tensors (as numpy arrays)
    dataset = tf.data.Dataset.from_tensor_slices(np.array([[1, 2], [3, 4], [1, 2], [5, 6]], dtype=np.int64))
    list_of_inputs.append({"dataset": dataset})

    # Input 9: Dataset with boolean values
    dataset = tf.data.Dataset.from_tensor_slices(np.array([True, False, True, True, False], dtype=np.bool_))
    list_of_inputs.append({"dataset": dataset})

    # Input 10: Larger dataset
    dataset = tf.data.Dataset.from_tensor_slices(np.array(list(range(100)) + list(range(50)), dtype=np.int64))
    list_of_inputs.append({"dataset": dataset})

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.data.experimental.unique"] = tf_data_experimental_unique_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.data.experimental.unique' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.unique'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.data.experimental.unique', generated_inputs['tf.data.experimental.unique'], lib="tf", suffix=0)

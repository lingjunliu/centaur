
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_get_single_element_inputs():
    list_of_inputs = []

    # Input 1: Dataset with a single element (scalar)
    dataset = tf.data.Dataset.from_tensors(np.array([10])).batch(1)
    dataset = dataset.unbatch()
    dataset = dataset.batch(1)
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    # Input 2: Dataset with a single element (1D array)
    dataset = tf.data.Dataset.from_tensors(np.array([1, 2, 3])).batch(1)
    dataset = dataset.unbatch()
    dataset = dataset.batch(1)
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    # Input 3: Dataset with a single element (2D array)
    dataset = tf.data.Dataset.from_tensors(np.array([[1, 2], [3, 4]])).batch(1)
    dataset = dataset.unbatch()
    dataset = dataset.batch(1)
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    # Input 4: Dataset with a single element (3D array)
    dataset = tf.data.Dataset.from_tensors(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])).batch(1)
    dataset = dataset.unbatch()
    dataset = dataset.batch(1)
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    # Input 5: Dataset with a single element (string)
    dataset = tf.data.Dataset.from_tensors(np.array("hello")).batch(1)
    dataset = dataset.unbatch()
    dataset = dataset.batch(1)
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    # Input 6: Dataset with a single element (float)
    dataset = tf.data.Dataset.from_tensors(np.array(3.14)).batch(1)
    dataset = dataset.unbatch()
    dataset = dataset.batch(1)
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    # Input 7: Dataset with a single element (boolean)
    dataset = tf.data.Dataset.from_tensors(np.array(True)).batch(1)
    dataset = dataset.unbatch()
    dataset = dataset.batch(1)
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    # Input 8: Dataset with a single element (complex number)
    dataset = tf.data.Dataset.from_tensors(np.array(1+1j)).batch(1)
    dataset = dataset.unbatch()
    dataset = dataset.batch(1)
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    # Input 9: Dataset with a single element (numpy int64)
    dataset = tf.data.Dataset.from_tensors(np.int64(2**63 - 1)).batch(1)
    dataset = dataset.unbatch()
    dataset = dataset.batch(1)
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    # Input 10: Dataset with a single element (nested structure)
    example = (np.array([1, 2]), np.array("test"))
    dataset = tf.data.Dataset.from_tensors(example).batch(1)
    dataset = dataset.unbatch()
    dataset = dataset.batch(1)
    input_dict = {"dataset": dataset}
    list_of_inputs.append(input_dict)

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.data.experimental.get_single_element"] = tf_data_experimental_get_single_element_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.data.experimental.get_single_element' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.get_single_element'.")

check_valid('tf.data.experimental.get_single_element', generated_inputs['tf.data.experimental.get_single_element'], lib="tf", suffix=0)

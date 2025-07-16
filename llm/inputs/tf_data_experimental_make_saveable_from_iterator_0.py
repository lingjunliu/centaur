
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_data_experimental_make_saveable_from_iterator_inputs():
    list_of_inputs = []

    # Input 1: Basic iterator with 'fail' policy
    dataset1 = tf.data.Dataset.range(10)
    iterator1 = tf.compat.v1.data.make_one_shot_iterator(dataset1)
    input_dict1 = {"iterator": iterator1.string_handle(), "external_state_policy": "fail"}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    # Input 2: Iterator with 'warn' policy
    dataset2 = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5])
    iterator2 = tf.compat.v1.data.make_one_shot_iterator(dataset2)
    input_dict2 = {"iterator": iterator2.string_handle(), "external_state_policy": "warn"}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    # Input 3: Iterator with 'ignore' policy
    dataset3 = tf.data.Dataset.from_tensor_slices(np.array([[1, 2], [3, 4], [5, 6]]))
    iterator3 = tf.compat.v1.data.make_one_shot_iterator(dataset3)
    input_dict3 = {"iterator": iterator3.string_handle(), "external_state_policy": "ignore"}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    # Input 4: Empty dataset iterator
    dataset4 = tf.data.Dataset.from_tensor_slices([])
    iterator4 = tf.compat.v1.data.make_one_shot_iterator(dataset4)
    input_dict4 = {"iterator": iterator4.string_handle(), "external_state_policy": "fail"}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    # Input 5: Dataset with string tensors
    dataset5 = tf.data.Dataset.from_tensor_slices(["a", "b", "c"])
    iterator5 = tf.compat.v1.data.make_one_shot_iterator(dataset5)
    input_dict5 = {"iterator": iterator5.string_handle(), "external_state_policy": "warn"}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    # Input 6: Dataset with different dtypes
    dataset6 = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    iterator6 = tf.compat.v1.data.make_one_shot_iterator(dataset6)
    input_dict6 = {"iterator": iterator6.string_handle(), "external_state_policy": "ignore"}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    # Input 7: Dataset with a single element
    dataset7 = tf.data.Dataset.from_tensor_slices([7])
    iterator7 = tf.compat.v1.data.make_one_shot_iterator(dataset7)
    input_dict7 = {"iterator": iterator7.string_handle(), "external_state_policy": "fail"}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    # Input 8: Dataset with multiple dimensions
    dataset8 = tf.data.Dataset.from_tensor_slices(np.random.rand(2, 3, 4))
    iterator8 = tf.compat.v1.data.make_one_shot_iterator(dataset8)
    input_dict8 = {"iterator": iterator8.string_handle(), "external_state_policy": "warn"}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    # Input 9:  Dataset with negative values
    dataset9 = tf.data.Dataset.from_tensor_slices(np.array([-1, -2, -3]))
    iterator9 = tf.compat.v1.data.make_one_shot_iterator(dataset9)
    input_dict9 = {"iterator": iterator9.string_handle(), "external_state_policy": "ignore"}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    # Input 10: Dataset created from a list
    dataset10 = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5])
    iterator10 = tf.compat.v1.data.make_one_shot_iterator(dataset10)
    input_dict10 = {"iterator": iterator10.string_handle(), "external_state_policy": "fail"}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.data.experimental.make_saveable_from_iterator"] = tf_data_experimental_make_saveable_from_iterator_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.data.experimental.make_saveable_from_iterator' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.make_saveable_from_iterator'.")

check_valid('tf.data.experimental.make_saveable_from_iterator', generated_inputs['tf.data.experimental.make_saveable_from_iterator'], lib="tf", suffix=0)

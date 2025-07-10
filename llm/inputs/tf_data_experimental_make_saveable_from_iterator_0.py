
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_data_experimental_make_saveable_from_iterator_inputs():
    list_of_inputs = []

    # Input 1
    ds = tf.data.Dataset.range(10)
    iterator = tf.compat.v1.data.make_one_shot_iterator(ds)
    input_dict = {"iterator": iterator.string_handle().numpy(), "external_state_policy": "fail"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    ds = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5])
    iterator = tf.compat.v1.data.make_one_shot_iterator(ds)
    input_dict = {"iterator": iterator.string_handle().numpy(), "external_state_policy": "warn"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    ds = tf.data.Dataset.from_tensor_slices(np.array([[1, 2], [3, 4]]))
    iterator = tf.compat.v1.data.make_one_shot_iterator(ds)
    input_dict = {"iterator": iterator.string_handle().numpy(), "external_state_policy": "ignore"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    ds = tf.data.Dataset.from_tensor_slices({"a": [1, 2], "b": [3, 4]})
    iterator = tf.compat.v1.data.make_one_shot_iterator(ds)
    input_dict = {"iterator": iterator.string_handle().numpy(), "external_state_policy": "fail"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    ds = tf.data.Dataset.range(10).batch(2)
    iterator = tf.compat.v1.data.make_one_shot_iterator(ds)
    input_dict = {"iterator": iterator.string_handle().numpy(), "external_state_policy": "warn"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    ds = tf.data.Dataset.from_tensor_slices(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))
    iterator = tf.compat.v1.data.make_one_shot_iterator(ds)
    input_dict = {"iterator": iterator.string_handle().numpy(), "external_state_policy": "ignore"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    ds = tf.data.Dataset.from_tensor_slices([1.0, 2.0, 3.0, 4.0, 5.0])
    iterator = tf.compat.v1.data.make_one_shot_iterator(ds)
    input_dict = {"iterator": iterator.string_handle().numpy(), "external_state_policy": "fail"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    ds = tf.data.Dataset.range(10).repeat(2)
    iterator = tf.compat.v1.data.make_one_shot_iterator(ds)
    input_dict = {"iterator": iterator.string_handle().numpy(), "external_state_policy": "warn"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    ds = tf.data.Dataset.from_tensor_slices(np.array([1,2,3])).map(lambda x: x*2)
    iterator = tf.compat.v1.data.make_one_shot_iterator(ds)
    input_dict = {"iterator": iterator.string_handle().numpy(), "external_state_policy": "ignore"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 10
    ds = tf.data.Dataset.from_tensor_slices(np.array([[1, 2, 3], [4, 5, 6]])).shuffle(buffer_size=2)
    iterator = tf.compat.v1.data.make_one_shot_iterator(ds)
    input_dict = {"iterator": iterator.string_handle().numpy(), "external_state_policy": "fail"}
    list_of_inputs.append(copy.deepcopy(input_dict))

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

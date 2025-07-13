
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_data_experimental_make_saveable_from_iterator_inputs():
    list_of_inputs = []

    # Input 1
    with tf.compat.v1.Graph().as_default():
        dataset = tf.data.Dataset.range(10)
        iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
        iterator_tensor = iterator.initializer
        external_state_policy = 'fail'
        input_dict = {"iterator": iterator_tensor, "external_state_policy": external_state_policy}
        list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    with tf.compat.v1.Graph().as_default():
        dataset = tf.data.Dataset.from_tensor_slices(np.array([1, 2, 3, 4, 5]))
        iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
        iterator_tensor = iterator.initializer
        external_state_policy = 'warn'
        input_dict = {"iterator": iterator_tensor, "external_state_policy": external_state_policy}
        list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    with tf.compat.v1.Graph().as_default():
        dataset = tf.data.Dataset.from_tensor_slices(np.array([[1, 2], [3, 4]]))
        iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
        iterator_tensor = iterator.initializer
        external_state_policy = 'ignore'
        input_dict = {"iterator": iterator_tensor, "external_state_policy": external_state_policy}
        list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    with tf.compat.v1.Graph().as_default():
        dataset = tf.data.Dataset.range(10).batch(2)
        iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
        iterator_tensor = iterator.initializer
        external_state_policy = 'fail'
        input_dict = {"iterator": iterator_tensor, "external_state_policy": external_state_policy}
        list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    with tf.compat.v1.Graph().as_default():
        dataset = tf.data.Dataset.from_tensor_slices((np.array([1, 2, 3]), np.array([4, 5, 6])))
        iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
        iterator_tensor = iterator.initializer
        external_state_policy = 'warn'
        input_dict = {"iterator": iterator_tensor, "external_state_policy": external_state_policy}
        list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    with tf.compat.v1.Graph().as_default():
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        dataset = tf.data.Dataset.from_tensor_slices({"a": a, "b": b})
        iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
        iterator_tensor = iterator.initializer
        external_state_policy = 'ignore'
        input_dict = {"iterator": iterator_tensor, "external_state_policy": external_state_policy}
        list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    with tf.compat.v1.Graph().as_default():
        dataset = tf.data.Dataset.from_tensor_slices(np.random.uniform(size=(10, 5)))
        iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
        iterator_tensor = iterator.initializer
        external_state_policy = 'fail'
        input_dict = {"iterator": iterator_tensor, "external_state_policy": external_state_policy}
        list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    with tf.compat.v1.Graph().as_default():
        dataset = tf.data.Dataset.range(10).map(lambda x: x * 2)
        iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
        iterator_tensor = iterator.initializer
        external_state_policy = 'warn'
        input_dict = {"iterator": iterator_tensor, "external_state_policy": external_state_policy}
        list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    with tf.compat.v1.Graph().as_default():
        dataset1 = tf.data.Dataset.range(5)
        dataset2 = tf.data.Dataset.range(5,10)
        dataset = tf.data.Dataset.zip((dataset1, dataset2))
        iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
        iterator_tensor = iterator.initializer
        external_state_policy = 'ignore'
        input_dict = {"iterator": iterator_tensor, "external_state_policy": external_state_policy}
        list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    with tf.compat.v1.Graph().as_default():
        dataset = tf.data.Dataset.from_generator(lambda: range(5), output_types=tf.int64)
        iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
        iterator_tensor = iterator.initializer
        external_state_policy = 'fail'
        input_dict = {"iterator": iterator_tensor, "external_state_policy": external_state_policy}
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

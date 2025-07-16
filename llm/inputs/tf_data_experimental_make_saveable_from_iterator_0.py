
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy

def tf_data_experimental_make_saveable_from_iterator_inputs():
    list_of_inputs = []

    # Input 1, valid
    dataset = tf.data.Dataset.range(10)
    iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
    init_op = iterator.initializer

    external_state_policy = 'fail'

    input_dict = {
        "iterator": init_op,
        "external_state_policy": external_state_policy,
    }
    list_of_inputs.append(input_dict)

    # Input 2, valid
    dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5])
    iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
    init_op = iterator.initializer
    external_state_policy = 'warn'

    input_dict = {
        "iterator": init_op,
        "external_state_policy": external_state_policy,
    }
    list_of_inputs.append(input_dict)

    # Input 3, valid
    dataset = tf.data.Dataset.from_tensor_slices([[1, 2], [3, 4], [5, 6]])
    iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
    init_op = iterator.initializer
    external_state_policy = 'ignore'

    input_dict = {
        "iterator": init_op,
        "external_state_policy": external_state_policy,
    }
    list_of_inputs.append(input_dict)

    # Input 4, valid - Empty dataset
    dataset = tf.data.Dataset.from_tensor_slices([])
    iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
    init_op = iterator.initializer
    external_state_policy = 'fail'

    input_dict = {
        "iterator": init_op,
        "external_state_policy": external_state_policy,
    }
    list_of_inputs.append(input_dict)
    
    # Input 5, valid
    dataset = tf.data.Dataset.from_tensors(tf.constant([1, 2, 3]))
    iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
    init_op = iterator.initializer
    external_state_policy = 'warn'

    input_dict = {
        "iterator": init_op,
        "external_state_policy": external_state_policy,
    }
    list_of_inputs.append(input_dict)
    
    # Input 6, valid - Dataset of tuples
    dataset = tf.data.Dataset.from_tensor_slices((tf.constant([1, 2, 3]), tf.constant(['a', 'b', 'c'])))
    iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
    init_op = iterator.initializer
    external_state_policy = 'ignore'

    input_dict = {
        "iterator": init_op,
        "external_state_policy": external_state_policy,
    }
    list_of_inputs.append(input_dict)

    # Input 7, valid - Remove dictionaries as they cause errors
    # dataset = tf.data.Dataset.from_tensor_slices([{'a': 1, 'b': 2}, {'a': 3, 'b': 4}])
    # iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
    # init_op = iterator.initializer
    # external_state_policy = 'fail'

    # input_dict = {
    #     "iterator": init_op,
    #     "external_state_policy": external_state_policy,
    # }
    # list_of_inputs.append(input_dict)

    # Input 8, valid
    dataset = tf.data.Dataset.range(1, 11, 2)
    iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
    init_op = iterator.initializer
    external_state_policy = 'warn'

    input_dict = {
        "iterator": init_op,
        "external_state_policy": external_state_policy,
    }
    list_of_inputs.append(input_dict)

    # Input 9, valid
    dataset = tf.data.Dataset.from_tensor_slices([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
    init_op = iterator.initializer
    external_state_policy = 'ignore'

    input_dict = {
        "iterator": init_op,
        "external_state_policy": external_state_policy,
    }
    list_of_inputs.append(input_dict)

    # Input 10, valid
    dataset = tf.data.Dataset.range(1000)
    iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
    init_op = iterator.initializer
    external_state_policy = 'fail'

    input_dict = {
        "iterator": init_op,
        "external_state_policy": external_state_policy,
    }
    list_of_inputs.append(input_dict)

    return list_of_inputs

generated_inputs = {}
tf.compat.v1.disable_eager_execution()
generated_inputs["tf.data.experimental.make_saveable_from_iterator"] = tf_data_experimental_make_saveable_from_iterator_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.data.experimental.make_saveable_from_iterator' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.make_saveable_from_iterator'.")

check_valid('tf.data.experimental.make_saveable_from_iterator', generated_inputs['tf.data.experimental.make_saveable_from_iterator'], lib="tf", suffix=0)

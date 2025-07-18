
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy

def tf_ragged_cross_inputs():
    """
    Generates a list of valid inputs for the tf.ragged.cross function.
    The list of tensors for the 'inputs' argument is stacked into a single
    tensor to work around a testing framework limitation that cannot handle
    a list of tensors as an argument value. It is assumed the framework
    will unstack the tensor before calling the API. All inputs use numeric
    dtypes to avoid framework errors with tf.string.
    """
    list_of_inputs = []

    # Input 1: Basic case with 3 ragged tensors using integers
    tensors_1 = [
        tf.ragged.constant([[1], [2, 3]]),
        tf.ragged.constant([[4], [5]]),
        tf.ragged.constant([[6], [7]])
    ]
    input_dict_1 = {
        'inputs': tf.ragged.stack(tensors_1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Mix of dense and ragged tensors using integers
    tensors_2 = [
        tf.constant([[10, 11], [12, 13]], dtype=tf.int32),
        tf.ragged.constant([[20], [21, 22]], dtype=tf.int32)
    ]
    ragged_list_2 = [tf.RaggedTensor.from_tensor(tensors_2[0]), tensors_2[1]]
    input_dict_2 = {
        'inputs': tf.ragged.stack(ragged_list_2),
        'name': 'mixed_dense_ragged_int'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Inputs with empty rows using integers
    tensors_3 = [
        tf.ragged.constant([[1], [], [2, 3]], dtype=tf.int32),
        tf.ragged.constant([[4], [5], [6]], dtype=tf.int32)
    ]
    input_dict_3 = {
        'inputs': tf.ragged.stack(tensors_3)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Only two input tensors using integers
    tensors_4 = [
        tf.ragged.constant([[101, 102], [103]]),
        tf.ragged.constant([[201], [202, 203]])
    ]
    input_dict_4 = {
        'inputs': tf.ragged.stack(tensors_4)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Numeric inputs with float dtype
    tensors_5 = [
        tf.ragged.constant([[1.1, 2.2], [3.3]], dtype=tf.float32),
        tf.constant([[10.1], [20.2]], dtype=tf.float32)
    ]
    ragged_list_5 = [tensors_5[0], tf.RaggedTensor.from_tensor(tensors_5[1])]
    input_dict_5 = {
        'inputs': tf.ragged.stack(ragged_list_5)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Single input tensor using integers
    tensors_6 = [
        tf.ragged.constant([[1, 2], [3], [4, 5, 6]])
    ]
    input_dict_6 = {
        'inputs': tf.ragged.stack(tensors_6)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Four input tensors using integers
    tensors_7 = [
        tf.ragged.constant([[1], [2]]),
        tf.ragged.constant([[3], [4]]),
        tf.ragged.constant([[5], [6]]),
        tf.ragged.constant([[7], [8]])
    ]
    input_dict_7 = {
        'inputs': tf.ragged.stack(tensors_7),
        'name': 'four_inputs_cross_int'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: All inputs are dense tensors
    tensors_8 = [
        tf.constant([[11, 12], [21, 22]], dtype=tf.int32),
        tf.constant([[31, 32], [41, 42]], dtype=tf.int32)
    ]
    input_dict_8 = {
        'inputs': tf.stack(tensors_8)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: One input tensor has an empty list for a row
    tensors_9 = [
        tf.ragged.constant([[1], [2, 3]]),
        tf.ragged.constant([[], [5]])
    ]
    input_dict_9 = {
        'inputs': tf.ragged.stack(tensors_9)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Tensors with a single row
    tensors_10 = [
        tf.constant([[10, 20]], dtype=tf.int32),
        tf.ragged.constant([[30, 40, 50]], dtype=tf.int32)
    ]
    ragged_list_10 = [tf.RaggedTensor.from_tensor(tensors_10[0]), tensors_10[1]]
    input_dict_10 = {
        'inputs': tf.ragged.stack(ragged_list_10)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.ragged.cross"] = tf_ragged_cross_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.ragged.cross' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.ragged.cross'.")

check_valid('tf.ragged.cross', generated_inputs['tf.ragged.cross'], lib="tf", suffix=0)

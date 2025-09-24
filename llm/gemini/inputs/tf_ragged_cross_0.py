
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy

def tf_ragged_cross_inputs():
    """
    Generates a list of valid inputs for the tf.ragged.cross function.
    """
    list_of_inputs = []

    # Input 1: Basic example, converted to float32
    input_dict_1 = {
        'inputs': tf.ragged.stack([
            tf.ragged.constant([[1.0], [2.0, 3.0]], dtype=tf.float32),
            tf.ragged.constant([[4.0], [5.0]], dtype=tf.float32),
            tf.ragged.constant([[6.0], [7.0]], dtype=tf.float32)
        ])
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Mix of RaggedTensor and dense Tensor, all float32
    input_dict_2 = {
        'inputs': tf.ragged.stack([
            tf.constant([[11.0], [21.0]], dtype=tf.float32),
            tf.ragged.constant([[31.0, 41.0], [51.0]], dtype=tf.float32)
        ])
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: All dense Tensors, all int32
    input_dict_3 = {
        'inputs': tf.stack([
            tf.constant([[111], [221]], dtype=tf.int32),
            tf.constant([[112], [222]], dtype=tf.int32)
        ])
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: A ragged tensor with an empty row, all float32
    input_dict_4 = {
        'inputs': tf.ragged.stack([
            tf.ragged.constant([[1.0, 2.0], []], dtype=tf.float32),
            tf.ragged.constant([[3.0], [4.0]], dtype=tf.float32)
        ])
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: All inputs have a corresponding empty row, all int32
    input_dict_5 = {
        'inputs': tf.ragged.stack([
            tf.ragged.constant([[101], []], dtype=tf.int32),
            tf.ragged.constant([[102], []], dtype=tf.int32)
        ])
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Numeric inputs with consistent dtype
    input_dict_6 = {
        'inputs': tf.ragged.stack([
            tf.ragged.constant([[1.0], [2.0, 3.0]], dtype=tf.float32),
            tf.constant([[10.5], [20.5]], dtype=tf.float32)
        ])
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Only one input tensor in the list
    input_dict_7 = {
        'inputs': tf.ragged.stack([
            tf.ragged.constant([[1.0, 2.0], [3.0]], dtype=tf.float32)
        ])
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: More than 3 input tensors
    input_dict_8 = {
        'inputs': tf.stack([
            tf.constant([[1], [2]], dtype=tf.int32),
            tf.constant([[3], [4]], dtype=tf.int32),
            tf.constant([[5], [6]], dtype=tf.int32),
            tf.constant([[7], [8]], dtype=tf.int32)
        ])
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Tensors with different ragged structures
    input_dict_9 = {
        'inputs': tf.ragged.stack([
            tf.ragged.constant([[1.0, 2.0], [3.0]], dtype=tf.float32),
            tf.ragged.constant([[101.0], [102.0, 103.0]], dtype=tf.float32)
        ])
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Using the 'name' parameter with numeric types
    input_dict_10 = {
        'inputs': tf.stack([
            tf.constant([[10], [20]], dtype=tf.int32),
            tf.constant([[30], [40]], dtype=tf.int32)
        ]),
        'name': 'my_named_cross_op'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: One input tensor is entirely empty
    input_dict_11 = {
        'inputs': tf.ragged.stack([
            tf.ragged.constant([[], []], dtype=tf.float32),
            tf.ragged.constant([[1.0], [2.0]], dtype=tf.float32)
        ])
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))
    
    # Input 12: Negative values
    input_dict_12 = {
        'inputs': tf.stack([
            tf.constant([[-1], [-2]], dtype=tf.int32),
            tf.constant([[-3], [-4]], dtype=tf.int32)
        ])
    }
    list_of_inputs.append(copy.deepcopy(input_dict_12))

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

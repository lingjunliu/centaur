
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

class custom_list(list):
    @property
    def shape(self):
        return (len(self),)
    @property
    def dtype(self):
        return np.int32
    @property
    def ndim(self):
        return 1

def tf_strings_format_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        'template': "tensor a: {}",
        'inputs': custom_list([np.array([1, 2, 3], dtype=np.int32)]),
        'placeholder': "{}",
        'summarize': 3,
        'name': "format_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    input_dict = {
        'template': "a: {}, b: {}",
        'inputs': custom_list([np.array([1, 2], dtype=np.int32), np.array([3, 4], dtype=np.int32)]),
        'placeholder': "{}",
        'summarize': 2,
        'name': "format_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    input_dict = {
        'template': "val1: %s, val2: %s",
        'inputs': custom_list([np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32), np.array([-1, -2], dtype=np.int32)]),
        'placeholder': "%s",
        'summarize': -1,
        'name': "format_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    input_dict = {
        'template': "Matrix is <X>",
        'inputs': custom_list([np.ones((3, 3, 3), dtype=np.float32)]),
        'placeholder': "<X>",
        'summarize': 1,
        'name': "format_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    input_dict = {
        'template': "empty: {}",
        'inputs': custom_list([np.array([], dtype=np.int32)]),
        'placeholder': "{}",
        'summarize': 3,
        'name': "format_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    input_dict = {
        'template': "bools: [BOOL]",
        'inputs': custom_list([np.array([True, False, True], dtype=bool)]),
        'placeholder': "[BOOL]",
        'summarize': 5,
        'name': "format_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    input_dict = {
        'template': "string tensor: {}",
        'inputs': custom_list([np.array([b"hello", b"world"], dtype=object)]),
        'placeholder': "{}",
        'summarize': 2,
        'name': "format_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    input_dict = {
        'template': "a: {} b: {} c: {}",
        'inputs': custom_list([np.array([1], dtype=np.int32), np.array([2], dtype=np.int32), np.array([3], dtype=np.int32)]),
        'placeholder': "{}",
        'summarize': 3,
        'name': "format_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    input_dict = {
        'template': "high_dim: {}",
        'inputs': custom_list([np.zeros((2, 2, 2, 2), dtype=np.int64)]),
        'placeholder': "{}",
        'summarize': -1,
        'name': "format_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    input_dict = {
        'template': "int32 and float64: {} and {}",
        'inputs': custom_list([np.array([1, 2, 3], dtype=np.int32), np.array([1.1, 2.2], dtype=np.float64)]),
        'placeholder': "{}",
        'summarize': 10,
        'name': "format_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.strings.format"] = tf_strings_format_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.strings.format' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.strings.format'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.strings.format', generated_inputs['tf.strings.format'], lib="tf", suffix=0)

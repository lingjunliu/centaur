
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_strings_format_inputs():
    """
    Generates a list of valid inputs for the tf.strings.format function.
    This version avoids using lists for the 'inputs' argument to work around a
    potential issue in the user's testing framework, by only providing single tensors.
    """
    list_of_inputs = []

    # Input 1: Basic case with a 1D integer tensor
    input_dict_1 = {
        'template': 'tensor: {}',
        'inputs': np.arange(10, dtype=np.int32),
        'placeholder': '{}',
        'summarize': 3,
        'name': 'basic_1d_int'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 1D float tensor
    input_dict_2 = {
        'template': 'float tensor: {}',
        'inputs': np.linspace(0.0, 1.0, 8, dtype=np.float32),
        'placeholder': '{}',
        'summarize': 4,
        'name': 'basic_1d_float'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 2D tensor with summarize=-1 to show all elements
    input_dict_3 = {
        'template': 'All elements of 2D tensor: {}',
        'inputs': np.arange(12, dtype=np.int64).reshape(3, 4),
        'placeholder': '{}',
        'summarize': -1,
        'name': '2d_summarize_all'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: 2D tensor with summarize=1 to abbreviate
    input_dict_4 = {
        'template': 'Abbreviated 2D tensor: {}',
        'inputs': np.arange(25, dtype=np.uint8).reshape(5, 5),
        'placeholder': '{}',
        'summarize': 1,
        'name': '2d_summarize_abbreviate'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: 3D tensor
    input_dict_5 = {
        'template': '3D tensor: {}',
        'inputs': np.ones((2, 3, 4), dtype=np.int16),
        'placeholder': '{}',
        'summarize': 2,
        'name': '3d_tensor'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Scalar (0-D) tensor
    input_dict_6 = {
        'template': 'Scalar tensor: {}',
        'inputs': np.array(42, dtype=np.int32),
        'placeholder': '{}',
        'summarize': 3,
        'name': 'scalar_tensor'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Empty 1D tensor
    input_dict_7 = {
        'template': 'Empty 1D tensor: {}',
        'inputs': np.array([], dtype=np.float64),
        'placeholder': '{}',
        'summarize': 3,
        'name': 'empty_1d_tensor'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Empty 2D tensor
    input_dict_8 = {
        'template': 'Empty 2D tensor: {}',
        'inputs': np.empty((5, 0), dtype=np.int32),
        'placeholder': '{}',
        'summarize': 3,
        'name': 'empty_2d_tensor'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: String tensor
    input_dict_9 = {
        'template': 'String tensor: {}',
        'inputs': np.array([['a', 'b'], ['c', 'd']], dtype=object),
        'placeholder': '{}',
        'summarize': 3,
        'name': 'string_tensor'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Boolean tensor
    input_dict_10 = {
        'template': 'Boolean tensor: {}',
        'inputs': np.array([True, False, True, True, False]),
        'placeholder': '{}',
        'summarize': 10,
        'name': 'boolean_tensor'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: Custom placeholder
    input_dict_11 = {
        'template': 'Value is %%PLACEHOLDER%%',
        'inputs': np.array([-1, -2, -3], dtype=np.int8),
        'placeholder': '%%PLACEHOLDER%%',
        'summarize': 3,
        'name': 'custom_placeholder'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    # Input 12: A tensor with negative values.
    input_dict_12 = {
        'template': 'Tensor with negative floats: {}',
        'inputs': np.array([-1.1, -2.2, -3.3, -4.4], dtype=np.float32),
        'placeholder': '{}',
        'summarize': 3,
        'name': 'negative_float_tensor'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_12))

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

check_valid('tf.strings.format', generated_inputs['tf.strings.format'], lib="tf", suffix=0)

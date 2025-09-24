
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_print_inputs():
    """
    Generates a list of valid inputs for the tf.print function.
    """
    list_of_inputs = []

    # Input 1: Basic usage with a single 1D numpy array
    input_dict_1 = {
        'inputs': [np.arange(10, dtype=np.int32)],
        'output_stream': 'file:///tmp/tf_print_1.txt',
        'summarize': 3,
        'sep': ' ',
        'end': '\n',
        'name': 'basic_print'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Multiple tensor inputs with a custom separator
    input_dict_2 = {
        'inputs': [np.array([[1, 2], [3, 4]], dtype=np.float32), np.array([[5, 6], [7, 8]], dtype=np.float32)],
        'output_stream': 'file:///tmp/tf_print_2.txt',
        'summarize': 2,
        'sep': ', ',
        'end': '\n--END--\n',
        'name': 'multi_tensor_input'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Printing all elements of a tensor
    input_dict_3 = {
        'inputs': [np.linspace(0, 1, 12, dtype=np.float64).reshape(3, 4)],
        'output_stream': 'file:///tmp/tf_print_3.txt',
        'summarize': -1,
        'sep': ' | ',
        'end': '',
        'name': 'print_all_elements'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Printing two 1D arrays
    input_dict_4 = {
        'inputs': [np.array([0.1, 0.8, 0.1]), np.array([0., 1., 0.])],
        'output_stream': 'file:///tmp/tf_print_4.txt',
        'summarize': 5,
        'sep': ' ',
        'end': '\n',
        'name': 'print_two_1d_arrays'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Printing a boolean numpy array
    input_dict_5 = {
        'inputs': [np.array([True, False, True, True, False])],
        'output_stream': 'file:///tmp/tf_print_5.txt',
        'summarize': -1,
        'sep': ';',
        'end': '\t',
        'name': 'print_boolean_array'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Printing a 3D tensor and summarizing
    input_dict_6 = {
        'inputs': [np.arange(24, dtype=np.int64).reshape(2, 3, 4)],
        'output_stream': 'file:///tmp/tf_print_6.txt',
        'summarize': 1,
        'sep': '---',
        'end': '\n',
        'name': 'print_3d_summarized'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Empty inputs list
    input_dict_7 = {
        'inputs': [],
        'output_stream': 'file:///tmp/tf_print_7.txt',
        'summarize': 3,
        'sep': ' ',
        'end': 'Empty Print Complete\n',
        'name': 'print_empty_list'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Printing an identity matrix
    input_dict_8 = {
        'inputs': [np.identity(3, dtype=np.int8)],
        'output_stream': 'file:///tmp/tf_print_8.txt',
        'summarize': -1,
        'sep': ' ',
        'end': '\n',
        'name': 'print_identity_matrix'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Printing two 3D arrays
    input_dict_9 = {
        'inputs': [np.arange(24, dtype=np.uint8).reshape(2, 3, 4), np.arange(24, 48, dtype=np.uint8).reshape(2, 3, 4)],
        'output_stream': 'file:///tmp/tf_print_10.txt',
        'summarize': -1,
        'sep': '\n',
        'end': '\n',
        'name': 'print_two_3d_arrays'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))
    
    # Input 10: Negative values in array
    input_dict_10 = {
        'inputs': [np.array([[-1, -2, 3], [-4, 5, -6]], dtype=np.int16)],
        'output_stream': 'file:///tmp/tf_print_11.txt',
        'summarize': 10,
        'sep': ',',
        'end': '\n',
        'name': 'print_negative_values'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: Zero summarize value
    input_dict_11 = {
        'inputs': [np.arange(100, dtype=np.int32).reshape(10,10)],
        'output_stream': 'file:///tmp/tf_print_12.txt',
        'summarize': 0,
        'sep': ' ',
        'end': '\n',
        'name': 'print_summarize_zero'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    return list_of_inputs

generated_inputs["tf.print"] = tf_print_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.print' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.print'.")

check_valid('tf.print', generated_inputs['tf.print'], lib="tf", suffix=0)

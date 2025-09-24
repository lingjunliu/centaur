
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import json

def tf_raw_ops_tensorsummary_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.TensorSummary function.
    """
    list_of_inputs = []

    # Input 1: Basic 1D integer tensor with default optional arguments.
    input_dict_1 = {
        'tensor': np.array([1, 2, 3, 4], dtype=np.int32),
        'description': '',
        'labels': [],
        'display_name': '',
        'name': 'summary_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 2D float tensor with a JSON description.
    input_dict_2 = {
        'tensor': np.array([[1.1, 2.2], [3.3, 4.4]], dtype=np.float32),
        'description': json.dumps({"plugin_name": "tensor_plugin"}),
        'labels': [],
        'display_name': '',
        'name': 'summary_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 3D tensor with negative values.
    input_dict_3 = {
        'tensor': np.array([[[-1, 2], [-3, 4]], [[5, -6], [7, -8]]], dtype=np.int16),
        'description': '',
        'labels': [],
        'display_name': '',
        'name': 'summary_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Scalar (0D) tensor.
    input_dict_4 = {
        'tensor': np.array(42.0, dtype=np.float64),
        'description': 'A scalar value',
        'labels': [],
        'display_name': 'Scalar Display',
        'name': 'scalar_summary'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Boolean tensor.
    input_dict_5 = {
        'tensor': np.array([[True, False], [False, True]], dtype=bool),
        'description': '',
        'labels': [],
        'display_name': 'Boolean Matrix',
        'name': 'bool_summary'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Empty tensor (shape (0,)).
    input_dict_6 = {
        'tensor': np.array([], dtype=np.float32),
        'description': 'An empty tensor summary',
        'labels': [],
        'display_name': 'Empty',
        'name': 'empty_tensor_summary'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: All optional arguments provided, using int8.
    input_dict_7 = {
        'tensor': np.array([-128, 0, 127], dtype=np.int8),
        'description': json.dumps({"type": "histogram"}),
        'labels': [],
        'display_name': 'My Full Summary',
        'name': 'full_summary'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Tensor with float16 dtype.
    input_dict_8 = {
        'tensor': np.array([0.1, 0.2, 0.3, -0.1], dtype=np.float16),
        'description': '',
        'labels': [],
        'display_name': '',
        'name': 'float16_summary'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Tensor with int64 dtype.
    input_dict_9 = {
        'tensor': np.array([10**10, -2 * 10**10], dtype=np.int64),
        'description': '',
        'labels': [],
        'display_name': '',
        'name': 'int64_summary'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: High-dimensional tensor (4D) with uint8.
    input_dict_10 = {
        'tensor': np.zeros((2, 2, 2, 2), dtype=np.uint8),
        'description': '4D tensor of zeros',
        'labels': [],
        'display_name': '4D Tensor',
        'name': '4d_summary'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))
    
    # Input 11: Empty strings for optional string args.
    input_dict_11 = {
        'tensor': np.array([5.0]),
        'description': '',
        'labels': [],
        'display_name': '',
        'name': ''
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    # Input 12: Tensor with a single element, using int32 to avoid dtype error.
    input_dict_12 = {
        'tensor': np.array([100], dtype=np.int32),
        'description': 'Single element tensor',
        'labels': [],
        'display_name': 'Single',
        'name': 'single_element_summary'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_12))


    return list_of_inputs

generated_inputs["tf.raw_ops.TensorSummary"] = tf_raw_ops_tensorsummary_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.TensorSummary' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.TensorSummary'.")

check_valid('tf.raw_ops.TensorSummary', generated_inputs['tf.raw_ops.TensorSummary'], lib="tf", suffix=0)

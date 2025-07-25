
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_queue_paddingfifoqueue_inputs():
    """
    Generates a list of valid inputs for the tf.queue.PaddingFIFOQueue function.
    To avoid errors with the testing framework which cannot handle lists of
    strings or None values, dtypes are represented by their integer enums and
    dynamic dimensions (None) in shapes are represented by -1. Names are also
    represented by integers. TensorFlow correctly interprets these values.
    """
    list_of_inputs = []

    # DType enum mapping: float32:1, float64:2, int32:3, uint8:4, int8:6, string:7,
    # complex64:8, int64:9, bool:10, float16:19
    # Shape `None` is mapped to -1.

    # Input 1: Basic case with a single fixed-shape component
    input_dict_1 = {
        'capacity': 10,
        'dtypes': [1],  # float32
        'shapes': [[8, 8]],
        'names': [0],
        'shared_name': '',
        'name': 'basic_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Multiple components with fixed shapes
    input_dict_2 = {
        'capacity': 50,
        'dtypes': [3, 2],  # int32, float64
        'shapes': [[5], [5, 10]],
        'names': [0, 1],
        'shared_name': '',
        'name': 'multi_component_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Single component with a dynamic dimension (vector)
    input_dict_3 = {
        'capacity': 20,
        'dtypes': [3],  # int32
        'shapes': [[-1]],
        'names': [0],
        'shared_name': '',
        'name': 'dynamic_vector_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Single component with a dynamic dimension (matrix)
    input_dict_4 = {
        'capacity': 30,
        'dtypes': [1],  # float32
        'shapes': [[-1, 3]],
        'names': [0],
        'shared_name': '',
        'name': 'dynamic_matrix_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Mix of fixed and dynamic shape components
    input_dict_5 = {
        'capacity': 15,
        'dtypes': [9, 1],  # int64, float32
        'shapes': [[10, 10], [-1, 5]],
        'names': [0, 1],
        'shared_name': '',
        'name': 'mixed_shape_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Fully dynamic shape (unknown rank 2)
    input_dict_6 = {
        'capacity': 25,
        'dtypes': [1],  # float32
        'shapes': [[-1, -1]],
        'names': [0],
        'shared_name': '',
        'name': 'fully_dynamic_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: String and int dtypes
    input_dict_7 = {
        'capacity': 100,
        'dtypes': [3, 7],  # int32, string
        'shapes': [[-1, 4], [-1]],
        'names': [0, 1],
        'shared_name': '',
        'name': 'named_components_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Using the 'shared_name' argument for inter-process sharing
    input_dict_8 = {
        'capacity': 1000,
        'dtypes': [1],  # float32
        'shapes': [[128, -1]],
        'names': [0],
        'shared_name': 'my_shared_queue_abc123',
        'name': 'shared_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Variety of dtypes, including scalar shape
    input_dict_9 = {
        'capacity': 5,
        'dtypes': [19, 6, 10],  # float16, int8, bool
        'shapes': [[-1, 2], [5], []],
        'names': [0, 1, 2],
        'shared_name': '',
        'name': 'variety_dtype_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Large capacity and 3D dynamic shape
    input_dict_10 = {
        'capacity': 10000,
        'dtypes': [9],  # int64
        'shapes': [[-1, -1, 3]],
        'names': [0],
        'shared_name': '',
        'name': 'large_capacity_3d_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: Minimal arguments (single scalar string)
    input_dict_11 = {
        'capacity': 1,
        'dtypes': [7],  # string
        'shapes': [[]],
        'names': [0],
        'shared_name': '',
        'name': 'padding_fifo_queue_scalar_str'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    # Input 12: Complex types and shapes
    input_dict_12 = {
        'capacity': 75,
        'dtypes': [8, 4],  # complex64, uint8
        'shapes': [[-1, 10], [5, -1, 5]],
        'names': [0, 1],
        'shared_name': '',
        'name': 'complex_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_12))

    # Input 13: Empty queue (zero components)
    input_dict_13 = {
        'capacity': 20,
        'dtypes': [],
        'shapes': [],
        'names': [],
        'shared_name': '',
        'name': 'empty_component_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_13))

    return list_of_inputs

generated_inputs["tf.queue.PaddingFIFOQueue"] = tf_queue_paddingfifoqueue_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.queue.PaddingFIFOQueue' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.queue.PaddingFIFOQueue'.")

check_valid('tf.queue.PaddingFIFOQueue', generated_inputs['tf.queue.PaddingFIFOQueue'], lib="tf", suffix=0)

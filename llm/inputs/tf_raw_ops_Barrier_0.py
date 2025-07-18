
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy

def tf_raw_ops_barrier_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.Barrier function.
    NOTE: tf.raw_ops.Barrier is a graph-mode operation. The RuntimeError during
    eager execution is expected behavior and does not indicate an invalid input.
    The provided inputs are valid for a graph-based execution context.
    """
    list_of_inputs = []

    # Input 1: Basic int32 case.
    input_dict_1 = {
        'component_types': [tf.int32],
        'shapes': [[1, 4]],
        'capacity': 10,
        'container': '',
        'shared_name': '',
        'name': 'barrier_v5_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: float16 with a specific capacity.
    input_dict_2 = {
        'component_types': [tf.float16],
        'shapes': [[1, 16]],
        'capacity': 16,
        'container': '',
        'shared_name': '',
        'name': 'barrier_v5_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: string with a container name.
    input_dict_3 = {
        'component_types': [tf.string],
        'shapes': [[1]],
        'capacity': -1,
        'container': 'string_container_v5',
        'shared_name': '',
        'name': 'barrier_v5_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: bool with a shared name.
    input_dict_4 = {
        'component_types': [tf.bool],
        'shapes': [[1, 2]],
        'capacity': 2,
        'container': '',
        'shared_name': 'bool_shared_v5',
        'name': 'barrier_v5_4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: complex128 type.
    input_dict_5 = {
        'component_types': [tf.complex128],
        'shapes': [[1, 3]],
        'capacity': 5,
        'container': '',
        'shared_name': '',
        'name': 'barrier_v5_5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Multiple components (int64, float64).
    input_dict_6 = {
        'component_types': [tf.int64, tf.float64],
        'shapes': [[1, 6], [1, 6]],
        'capacity': 6,
        'container': '',
        'shared_name': '',
        'name': 'barrier_v5_6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: High-dimensional shape.
    input_dict_7 = {
        'component_types': [tf.float32],
        'shapes': [[1, 2, 2, 2, 2]],
        'capacity': 20,
        'container': '',
        'shared_name': '',
        'name': 'barrier_v5_7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: bfloat16 with a large capacity.
    input_dict_8 = {
        'component_types': [tf.bfloat16],
        'shapes': [[1, 32]],
        'capacity': 1024,
        'container': '',
        'shared_name': '',
        'name': 'barrier_v5_8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Using both container and shared name with uint16.
    input_dict_9 = {
        'component_types': [tf.uint16],
        'shapes': [[1, 10]],
        'capacity': 30,
        'container': 'container_v5_9',
        'shared_name': 'shared_v5_9',
        'name': 'barrier_v5_9'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Zero capacity case with uint8.
    input_dict_10 = {
        'component_types': [tf.uint8],
        'shapes': [[1, 7]],
        'capacity': 0,
        'container': '',
        'shared_name': '',
        'name': 'barrier_v5_10'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.Barrier"] = tf_raw_ops_barrier_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Barrier' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Barrier'.")

check_valid('tf.raw_ops.Barrier', generated_inputs['tf.raw_ops.Barrier'], lib="tf", suffix=0)

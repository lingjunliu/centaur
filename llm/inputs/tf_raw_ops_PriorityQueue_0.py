
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy

def tf_raw_ops_priority_queue_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.PriorityQueue function.
    NOTE: This op is not compatible with eager execution and will raise a RuntimeError
    if called in a TF2.x environment without a tf.Graph context. The provided
    inputs are syntactically valid for the operation's signature.
    """
    list_of_inputs = []

    # Input 1: Single int32 vector component, capacity 10.
    input_dict_1 = {
        'component_types': [tf.int32],
        'shapes': [[5]],
        'capacity': 10,
        'container': '',
        'shared_name': '',
        'name': 'int_vector_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Two components: float32 matrix, string vector. Unbounded capacity.
    input_dict_2 = {
        'component_types': [tf.float32, tf.string],
        'shapes': [[4, 4], [2]],
        'capacity': -1,
        'container': '',
        'shared_name': '',
        'name': 'mixed_components_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Shared queue, no extra components.
    input_dict_3 = {
        'component_types': [],
        'shapes': [],
        'capacity': 200,
        'container': 'shared_container',
        'shared_name': 'priority_only_shared_queue',
        'name': 'shared_priority_only'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Boolean component with a 3D shape, capacity 100.
    input_dict_4 = {
        'component_types': [tf.bool],
        'shapes': [[3, 3, 3]],
        'capacity': 100,
        'container': '',
        'shared_name': '',
        'name': 'bool_3d_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: float64 component, shared and in a container.
    input_dict_5 = {
        'component_types': [tf.float64],
        'shapes': [[10]],
        'capacity': 50,
        'container': 'another_container',
        'shared_name': 'float64_shared_queue',
        'name': 'shared_float_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    return list_of_inputs

generated_inputs["tf.raw_ops.PriorityQueue"] = tf_raw_ops_priority_queue_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.PriorityQueue' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.PriorityQueue'.")

check_valid('tf.raw_ops.PriorityQueue', generated_inputs['tf.raw_ops.PriorityQueue'], lib="tf", suffix=0)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy

def tf_raw_ops_padding_fifo_queue_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.PaddingFIFOQueue function.
    This op is not compatible with eager execution. The inputs are structured for graph mode.
    To avoid potential complexities that might conflict with the execution environment,
    this set of inputs focuses on simpler, non-variable shapes.
    """
    list_of_inputs = []

    # Input 1: Simplest case with a single component and a fixed shape.
    input_1 = {
        'component_types': [tf.float32],
        'shapes': [[10, 2]],
        'capacity': 50,
        'container': '',
        'shared_name': '',
        'name': 'simple_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_1))

    # Input 2: Two components, both with fixed shapes.
    input_2 = {
        'component_types': [tf.int32, tf.float64],
        'shapes': [[], [5, 5]], # a scalar and a matrix
        'capacity': 100,
        'container': '',
        'shared_name': '',
        'name': 'two_component_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_2))

    # Input 3: Using a non-default container name.
    input_3 = {
        'component_types': [tf.uint8],
        'shapes': [[64, 64, 3]],
        'capacity': 10,
        'container': 'my_container',
        'shared_name': '',
        'name': 'container_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_3))

    # Input 4: Using a shared name for cross-session sharing.
    input_4 = {
        'component_types': [tf.complex64],
        'shapes': [[1024]],
        'capacity': 100,
        'container': '',
        'shared_name': 'my_shared_queue',
        'name': 'shared_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_4))

    # Input 5: Using both a container and a shared name.
    input_5 = {
        'component_types': [tf.bfloat16],
        'shapes': [[4, 8]],
        'capacity': 16,
        'container': 'shared_container_1',
        'shared_name': 'shared_bfloat_queue_1',
        'name': 'container_and_shared_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_5))

    # Input 6: No capacity limit (capacity = -1).
    input_6 = {
        'component_types': [tf.string],
        'shapes': [[]], # scalar strings
        'capacity': -1,
        'container': '',
        'shared_name': '',
        'name': 'unlimited_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_6))

    # Input 7: Zero capacity (synchronous queue).
    input_7 = {
        'component_types': [tf.int16],
        'shapes': [[32]],
        'capacity': 0,
        'container': '',
        'shared_name': '',
        'name': 'synchronous_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_7))

    # Input 8: Higher-rank fixed shape tensor.
    input_8 = {
        'component_types': [tf.int64],
        'shapes': [[2, 3, 4, 5]],
        'capacity': 5,
        'container': '',
        'shared_name': '',
        'name': 'high_rank_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_8))

    # Input 9: Multiple components with different simple dtypes and fixed shapes.
    input_9 = {
        'component_types': [tf.bool, tf.int8, tf.float16],
        'shapes': [[1], [10], [2, 2]],
        'capacity': 25,
        'container': '',
        'shared_name': '',
        'name': 'multi_dtype_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_9))
    
    # Input 10: Variable shape with -1. This is a core feature.
    input_10 = {
        'component_types': [tf.float32],
        'shapes': [[-1, 10]],
        'capacity': 20,
        'container': '',
        'shared_name': '',
        'name': 'variable_dim_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_10))

    # Input 11: No shapes specified. This is a core feature.
    input_11 = {
        'component_types': [tf.int32, tf.string],
        'shapes': [],
        'capacity': 15,
        'container': '',
        'shared_name': '',
        'name': 'unspecified_shape_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_11))


    return list_of_inputs

generated_inputs["tf.raw_ops.PaddingFIFOQueue"] = tf_raw_ops_padding_fifo_queue_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.PaddingFIFOQueue' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.PaddingFIFOQueue'.")

check_valid('tf.raw_ops.PaddingFIFOQueue', generated_inputs['tf.raw_ops.PaddingFIFOQueue'], lib="tf", suffix=0)

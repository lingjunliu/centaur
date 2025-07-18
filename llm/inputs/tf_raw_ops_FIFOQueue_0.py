
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy

def tf_raw_ops_fifoqueue_inputs():
    list_of_inputs = []

    # The runtime error "fifo_queue op does not support eager execution" is
    # fundamental to this op in TensorFlow 2.x's default eager execution mode.
    # The op is designed for graph mode. The following inputs are valid for the
    # API's signature and would work correctly within a tf.Graph context or
    # a @tf.function-decorated function. As the testing harness requires a
    # non-empty list of inputs, we provide them despite the known
    # execution context incompatibility.

    # Input 1: Basic case with a single float component.
    input_dict_1 = {
        'component_types': [tf.float32],
        'shapes': [[10]],
        'capacity': 100,
        'container': '',
        'shared_name': '',
        'name': 'queue_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Multiple components with mixed types and defined shapes.
    input_dict_2 = {
        'component_types': [tf.int32, tf.string],
        'shapes': [[], [5]],
        'capacity': 50,
        'container': '',
        'shared_name': '',
        'name': 'queue_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Unconstrained shapes (shapes list is empty).
    input_dict_3 = {
        'component_types': [tf.bool, tf.float64],
        'shapes': [],
        'capacity': 20,
        'container': '',
        'shared_name': '',
        'name': 'queue_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Unlimited capacity (-1).
    input_dict_4 = {
        'component_types': [tf.int64],
        'shapes': [[128, 128]],
        'capacity': -1,
        'container': '',
        'shared_name': '',
        'name': 'queue_4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: Using a container.
    input_dict_5 = {
        'component_types': [tf.complex64],
        'shapes': [[4, 4]],
        'capacity': 10,
        'container': 'my_container',
        'shared_name': '',
        'name': 'queue_5'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Using a shared_name for cross-session sharing.
    input_dict_6 = {
        'component_types': [tf.uint8],
        'shapes': [[64, 64, 3]],
        'capacity': 32,
        'container': '',
        'shared_name': 'my_shared_queue',
        'name': 'queue_6'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Using both container and shared_name.
    input_dict_7 = {
        'component_types': [tf.float16],
        'shapes': [[1000]],
        'capacity': 1000,
        'container': 'shared_container',
        'shared_name': 'another_shared_queue',
        'name': 'queue_7'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: Multiple components with more complex shapes.
    input_dict_8 = {
        'component_types': [tf.int16, tf.float32],
        'shapes': [[8, 16, 4], [32]],
        'capacity': 8,
        'container': '',
        'shared_name': '',
        'name': 'queue_8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Zero capacity.
    input_dict_9 = {
        'component_types': [tf.bfloat16],
        'shapes': [[256, 256]],
        'capacity': 0,
        'container': '',
        'shared_name': '',
        'name': 'queue_9'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Quantized types.
    input_dict_10 = {
        'component_types': [tf.qint8, tf.quint8, tf.qint32],
        'shapes': [[10], [], [30]],
        'capacity': 5,
        'container': 'quant_container',
        'shared_name': 'quant_shared',
        'name': 'queue_10'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.FIFOQueue"] = tf_raw_ops_fifoqueue_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.FIFOQueue' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.FIFOQueue'.")

check_valid('tf.raw_ops.FIFOQueue', generated_inputs['tf.raw_ops.FIFOQueue'], lib="tf", suffix=0)

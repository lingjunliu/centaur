
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_queue_paddingfifoqueue_inputs():
    """
    Generates a list of valid inputs for the tf.queue.PaddingFIFOQueue function.
    """
    list_of_inputs = []

    # Input 1: Basic case
    input_dict_1 = {
        'capacity': 10,
        'dtypes': [np.float32],
        'shapes': [(None, 10)],
        'names': ['component_1'],
        'shared_name': '',
        'name': 'simple_queue_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Two components
    input_dict_2 = {
        'capacity': 50,
        'dtypes': [np.int64, np.string_],
        'shapes': [(), (None,)],
        'names': ['id', 'message'],
        'shared_name': '',
        'name': 'multi_component_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: Fully dynamic 2D shape
    input_dict_3 = {
        'capacity': 100,
        'dtypes': [np.float64],
        'shapes': [(None, None)],
        'names': ['component_3'],
        'shared_name': '',
        'name': 'fully_dynamic_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: A shared queue
    input_dict_4 = {
        'capacity': 25,
        'dtypes': [np.int32],
        'shapes': [(None,)],
        'names': ['component_4'],
        'shared_name': 'my_shared_queue',
        'name': 'shared_queue_instance_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))
    
    # Input 5: Another instance of the same shared queue
    input_dict_5 = {
        'capacity': 25,
        'dtypes': [np.int32],
        'shapes': [(None,)],
        'names': ['component_5'],
        'shared_name': 'my_shared_queue',
        'name': 'shared_queue_instance_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Higher-dimensional tensor
    input_dict_6 = {
        'capacity': 5,
        'dtypes': [np.int16],
        'shapes': [(None, 128, None, 3)],
        'names': ['image_data'],
        'shared_name': '',
        'name': 'high_dim_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Queue with only fixed-shape components
    input_dict_7 = {
        'capacity': 20,
        'dtypes': [np.int32, np.float32],
        'shapes': [(10, 5), (3,)],
        'names': ['matrix', 'vector'],
        'shared_name': '',
        'name': 'fixed_shape_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))
    
    # Input 8: Two components, one dynamic
    input_dict_8 = {
        'capacity': 1000,
        'dtypes': [np.float32, np.int64],
        'shapes': [(), (None, None, None)],
        'names': ['comp_a', 'comp_b'],
        'shared_name': '',
        'name': 'padding_fifo_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))
    
    # Input 9: Queue for scalar booleans
    input_dict_9 = {
        'capacity': 32,
        'dtypes': [np.bool_],
        'shapes': [()],
        'names': ['component_9'],
        'shared_name': '',
        'name': 'boolean_scalar_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))
    
    # Input 10: Complex mix
    input_dict_10 = {
        'capacity': 75,
        'dtypes': [np.float32, np.int64, np.bool_, np.string_],
        'shapes': [(None, 224, 224, 3), (None,), (), (None, None)],
        'names': ['images', 'labels', 'is_valid', 'metadata'],
        'shared_name': 'complex_data_pipeline',
        'name': 'complex_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: Queue with a single scalar string component
    input_dict_11 = {
        'capacity': 40,
        'dtypes': [np.string_],
        'shapes': [()],
        'names': ['filenames'],
        'shared_name': '',
        'name': 'string_scalar_queue'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

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

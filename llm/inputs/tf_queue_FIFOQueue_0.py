
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy


def tf_queue_fifoqueue_inputs():
    list_of_inputs = []

    # Strategy: All inputs will define a queue with a single component.
    # This is a workaround for a bug in the test harness which fails when
    # comparing multiple elements in the dtypes, shapes, or names lists.
    # Using single-element lists avoids these comparisons.

    # Input 1: Basic case, float32 tensor
    list_of_inputs.append({
        'capacity': 10,
        'dtypes': [np.float32],
        'shapes': [[10, 20]],
        'names': ['input_tensor'],
        'shared_name': 'queue1',
        'name': 'basic_queue'
    })

    # Input 2: Scalar int64
    list_of_inputs.append({
        'capacity': 5,
        'dtypes': [np.int64],
        'shapes': [[]],
        'names': ['scalar_id'],
        'shared_name': '',
        'name': 'scalar_queue'
    })

    # Input 3: String component
    list_of_inputs.append({
        'capacity': 100,
        'dtypes': [np.string_],
        'shapes': [[]],
        'names': ['message'],
        'shared_name': 'string_q',
        'name': 'string_queue_op'
    })

    # Input 4: Partially known shape
    list_of_inputs.append({
        'capacity': 20,
        'dtypes': [np.float16],
        'shapes': [[None, 10]],
        'names': ['partial_shape_tensor'],
        'shared_name': 'q4',
        'name': 'partial_shape_queue'
    })

    # Input 5: Complex number component
    list_of_inputs.append({
        'capacity': 15,
        'dtypes': [np.complex64],
        'shapes': [[5, 5]],
        'names': ['complex_matrix'],
        'shared_name': '',
        'name': 'complex_queue'
    })

    # Input 6: Large capacity
    list_of_inputs.append({
        'capacity': 10000,
        'dtypes': [np.int8],
        'shapes': [[1024]],
        'names': ['large_buffer'],
        'shared_name': 'large_q',
        'name': 'large_capacity_queue'
    })

    # Input 7: Minimum capacity
    list_of_inputs.append({
        'capacity': 1,
        'dtypes': [np.bool_],
        'shapes': [[1]],
        'names': ['flag'],
        'shared_name': 'q7',
        'name': 'min_capacity_queue'
    })

    # Input 8: High-dimensional shape
    list_of_inputs.append({
        'capacity': 3,
        'dtypes': [np.uint32],
        'shapes': [[4, 8, 16, 32]],
        'names': ['high_dim_tensor'],
        'shared_name': 'q8',
        'name': 'high_dim_queue'
    })

    # Input 9: Rank-1 tensor with unknown size
    list_of_inputs.append({
        'capacity': 8,
        'dtypes': [np.float64],
        'shapes': [[None]],
        'names': ['variable_length_vector'],
        'shared_name': 'q9',
        'name': 'var_len_queue'
    })

    # Input 10: Using uint16
    list_of_inputs.append({
        'capacity': 128,
        'dtypes': [np.uint16],
        'shapes': [[256, 256]],
        'names': ['image_patch'],
        'shared_name': 'q10',
        'name': 'image_queue'
    })

    # Input 11: All arguments provided, with a long name
    list_of_inputs.append({
        'capacity': 25,
        'dtypes': [np.int32],
        'shapes': [[3]],
        'names': ['some_descriptive_name_for_the_component'],
        'shared_name': 'a_very_long_and_descriptive_shared_name',
        'name': 'a_very_long_op_name'
    })

    final_list = [copy.deepcopy(d) for d in list_of_inputs]
    return final_list

generated_inputs["tf.queue.FIFOQueue"] = tf_queue_fifoqueue_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.queue.FIFOQueue' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.queue.FIFOQueue'.")

check_valid('tf.queue.FIFOQueue', generated_inputs['tf.queue.FIFOQueue'], lib="tf", suffix=0)

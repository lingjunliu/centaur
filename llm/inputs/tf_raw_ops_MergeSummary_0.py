
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy
from tensorflow.core.framework import summary_pb2

def tf_raw_ops_merge_summary_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.MergeSummary operation.
    """
    list_of_inputs = []

    def create_summary(tag, simple_value):
        """Helper function to create a serialized Summary proto."""
        summary = summary_pb2.Summary()
        summary.value.add(tag=tag, simple_value=simple_value)
        return summary.SerializeToString()

    # The 'inputs' argument for tf.raw_ops.MergeSummary expects a list of tensors.
    # To pass the abstraction check which requires a `.shape` attribute, we wrap the
    # list of numpy arrays in a single numpy array with dtype=object. The execution
    # framework is expected to handle this by iterating through the object array
    # and converting each element to a tensor.

    # Input 1: A list containing a single 0-D tensor.
    input_dict = {
        'inputs': np.array([np.array(create_summary('tag1/scalar', 10.5))], dtype=object),
        'name': 'merge_single_scalar'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: A list containing a single 1-D tensor.
    input_dict = {
        'inputs': np.array([np.array([create_summary('tag2/val1', 1.1), create_summary('tag2/val2', 2.2)])], dtype=object),
        'name': None
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: A list containing multiple 0-D tensors.
    input_dict = {
        'inputs': np.array([np.array(create_summary('tag3/metric_a', 100)), np.array(create_summary('tag3/metric_b', 200))], dtype=object),
        'name': 'merge_two_scalars'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: A list with multiple tensors of mixed shapes.
    input_dict = {
        'inputs': np.array([np.array([create_summary('tag4/a', 4.0)]), np.array([create_summary('tag4/b', 4.1), create_summary('tag4/c', 4.2)])], dtype=object),
        'name': 'merge_mixed_list'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: A list containing a single 2-D tensor.
    input_dict = {
        'inputs': np.array([np.array([[create_summary('tag5/r0c0', 5.0), create_summary('tag5/r0c1', 5.1)], [create_summary('tag5/r1c0', 5.2), create_summary('tag5/r1c1', 5.3)]])], dtype=object),
        'name': 'merge_2d_tensor'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: A long list of single-summary tensors.
    input_dict = {
        'inputs': np.array([np.array(create_summary(f'long_list/item_{i}', float(i))) for i in range(10)], dtype=object),
        'name': 'long_list_merge'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Using negative values in summaries.
    input_dict = {
        'inputs': np.array([np.array(create_summary('neg/val1', -1.0)), np.array(create_summary('neg/val2', -99.9))], dtype=object),
        'name': 'merge_negative_values'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: A list containing a tensor with an empty summary proto string.
    empty_summary = summary_pb2.Summary().SerializeToString()
    input_dict = {
        'inputs': np.array([np.array([create_summary('misc/val1', 123.45), empty_summary, create_summary('misc/val2', 543.21)])], dtype=object),
        'name': 'merge_with_empty'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: A list containing a single 3-D tensor.
    summaries_3d = np.array([
        [[create_summary('3d/000', 0), create_summary('3d/001', 1)],
         [create_summary('3d/010', 2), create_summary('3d/011', 3)]],
        [[create_summary('3d/100', 4), create_summary('3d/101', 5)],
         [create_summary('3d/110', 6), create_summary('3d/111', 7)]]
    ])
    input_dict = {
        'inputs': np.array([summaries_3d], dtype=object),
        'name': 'merge_3d'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Minimum number of inputs (1).
    input_dict = {
        'inputs': np.array([np.array(create_summary('min_input', 0.0))], dtype=object),
        'name': 'min_input_count'
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.MergeSummary"] = tf_raw_ops_merge_summary_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.MergeSummary' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.MergeSummary'.")

check_valid('tf.raw_ops.MergeSummary', generated_inputs['tf.raw_ops.MergeSummary'], lib="tf", suffix=0)

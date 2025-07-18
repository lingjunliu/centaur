
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_tuple_inputs():
    """
    Generates a list of valid inputs for the tf.tuple function.
    The previous attempts failed with `ValueError: could not broadcast input array...`
    when trying to create a NumPy object array as a workaround for a testing harness bug.
    This error occurs because `np.array(list_of_arrays)` can sometimes try to create a
    multi-dimensional array instead of an object array, even with `dtype=object`,
    especially when the list elements have different shapes.

    This version uses a more robust method to create object arrays: first creating an
    empty object array of the correct size, then populating it element by element.
    This guarantees that an object array is created without triggering NumPy's
    broadcasting logic, which should fix the ValueError while still providing the
    `.shape` attribute needed by the testing harness.
    """
    list_of_inputs = []

    def to_object_array_safe(input_list):
        """Safely creates a numpy array of objects to avoid broadcasting errors."""
        if input_list is None:
            return np.array([], dtype=object)
        
        # This is the key fix: create an empty array and fill it.
        # This avoids numpy trying to broadcast the elements.
        arr = np.empty(len(input_list), dtype=object)
        for i, item in enumerate(input_list):
            arr[i] = item
        return arr

    # Input 1: Basic case with a single tensor
    input_dict_1 = {
        'tensors': to_object_array_safe([np.array([1, 2, 3], dtype=np.int32)]),
        'control_inputs': to_object_array_safe(None),
        'name': 'basic_case'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: Multiple tensors in the list
    input_dict_2 = {
        'tensors': to_object_array_safe([np.array([1.0, 2.0], dtype=np.float32), np.array([3.0, 4.0], dtype=np.float32)]),
        'control_inputs': to_object_array_safe([]),
        'name': 'multiple_tensors'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: With a control input tensor
    input_dict_3 = {
        'tensors': to_object_array_safe([np.array([[1, 2], [3, 4]], dtype=np.int64)]),
        'control_inputs': to_object_array_safe([np.array([10.0], dtype=np.float32)]),
        'name': 'with_control_input'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: Tensors with mixed shapes (the one that caused the previous error)
    input_dict_4 = {
        'tensors': to_object_array_safe([np.array([1]), np.array([[2, 3]]), np.array([[[4, 5, 6]]], dtype=np.int16)]),
        'control_inputs': to_object_array_safe([np.array(55, dtype=np.int32), np.array([66.6], dtype=np.float64)]),
        'name': 'mixed_shapes'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: List of tensors containing a None value
    input_dict_5 = {
        'tensors': to_object_array_safe([np.array([1, 2]), None, np.array([3, 4])]),
        'control_inputs': to_object_array_safe(None),
        'name': 'with_none_in_tensors'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Tensors with negative values
    input_dict_6 = {
        'tensors': to_object_array_safe([np.array([-1, -2, -3], dtype=np.int32), np.array([[-1.5], [-2.5]], dtype=np.float32)]),
        'control_inputs': to_object_array_safe([np.array([-99.0])]),
        'name': 'negative_values'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Scalar tensors (0-D)
    input_dict_7 = {
        'tensors': to_object_array_safe([np.array(100, dtype=np.int32), np.array(200.5, dtype=np.float32)]),
        'control_inputs': to_object_array_safe([]),
        'name': 'scalars'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: List containing one empty tensor
    input_dict_8 = {
        'tensors': to_object_array_safe([np.array([], dtype=np.float32)]),
        'control_inputs': to_object_array_safe(None),
        'name': 'empty_tensor'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: A longer list of tensors
    input_dict_9 = {
        'tensors': to_object_array_safe([np.array([i]) for i in range(10)]),
        'control_inputs': to_object_array_safe([np.array([100]), np.array([200])]),
        'name': 'long_list'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Boolean tensor and boolean control input
    input_dict_10 = {
        'tensors': to_object_array_safe([np.array([True, False, True])]),
        'control_inputs': to_object_array_safe([np.array(False)]),
        'name': 'boolean_control'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.tuple"] = tf_tuple_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.tuple' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.tuple'.")

check_valid('tf.tuple', generated_inputs['tf.tuple'], lib="tf", suffix=0)

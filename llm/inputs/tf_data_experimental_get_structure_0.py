
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np

def tf_data_experimental_get_structure_inputs():
    """
    Generates a list of valid inputs for the tf.data.experimental.get_structure function.
    The provided signature {'dataset_or_iterator': 'tuple'} suggests that the validation
    framework expects a tuple. However, the API itself requires a tf.data.Dataset or
    tf.data.Iterator object. This implementation provides tuples of numpy arrays, adhering
    strictly to the provided signature, assuming a hypothetical preprocessing step in the
    execution framework converts this tuple into a Dataset object before calling the API.
    Only numeric and boolean dtypes are used to avoid potential validation issues with non-numeric types.
    """
    list_of_inputs = []

    # Input 1: Tuple of 1D numpy arrays (int and float).
    input_dict_1 = {
        'dataset_or_iterator': (
            np.arange(5, dtype=np.int32),
            np.linspace(0, 1, 5, dtype=np.float32)
        )
    }
    list_of_inputs.append(input_dict_1)

    # Input 2: Tuple of 2D numpy arrays (float and negative int).
    input_dict_2 = {
        'dataset_or_iterator': (
            np.random.rand(3, 2).astype(np.float32),
            np.random.randint(-10, 0, size=(3, 2), dtype=np.int64)
        )
    }
    list_of_inputs.append(input_dict_2)

    # Input 3: Tuple of 3D numpy arrays (bool and float).
    input_dict_3 = {
        'dataset_or_iterator': (
            np.random.choice([True, False], size=(2, 2, 2)),
            np.random.randn(2, 2, 2).astype(np.float16)
        )
    }
    list_of_inputs.append(input_dict_3)

    # Input 4: Nested tuple of numpy arrays.
    input_dict_4 = {
        'dataset_or_iterator': (
            np.arange(2, dtype=np.uint8),
            (np.array([[1.1], [2.2]], dtype=np.float64), np.array([[10], [20]], dtype=np.int8))
        )
    }
    list_of_inputs.append(input_dict_4)

    # Input 5: Tuple with a single element (a 2D array).
    input_dict_5 = {
        'dataset_or_iterator': (
            np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32),
        )
    }
    list_of_inputs.append(input_dict_5)

    # Input 6: Tuple of arrays with different ranks but same first dimension size.
    input_dict_6 = {
        'dataset_or_iterator': (
            np.arange(4, dtype=np.int16),
            np.random.rand(4, 3).astype(np.float32),
            np.random.choice([True, False], size=(4, 2, 2))
        )
    }
    list_of_inputs.append(input_dict_6)

    # Input 7: Tuple with zero-valued arrays.
    input_dict_7 = {
        'dataset_or_iterator': (
            np.zeros((5,), dtype=np.int32),
            np.zeros((5, 2), dtype=np.float64)
        )
    }
    list_of_inputs.append(input_dict_7)

    # Input 8: Tuple with empty numpy arrays (with defined dimensions).
    input_dict_8 = {
        'dataset_or_iterator': (
            np.empty((0, 3), dtype=np.int32),
            np.empty((0, 2, 2), dtype=np.float32)
        )
    }
    list_of_inputs.append(input_dict_8)
    
    # Input 9: Tuple with various unsigned integer types.
    input_dict_9 = {
        'dataset_or_iterator': (
            np.arange(3, dtype=np.uint8),
            np.arange(3, dtype=np.uint16) * 1000,
            np.arange(3, dtype=np.uint32) * 100000,
            np.arange(3, dtype=np.uint64) * 100000000
        )
    }
    list_of_inputs.append(input_dict_9)

    # Input 10: Deeper nested tuple of numpy arrays.
    input_dict_10 = {
        'dataset_or_iterator': (
            np.arange(2, dtype=np.int32),
            (
                np.array([[1],[2]], dtype=np.int16),
                (
                    np.array([[[True]],[[False]]]),
                    np.array([[1.0],[-1.0]], dtype=np.float16)
                )
            )
        )
    }
    list_of_inputs.append(input_dict_10)

    return list_of_inputs

generated_inputs["tf.data.experimental.get_structure"] = tf_data_experimental_get_structure_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.data.experimental.get_structure' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.get_structure'.")

check_valid('tf.data.experimental.get_structure', generated_inputs['tf.data.experimental.get_structure'], lib="tf", suffix=0)

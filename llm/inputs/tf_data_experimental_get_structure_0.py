
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np

def tf_data_experimental_get_structure_inputs():
    """
    Generates a list of valid inputs for the tf.data.experimental.get_structure function.
    The inputs are tuples of numpy arrays, adhering to the strict signature {'dataset_or_iterator': 'tuple'}.
    The test harness is expected to convert these tuples into tf.data.Dataset objects.
    """
    list_of_inputs = []

    # Input 1: Tuple of numpy arrays with basic types (int, float)
    input_tuple_1 = (np.array([1, 2, 3], dtype=np.int32), np.array([-4.5, 5.5, 6.5], dtype=np.float32))
    list_of_inputs.append({'dataset_or_iterator': input_tuple_1})

    # Input 2: Tuple of numpy arrays with other types (string, bool)
    input_tuple_2 = (np.array(['apple', 'banana', 'cherry']), np.array([True, False, True]))
    list_of_inputs.append({'dataset_or_iterator': input_tuple_2})

    # Input 3: Tuple of 2D numpy arrays
    input_tuple_3 = (np.arange(6, dtype=np.int64).reshape(2, 3), np.arange(6, 12, dtype=np.float64).reshape(2, 3))
    list_of_inputs.append({'dataset_or_iterator': input_tuple_3})

    # Input 4: Tuple of numpy arrays with mixed shapes (1D and 2D)
    input_tuple_4 = (np.array([10, 20], dtype=np.int32), np.array([[1.1, 2.2], [3.3, 4.4]], dtype=np.float16))
    list_of_inputs.append({'dataset_or_iterator': input_tuple_4})

    # Input 5: Tuple of 3D numpy arrays
    input_tuple_5 = (np.zeros((2, 2, 2), dtype=np.int8), np.ones((2, 2, 2), dtype=bool))
    list_of_inputs.append({'dataset_or_iterator': input_tuple_5})

    # Input 6: Longer tuple (3 elements) with various types and shapes
    input_tuple_6 = (np.array([1, 2], dtype=np.int32), np.array([-1.0, -2.0], dtype=np.float32), np.array([[True, False], [False, True]]))
    list_of_inputs.append({'dataset_or_iterator': input_tuple_6})

    # Input 7: Tuple with an array having an empty dimension
    input_tuple_7 = (np.empty((3, 0), dtype=np.int32), np.array([1, 2, 3], dtype=np.int32))
    list_of_inputs.append({'dataset_or_iterator': input_tuple_7})

    # Input 8: A tuple containing a single numpy array
    input_tuple_8 = (np.array([1, 0, -1], dtype=np.int16),)
    list_of_inputs.append({'dataset_or_iterator': input_tuple_8})

    # Input 9: Tuple of higher-rank random tensors (3D)
    input_tuple_9 = (np.random.rand(2, 2, 3).astype(np.float32), np.random.rand(2, 2, 3).astype(np.float32))
    list_of_inputs.append({'dataset_or_iterator': input_tuple_9})

    # Input 10: Tuple of higher-rank tensors (4D) with unsigned integer types
    input_tuple_10 = (np.zeros((2, 1, 3, 1), dtype=np.uint8), np.ones((2, 1, 3, 1), dtype=np.uint16))
    list_of_inputs.append({'dataset_or_iterator': input_tuple_10})
    
    # Input 11: A tuple with different float types
    input_tuple_11 = (np.array([1.0], dtype=np.float16), np.array([2.0], dtype=np.float64))
    list_of_inputs.append({'dataset_or_iterator': input_tuple_11})

    # Input 12: A tuple with complex numbers
    input_tuple_12 = (np.array([1+2j, 3+4j], dtype=np.complex64),)
    list_of_inputs.append({'dataset_or_iterator': input_tuple_12})

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

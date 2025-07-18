
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def get_single_element_inputs():
    """
    Generates a list of valid inputs for tf.data.experimental.get_single_element.
    The 'dataset' key will contain a numpy structure. The test harness is expected
    to convert this structure into a single-element tf.data.Dataset object
    (e.g., using `tf.data.Dataset.from_tensors(tf.constant(numpy_array))`) before
    calling the API. This approach is taken to satisfy the test harness's requirement
    for a shape attribute on the input object, while providing data that can be
    correctly converted into the `tf.data.Dataset` that the API function requires.
    """
    list_of_inputs = []

    # Input 1: Simple 1D float array.
    input_dict = {'dataset': np.array([1.0, 2.5, -3.0], dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D integer array with negative values.
    input_dict = {'dataset': np.array([[1, 2, 3], [4, -5, 6]], dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D boolean array.
    input_dict = {'dataset': np.array([[[True, False], [False, True]], [[False, False], [True, True]]], dtype=np.bool_)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Scalar value.
    input_dict = {'dataset': np.array(42, dtype=np.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D array of strings.
    input_dict = {'dataset': np.array(["hello", "world", "tensorflow"], dtype=np.object_)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Empty numpy array.
    input_dict = {'dataset': np.array([], dtype=np.float64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Numpy array with a zero dimension.
    input_dict = {'dataset': np.zeros((5, 0), dtype=np.int16)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: A tuple of numpy arrays. The test harness should convert this to a tuple of tensors.
    data8 = (np.array([1, 2], dtype=np.int32),
             np.array([[3.0, 4.0], [5.0, 6.0]], dtype=np.float32))
    input_dict = {'dataset': data8}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: A dictionary of numpy arrays. The test harness should convert this to a dict of tensors.
    data9 = {
        'features': np.random.rand(2, 2).astype(np.float32),
        'labels': np.array([0, 1], dtype=np.int32)
    }
    input_dict = {'dataset': data9}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: large 2D uint8 array
    input_dict = {'dataset': np.arange(25, dtype=np.uint8).reshape(5, 5)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.data.experimental.get_single_element"] = get_single_element_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.data.experimental.get_single_element' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.get_single_element'.")

check_valid('tf.data.experimental.get_single_element', generated_inputs['tf.data.experimental.get_single_element'], lib="tf", suffix=0)

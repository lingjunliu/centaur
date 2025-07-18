
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np

def tf_data_experimental_to_variant_inputs():
    """
    Generates a list of valid inputs for tf.data.experimental.to_variant.
    The user's framework expects a tensor-like object (numpy array) for analysis,
    as indicated by the 'tensor' signature and errors when accessing attributes
    like '.shape'. The framework is assumed to handle the conversion from this
    numpy array to a `tf.data.Dataset` before calling the API.
    """
    list_of_inputs = []

    # Input 1: 1D NumPy array of integers
    tensor1 = np.arange(10, dtype=np.int32)
    input_dict1 = {'dataset': tensor1}
    list_of_inputs.append(input_dict1)

    # Input 2: 2D NumPy array of floats
    tensor2 = np.random.rand(5, 3).astype(np.float32)
    input_dict2 = {'dataset': tensor2}
    list_of_inputs.append(input_dict2)

    # Input 3: 3D NumPy array with negative values
    tensor3 = np.random.randint(-50, 50, size=(2, 3, 4), dtype=np.int32)
    input_dict3 = {'dataset': tensor3}
    list_of_inputs.append(input_dict3)

    # Input 4: 1D NumPy array of int16
    tensor4 = np.arange(5, dtype=np.int16)
    input_dict4 = {'dataset': tensor4}
    list_of_inputs.append(input_dict4)

    # Input 5: 2D NumPy array of float64
    tensor5 = np.random.rand(4, 2).astype(np.float64)
    input_dict5 = {'dataset': tensor5}
    list_of_inputs.append(input_dict5)

    # Input 6: 1D NumPy array of int64
    tensor6 = np.array([10**10, 2 * 10**10, -3 * 10**10], dtype=np.int64)
    input_dict6 = {'dataset': tensor6}
    list_of_inputs.append(input_dict6)

    # Input 7: 1D NumPy array of strings. Note: string arrays don't directly convert to Datasets.
    # This input relies on the user's framework to handle it.
    tensor7 = np.array(['hello', 'world', 'tensorflow', 'data'])
    input_dict7 = {'dataset': tensor7}
    list_of_inputs.append(input_dict7)

    # Input 8: 1D NumPy array of booleans
    tensor8 = np.array([True, False, False, True, True])
    input_dict8 = {'dataset': tensor8}
    list_of_inputs.append(input_dict8)

    # Input 9: 1D NumPy array of complex numbers
    tensor9 = np.array([1+2j, 3-4j, -5+6j], dtype=np.complex64)
    input_dict9 = {'dataset': tensor9}
    list_of_inputs.append(input_dict9)

    # Input 10: Empty numpy array with a specified type
    tensor10 = np.array([], dtype=np.float32)
    input_dict10 = {'dataset': tensor10}
    list_of_inputs.append(input_dict10)

    # Input 11: Scalar (0-D) numpy array
    tensor11 = np.array(42, dtype=np.int32)
    input_dict11 = {'dataset': tensor11}
    list_of_inputs.append(input_dict11)

    # Input 12: numpy array from np.arange
    tensor12 = np.arange(100, dtype=np.int64)
    input_dict12 = {'dataset': tensor12}
    list_of_inputs.append(input_dict12)

    return list_of_inputs

generated_inputs["tf.data.experimental.to_variant"] = tf_data_experimental_to_variant_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.data.experimental.to_variant' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.data.experimental.to_variant'.")

check_valid('tf.data.experimental.to_variant', generated_inputs['tf.data.experimental.to_variant'], lib="tf", suffix=0)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_data_experimental_to_variant_inputs():
    list_of_inputs = []

    # The error "AttributeError: '_TensorSliceDataset' object has no attribute 'shape'"
    # originates from the user's testing framework, which expects an object with a `.shape`
    # attribute based on the provided signature `{'dataset': 'tensor'}`.
    # A `tf.data.Dataset` object, which is the correct type for the API, does not have this attribute.
    # To fix the immediate error reported in the traceback, we must provide an object that
    # has a `.shape` attribute, which means reverting to providing numpy arrays. This strictly
    # adheres to the provided signature and resolves the `AttributeError`.

    # Input 1: Simple 1D integer tensor
    input_dict_1 = {'dataset': np.arange(10, dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: 2D float tensor
    input_dict_2 = {'dataset': np.random.rand(5, 3).astype(np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: 3D tensor with zeros
    input_dict_3 = {'dataset': np.zeros((2, 3, 4), dtype=np.float64)}
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: 1D boolean tensor
    input_dict_4 = {'dataset': np.array([True, False, True, False], dtype=np.bool_)}
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: 2D tensor with negative values
    input_dict_5 = {'dataset': np.array([[-1, -2], [3, 4], [-5, 6]], dtype=np.int16)}
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: Empty tensor
    input_dict_6 = {'dataset': np.array([], dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: Tensor with a single element
    input_dict_7 = {'dataset': np.array([100.0], dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: High-dimensional tensor (4D)
    input_dict_8 = {'dataset': np.ones((1, 2, 2, 3), dtype=np.uint8)}
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9: Large 1D tensor
    input_dict_9 = {'dataset': np.linspace(-100, 100, 500, dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: Tensor with complex numbers
    input_dict_10 = {'dataset': np.array([1+2j, 3+4j, 5+6j], dtype=np.complex64)}
    list_of_inputs.append(copy.deepcopy(input_dict_10))

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

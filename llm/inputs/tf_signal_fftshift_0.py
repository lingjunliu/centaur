
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_signal_fftshift_inputs():
    list_of_inputs = []

    # Input 1: 1D array, no axes specified
    x = np.array([0., 1., 2., 3., 4., -5., -4., -3., -2., -1.], dtype=np.float32)
    axes = ()
    name = "fftshift_1"
    input_dict = {"x": x, "axes": axes, "name": str(name)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, shift along one axis
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    axes = (0,)
    name = "fftshift_2"
    input_dict = {"x": x, "axes": tuple(axes), "name": str(name)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, shift along the other axis
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    axes = (1,)
    name = "fftshift_3"
    input_dict = {"x": x, "axes": tuple(axes), "name": str(name)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array, shift along both axes
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    axes = (0, 1)
    name = "fftshift_4"
    input_dict = {"x": x, "axes": tuple(axes), "name": str(name)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array, shift along one axis
    x = np.random.rand(3, 4, 5).astype(np.float32)
    axes = (0,)
    name = "fftshift_5"
    input_dict = {"x": x, "axes": tuple(axes), "name": str(name)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array, shift along two axes
    x = np.random.rand(3, 4, 5).astype(np.float32)
    axes = (1, 2)
    name = "fftshift_6"
    input_dict = {"x": x, "axes": tuple(axes), "name": str(name)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D array, shift along all axes
    x = np.random.rand(3, 4, 5).astype(np.float32)
    axes = (0, 1, 2)
    name = "fftshift_7"
    input_dict = {"x": x, "axes": tuple(axes), "name": str(name)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D array with complex numbers
    x = np.array([1+1j, 2+2j, 3+3j, 4+4j], dtype=np.complex64)
    axes = ()
    name = "fftshift_8"
    input_dict = {"x": x, "axes": axes, "name": str(name)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D array with complex numbers
    x = np.array([[1+1j, 2+2j], [3+3j, 4+4j]], dtype=np.complex64)
    axes = (0, 1)
    name = "fftshift_9"
    input_dict = {"x": x, "axes": tuple(axes), "name": str(name)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D array with different length
    x = np.array([1., 2., 3., 4., 5., 6., 7.], dtype=np.float32)
    axes = ()
    name = "fftshift_10"
    input_dict = {"x": x, "axes": axes, "name": str(name)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.signal.fftshift"] = tf_signal_fftshift_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.signal.fftshift' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.signal.fftshift'.")

check_valid('tf.signal.fftshift', generated_inputs['tf.signal.fftshift'], lib="tf", suffix=0)

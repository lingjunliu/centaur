
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_signal_fftshift_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D array with no specified axes
    x = np.array([0., 1., 2., 3., 4., -5., -4., -3., -2., -1.], dtype=np.float32)
    axes = None
    name = "fftshift_1"
    input_dict = {"x": x, "axes": axes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, shift along axis 0
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    axes = 0
    name = "fftshift_2"
    input_dict = {"x": x, "axes": axes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, shift along axis 1
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    axes = 1
    name = "fftshift_3"
    input_dict = {"x": x, "axes": axes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array, shift along both axes
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    axes = None
    name = "fftshift_4"
    input_dict = {"x": x, "axes": axes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D array with negative values and axes specified
    x = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4], dtype=np.float32)
    axes = None
    name = "fftshift_5"
    input_dict = {"x": x, "axes": axes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array, shift along axis 0
    x = np.arange(24, dtype=np.float32).reshape((2, 3, 4))
    axes = 0
    name = "fftshift_6"
    input_dict = {"x": x, "axes": axes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D array, shift along axis 1
    x = np.arange(24, dtype=np.float32).reshape((2, 3, 4))
    axes = 1
    name = "fftshift_7"
    input_dict = {"x": x, "axes": axes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D array, shift along axis 2
    x = np.arange(24, dtype=np.float32).reshape((2, 3, 4))
    axes = 2
    name = "fftshift_8"
    input_dict = {"x": x, "axes": axes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D array, shift along multiple axes (0 and 1) - this is not directly possible as the input is an int and not a tuple, but let's provide it as 0 for example
    x = np.arange(24, dtype=np.float32).reshape((2, 3, 4))
    axes = 0
    name = "fftshift_9"
    input_dict = {"x": x, "axes": axes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D array, shift along axis 3
    x = np.arange(48, dtype=np.float32).reshape((2, 2, 3, 4))
    axes = 3
    name = "fftshift_10"
    input_dict = {"x": x, "axes": axes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.signal.fftshift_2"] = tf_signal_fftshift_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.signal.fftshift_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.signal.fftshift_2'.")

check_valid('tf.signal.fftshift', generated_inputs['tf.signal.fftshift_2'], lib="tf", suffix=2)

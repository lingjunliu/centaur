
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_signal_fftshift_inputs():
    list_of_inputs = []

    # Input 1: 1D tensor, no axes specified
    x = np.array([0., 1., 2., 3., 4., -5., -4., -3., -2., -1.], dtype=np.float32)
    x = tf.convert_to_tensor(x)
    axes = None
    name = None
    input_dict = {"x": x, "axes": axes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D tensor, shift along axis 0
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    x = tf.convert_to_tensor(x)
    axes = (0,)
    name = "shift_axis_0"
    input_dict = {"x": x, "axes": axes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D tensor, shift along axis 1
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    x = tf.convert_to_tensor(x)
    axes = (1,)
    name = "shift_axis_1"
    input_dict = {"x": x, "axes": axes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D tensor, shift along both axes
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    x = tf.convert_to_tensor(x)
    axes = (0, 1)
    name = "shift_both_axes"
    input_dict = {"x": x, "axes": axes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D tensor, shift along axis 0
    x = np.random.rand(3, 4, 5).astype(np.float32)
    x = tf.convert_to_tensor(x)
    axes = (0,)
    name = "shift_3d_axis_0"
    input_dict = {"x": x, "axes": axes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D tensor, shift along axis 1
    x = np.random.rand(3, 4, 5).astype(np.float32)
    x = tf.convert_to_tensor(x)
    axes = (1,)
    name = "shift_3d_axis_1"
    input_dict = {"x": x, "axes": axes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D tensor, shift along axis 2
    x = np.random.rand(3, 4, 5).astype(np.float32)
    x = tf.convert_to_tensor(x)
    axes = (2,)
    name = "shift_3d_axis_2"
    input_dict = {"x": x, "axes": axes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D tensor, shift along multiple axes
    x = np.random.rand(3, 4, 5).astype(np.float32)
    x = tf.convert_to_tensor(x)
    axes = (0, 2)
    name = "shift_3d_multiple_axes"
    input_dict = {"x": x, "axes": axes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D tensor with complex values
    x = np.array([1+1j, 2+2j, 3+3j, 4+4j, 5+5j], dtype=np.complex64)
    x = tf.convert_to_tensor(x)
    axes = None
    name = "complex_1d"
    input_dict = {"x": x, "axes": axes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D Tensor with negative values and axes specified
    x = np.array([[-1, -2, -3], [-4, -5, -6]], dtype=np.float32)
    x = tf.convert_to_tensor(x)
    axes = (0, 1)
    name = "negative_values_2d"
    input_dict = {"x": x, "axes": axes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11: 1D tensor, no axes specified, int type
    x = np.array([0, 1, 2, 3, 4, -5, -4, -3, -2, -1], dtype=np.int32)
    x = tf.convert_to_tensor(x)
    axes = None
    name = None
    input_dict = {"x": x, "axes": axes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Empty tensor
    x = np.array([], dtype=np.float32)
    x = tf.convert_to_tensor(x)
    axes = None
    name = "empty_tensor"
    input_dict = {"x": x, "axes": axes, "name": name}
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


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_signal_fftshift_inputs():
    list_of_inputs = []

    # Input 1
    x = np.array([0., 1., 2., 3., 4., -5., -4., -3., -2., -1.])
    axes = 0
    name = "fftshift_example_1"
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float32), "axes": int(axes), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    x = np.array([[1, 2], [3, 4]])
    axes = 0
    name = "fftshift_example_2"
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.int32), "axes": int(axes), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    x = np.array([[1, 2, 3], [4, 5, 6], [7,8,9]])
    axes = 1
    name = "fftshift_example_3"
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.int32), "axes": int(axes), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    axes = 0
    name = "fftshift_example_4"
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.int32), "axes": int(axes), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    x = np.array([1, 2, 3, 4])
    axes = -1
    name = "fftshift_example_5"
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.int32), "axes": int(axes), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    x = np.array([[-1, -2], [-3, -4]])
    axes = 0
    name = "fftshift_example_6"
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.int32), "axes": int(axes), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    x = np.array([1, 2, 3, 4, 5])
    axes = 0
    name = "fftshift_example_7"
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.int32), "axes": int(axes), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    x = np.array([[1, 2, 3], [4, 5, 6]])
    axes = -2
    name = "fftshift_example_8"
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.int32), "axes": int(axes), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    axes = 2
    name = "fftshift_example_9"
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.float32), "axes": int(axes), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    x = np.array([1, 2, 3, 4], dtype=np.complex64)
    axes = 0
    name = "fftshift_example_10"
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.complex64), "axes": int(axes), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11. integer axes values
    x = np.array([[1, 2], [3, 4]])
    axes = np.int32(0)
    name = "fftshift_example_11"
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.int32), "axes": int(axes), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12.  axes values
    x = np.array([1, 2, 3, 4])
    axes = np.int64(-1)
    name = "fftshift_example_12"
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.int32), "axes": int(axes), "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 13. float axes values
    x = np.array([1, 2, 3, 4])
    axes = 0.0
    name = "fftshift_example_13"
    input_dict = {"x": tf.convert_to_tensor(x, dtype=tf.int32), "axes": int(axes), "name": name}
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

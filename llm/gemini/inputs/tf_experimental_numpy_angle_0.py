
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_angle_inputs():
    list_of_inputs = []

    # Input 1
    z = tf.constant([1+1j, 1-1j, -1+1j, -1-1j], dtype=tf.complex128).numpy()
    deg = False
    input_dict = {"z": z, "deg": deg}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    z = tf.constant([1+1j, 1-1j, -1+1j, -1-1j], dtype=tf.complex128).numpy()
    deg = True
    input_dict = {"z": z, "deg": deg}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    z = tf.constant([0+0j, 1+0j, 0+1j, -1+0j, 0-1j], dtype=tf.complex128).numpy()
    deg = False
    input_dict = {"z": z, "deg": deg}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    z = tf.constant([[1+1j, 1-1j], [-1+1j, -1-1j]], dtype=tf.complex128).numpy()
    deg = True
    input_dict = {"z": z, "deg": deg}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    z = tf.constant([1, 2, 3], dtype=tf.float64).numpy()
    deg = False
    z = z.astype(np.complex128)
    input_dict = {"z": z, "deg": deg}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    z = tf.constant([-1, -2, -3], dtype=tf.float64).numpy()
    deg = True
    z = z.astype(np.complex128)
    input_dict = {"z": z, "deg": deg}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    z = tf.constant([0, 0, 0], dtype=tf.float64).numpy()
    deg = False
    z = z.astype(np.complex128)
    input_dict = {"z": z, "deg": deg}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    z = tf.constant([[1, 0], [0, 1]], dtype=tf.float64).numpy()
    deg = True
    z = z.astype(np.complex128)
    input_dict = {"z": z, "deg": deg}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    z = tf.constant([[-1, 0], [0, -1]], dtype=tf.float64).numpy()
    deg = False
    z = z.astype(np.complex128)
    input_dict = {"z": z, "deg": deg}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    z = tf.constant([1+0j, 0+1j, -1+0j, 0-1j], dtype=tf.complex128).numpy()
    deg = True
    input_dict = {"z": z, "deg": deg}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 11
    z = tf.constant(1j).numpy()
    deg = False
    input_dict = {"z": z, "deg": deg}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.angle"] = tf_experimental_numpy_angle_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.angle' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.angle'.")

check_valid('tf.experimental.numpy.angle', generated_inputs['tf.experimental.numpy.angle'], lib="tf", suffix=0)

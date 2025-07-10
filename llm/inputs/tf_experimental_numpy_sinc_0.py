
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_experimental_numpy_sinc_inputs():
    list_of_inputs = []

    # Input 1: Scalar input
    x = tf.constant(0.0, dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D tensor
    x = tf.constant(np.array([-np.pi, 0, np.pi]), dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D tensor
    x = tf.constant(np.array([[-np.pi/2, 0, np.pi/2], [np.pi, 2*np.pi, 3*np.pi]]), dtype=tf.float64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D tensor
    x = tf.constant(np.random.rand(2, 3, 4), dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative values
    x = tf.constant(np.array([-1.0, -0.5, 0.0, 0.5, 1.0]), dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Zeros
    x = tf.constant(np.zeros((5, 5)), dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Ones
    x = tf.constant(np.ones((3, 3)), dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Large values
    x = tf.constant(np.array([100.0, 200.0, 300.0]), dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 9: Complex numbers
    x = tf.constant(np.array([1+1j, 2+2j, 3+3j]), dtype=tf.complex64)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Another scalar
    x = tf.constant(np.pi, dtype=tf.float32)
    input_dict = {"x": x}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.experimental.numpy.sinc"] = tf_experimental_numpy_sinc_inputs()

for i in range(len(generated_inputs["tf.experimental.numpy.sinc"])):
    generated_inputs["tf.experimental.numpy.sinc"][i]['x'] = generated_inputs["tf.experimental.numpy.sinc"][i]['x'].numpy()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.experimental.numpy.sinc' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.numpy.sinc'.")

check_valid('tf.experimental.numpy.sinc', generated_inputs['tf.experimental.numpy.sinc'], lib="tf", suffix=0)

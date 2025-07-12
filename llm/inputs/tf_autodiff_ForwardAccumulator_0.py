
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_autodiff_forwardaccumulator_inputs():
    list_of_inputs = []

    # Input 1
    primals = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    tangents = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    input_dict = {"primals": primals, "tangents": tangents}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    primals = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    tangents = np.array([0.5, -0.2, 1.0], dtype=np.float32)
    input_dict = {"primals": primals, "tangents": tangents}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    primals = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    tangents = np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]], dtype=np.float32)
    input_dict = {"primals": primals, "tangents": tangents}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    primals = np.array([1.0], dtype=np.float32)
    tangents = np.array([-1.0], dtype=np.float32)
    input_dict = {"primals": primals, "tangents": tangents}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    primals = np.array([1, 2, 3], dtype=np.int32).astype(np.float32)
    tangents = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    input_dict = {"primals": primals, "tangents": tangents}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    primals = np.array(5.0, dtype=np.float32)
    tangents = np.array(2.0, dtype=np.float32)
    input_dict = {"primals": primals, "tangents": tangents}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    primals = np.zeros((2, 3), dtype=np.float32)
    tangents = np.ones((2, 3), dtype=np.float32)
    input_dict = {"primals": primals, "tangents": tangents}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    primals = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    tangents = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {"primals": primals, "tangents": tangents}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    primals = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    tangents = np.array([[-0.5, 0.5], [0.5, -0.5]], dtype=np.float32)
    input_dict = {"primals": primals, "tangents": tangents}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    primals = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    tangents = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    input_dict = {"primals": primals, "tangents": tangents}
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.autodiff.ForwardAccumulator"] = tf_autodiff_forwardaccumulator_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.autodiff.ForwardAccumulator' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.autodiff.ForwardAccumulator'.")

check_valid('tf.autodiff.ForwardAccumulator', generated_inputs['tf.autodiff.ForwardAccumulator'], lib="tf", suffix=0)

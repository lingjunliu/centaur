
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_indexed_slices_inputs():
    list_of_inputs = []

    values1 = np.array([1.0, 2.0, 3.0])
    indices1 = np.array([0, 2])
    dense_shape1 = np.array([5])
    input_dict1 = {'values': values1, 'indices': indices1, 'dense_shape': dense_shape1}
    list_of_inputs.append(copy.deepcopy(input_dict1))

    values2 = np.array([[1, 2], [3, 4]])
    indices2 = np.array([1, 3])
    dense_shape2 = np.array([5, 2])
    input_dict2 = {'values': values2, 'indices': indices2, 'dense_shape': dense_shape2}
    list_of_inputs.append(copy.deepcopy(input_dict2))

    values3 = np.array([[-1, 0, 1], [2, -3, 4]])
    indices3 = np.array([0, 1, 2])
    dense_shape3 = np.array([4, 3])
    input_dict3 = {'values': values3, 'indices': indices3, 'dense_shape': dense_shape3}
    list_of_inputs.append(copy.deepcopy(input_dict3))

    values4 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    indices4 = np.array([0, 2])
    dense_shape4 = np.array([4, 2, 2])
    input_dict4 = {'values': values4, 'indices': indices4, 'dense_shape': dense_shape4}
    list_of_inputs.append(copy.deepcopy(input_dict4))

    values5 = np.array([10, 20, 30, 40])
    indices5 = np.array([1, 3])
    dense_shape5 = np.array([6])
    input_dict5 = {'values': values5, 'indices': indices5, 'dense_shape': dense_shape5}
    list_of_inputs.append(copy.deepcopy(input_dict5))

    values6 = np.array([[1,2],[3,4],[5,6]])
    indices6 = np.array([0,1,2])
    dense_shape6 = np.array([3,2])
    input_dict6 = {'values': values6, 'indices': indices6, 'dense_shape': dense_shape6}
    list_of_inputs.append(copy.deepcopy(input_dict6))

    values7 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    indices7 = np.array([0, 2, 4])
    dense_shape7 = np.array([6])
    input_dict7 = {'values': values7, 'indices': indices7, 'dense_shape': dense_shape7}
    list_of_inputs.append(copy.deepcopy(input_dict7))

    values8 = np.array([[-1.0], [2.0], [-3.0]])
    indices8 = np.array([0, 1, 2])
    dense_shape8 = np.array([3, 1])
    input_dict8 = {'values': values8, 'indices': indices8, 'dense_shape': dense_shape8}
    list_of_inputs.append(copy.deepcopy(input_dict8))

    values9 = np.array([1, 2, 3])
    indices9 = np.array([0])
    dense_shape9 = np.array([3])
    input_dict9 = {'values': values9, 'indices': indices9, 'dense_shape': dense_shape9}
    list_of_inputs.append(copy.deepcopy(input_dict9))

    values10 = np.array([[1, 2, 3], [4, 5, 6]])
    indices10 = np.array([0, 1])
    dense_shape10 = np.array([2, 3])
    input_dict10 = {'values': values10, 'indices': indices10, 'dense_shape': dense_shape10}
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["tf.IndexedSlices"] = tf_indexed_slices_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.IndexedSlices' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.IndexedSlices'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.IndexedSlices', generated_inputs['tf.IndexedSlices'], lib="tf", suffix=0)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_norm_inputs():
    list_of_inputs = []

    def create_input(tensor, ord, axis, keepdims, name):
        input_dict = {'tensor': tensor, 'ord': ord, 'axis': axis, 'keepdims': keepdims, 'name': name}
        for k, v in input_dict.items():
            if v is None:
                input_dict[k] = None
        return input_dict

    # Input 1: Vector norm, default ord
    list_of_inputs.append(copy.deepcopy(create_input(np.array([1.0, 2.0, 3.0], dtype=np.float32), 'euclidean', None, False, 'norm_vector_default')))

    # Input 2: Matrix norm, Frobenius
    list_of_inputs.append(copy.deepcopy(create_input(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), 'fro', None, False, 'norm_matrix_fro')))

    # Input 3: Vector norm, 1-norm
    list_of_inputs.append(copy.deepcopy(create_input(np.array([-1.0, 2.0, -3.0], dtype=np.float32), '1', None, False, 'norm_vector_1')))

    # Input 4: Matrix norm, 1-norm, axis specified
    list_of_inputs.append(copy.deepcopy(create_input(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), '1', [0, 1], False, 'norm_matrix_1_axis')))

    # Input 5: Vector norm, inf-norm
    list_of_inputs.append(copy.deepcopy(create_input(np.array([-1.0, 2.0, -3.0], dtype=np.float32), 'np.inf', None, False, 'norm_vector_inf')))

    # Input 6: Matrix norm, inf-norm, axis specified
    list_of_inputs.append(copy.deepcopy(create_input(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), 'np.inf', [0, 1], True, 'norm_matrix_inf_axis_keepdims')))

    # Input 7: Batch of vectors, 2-norm along axis 1
    list_of_inputs.append(copy.deepcopy(create_input(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32), '2', [1], False, 'norm_batch_vector_axis')))

    # Input 8: Batch of matrices, euclidean norm, axis specified
    list_of_inputs.append(copy.deepcopy(create_input(np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32), 'euclidean', (1, 2), False, 'norm_batch_matrix_axis')))

    # Input 9: Vector norm, p-norm (p=1.5)
    list_of_inputs.append(copy.deepcopy(create_input(np.array([1.0, 2.0, 3.0], dtype=np.float32), '1.5', None, False, 'norm_vector_p')))

    # Input 10: Batch of vectors, default norm along axis 0, keepdims=True
    list_of_inputs.append(copy.deepcopy(create_input(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), 'euclidean', [0], True, 'norm_batch_vector_axis_keepdims')))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.norm_2"] = tf_norm_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.norm_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.norm_2'.")

check_valid('tf.norm', generated_inputs['tf.norm_2'], lib="tf", suffix=2)

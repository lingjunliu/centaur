
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_linalg_matrix_transpose_inputs():
    list_of_inputs = []
    
    a = np.array([[1, 2, 3], [4, 5, 6]])
    name = 'matrix_transpose'
    conjugate = False
    input_dict = {"a": a, "name": name, "conjugate": conjugate}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    a = np.array([[1 + 1j, 2 + 2j, 3 + 3j], [4 + 4j, 5 + 5j, 6 + 6j]])
    name = 'complex_transpose'
    conjugate = True
    input_dict = {"a": a, "name": name, "conjugate": conjugate}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    name = 'batch_transpose'
    conjugate = False
    input_dict = {"a": a, "name": name, "conjugate": conjugate}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    a = np.random.randn(1, 2, 3, 4)
    name = 'multi_batch_transpose'
    conjugate = False
    input_dict = {"a": a, "name": name, "conjugate": conjugate}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    name = 'square_transpose'
    conjugate = False
    input_dict = {"a": a, "name": name, "conjugate": conjugate}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    a = np.array([[-1, -2, -3], [4, -5, 6]])
    name = 'negative_transpose'
    conjugate = False
    input_dict = {"a": a, "name": name, "conjugate": conjugate}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    a = np.array([[1.5, 2.7, 3.9], [4.1, 5.3, 6.5]])
    name = 'float_transpose'
    conjugate = False
    input_dict = {"a": a, "name": name, "conjugate": conjugate}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    a = np.array([[1], [2]])
    name = 'single_col_transpose'
    conjugate = False
    input_dict = {"a": a, "name": name, "conjugate": conjugate}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    a = np.random.randn(2, 3, 4) + 1j * np.random.randn(2, 3, 4)
    name = 'large_complex_transpose'
    conjugate = True
    input_dict = {"a": a, "name": name, "conjugate": conjugate}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    a = np.random.randn(2, 3, 4, 5, 6)
    name = 'five_d_transpose'
    conjugate = False
    input_dict = {"a": a, "name": name, "conjugate": conjugate}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.linalg.matrix_transpose"] = tf_linalg_matrix_transpose_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.linalg.matrix_transpose' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.matrix_transpose'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.linalg.matrix_transpose', generated_inputs['tf.linalg.matrix_transpose'], lib="tf", suffix=0)

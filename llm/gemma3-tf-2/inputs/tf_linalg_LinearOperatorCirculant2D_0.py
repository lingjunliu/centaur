
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def linear_operator_circulant2d_inputs():
    list_of_inputs = []

    spectrum1 = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]], dtype=np.complex64)
    input_dict1 = {
        'spectrum': spectrum1,
        'input_output_dtype': tf.complex64,
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'circulant_2d_1'
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    spectrum2 = np.array([[1. + 1j, 2. - 1j], [3. + 2j, 4. - 2j]], dtype=np.complex128)
    input_dict2 = {
        'spectrum': spectrum2,
        'input_output_dtype': tf.complex128,
        'is_non_singular': False,
        'is_self_adjoint': True,
        'is_positive_definite': False,
        'is_square': True,
        'name': 'circulant_2d_2'
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    spectrum3 = np.array([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]], dtype=np.float32)
    input_dict3 = {
        'spectrum': spectrum3,
        'input_output_dtype': tf.float32,
        'is_non_singular': None,
        'is_self_adjoint': True,
        'is_positive_definite': None,
        'is_square': True,
        'name': 'circulant_2d_3'
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    spectrum4 = np.array([[1., 0.], [0., 1.]], dtype=np.complex64)
    input_dict4 = {
        'spectrum': spectrum4,
        'input_output_dtype': tf.complex64,
        'is_non_singular': True,
        'is_self_adjoint': True,
        'is_positive_definite': True,
        'is_square': True,
        'name': 'circulant_2d_4'
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    return list_of_inputs

generated_inputs["tf.linalg.LinearOperatorCirculant2D"] = linear_operator_circulant2d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.linalg.LinearOperatorCirculant2D' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.LinearOperatorCirculant2D'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.linalg.LinearOperatorCirculant2D', generated_inputs['tf.linalg.LinearOperatorCirculant2D'], lib="tf", suffix=0)

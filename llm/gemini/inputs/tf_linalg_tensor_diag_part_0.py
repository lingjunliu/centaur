
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_linalg_tensor_diag_part_inputs():
    list_of_inputs = []

    list_of_inputs.append({
        'input': np.random.randn(3, 3).astype(np.float32),
        'name': 'diag_1'
    })

    list_of_inputs.append({
        'input': np.random.randint(-10, 10, size=(1, 1)).astype(np.int32),
        'name': 'diag_2'
    })

    list_of_inputs.append({
        'input': np.random.uniform(-5.0, 5.0, size=(5, 5)).astype(np.float64),
        'name': 'diag_3'
    })

    list_of_inputs.append({
        'input': np.random.randint(-5, 5, size=(2, 2, 2, 2)).astype(np.int64),
        'name': 'diag_4'
    })

    list_of_inputs.append({
        'input': np.random.randn(3, 4, 3, 4).astype(np.float32),
        'name': 'diag_5'
    })

    list_of_inputs.append({
        'input': np.random.randn(1, 5, 1, 5).astype(np.float32),
        'name': 'diag_6'
    })

    list_of_inputs.append({
        'input': np.random.randint(0, 100, size=(2, 2, 2, 2, 2, 2)).astype(np.int32),
        'name': 'diag_7'
    })

    list_of_inputs.append({
        'input': np.random.randn(2, 3, 1, 2, 3, 1).astype(np.float64),
        'name': 'diag_8'
    })

    real = np.random.randn(10, 10).astype(np.float32)
    imag = np.random.randn(10, 10).astype(np.float32)
    list_of_inputs.append({
        'input': (real + 1j * imag),
        'name': 'diag_9'
    })

    list_of_inputs.append({
        'input': np.random.randn(2, 3, 2, 3).astype(np.float16),
        'name': 'diag_10'
    })

    return list_of_inputs

generated_inputs["tf.linalg.tensor_diag_part"] = tf_linalg_tensor_diag_part_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.linalg.tensor_diag_part' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.linalg.tensor_diag_part'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.linalg.tensor_diag_part', generated_inputs['tf.linalg.tensor_diag_part'], lib="tf", suffix=0)

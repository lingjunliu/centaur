
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_signal_ifft2d_inputs():
    list_of_inputs = []

    # Input 1: Basic complex64 input
    input_tensor = tf.complex(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32))
    input_dict = {"input": input_tensor.numpy(), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Basic complex128 input
    input_tensor = tf.complex(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64), np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64))
    input_dict = {"input": input_tensor.numpy(), "name": "test_ifft2d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D complex64 input
    input_tensor = tf.complex(np.random.rand(2, 3, 4).astype(np.float32), np.random.rand(2, 3, 4).astype(np.float32))
    input_dict = {"input": input_tensor.numpy(), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D complex128 input
    input_tensor = tf.complex(np.random.rand(2, 3, 4).astype(np.float64), np.random.rand(2, 3, 4).astype(np.float64))
    input_dict = {"input": input_tensor.numpy(), "name": "test_ifft2d_3d"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Complex64 input with negative values
    input_tensor = tf.complex(np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32), np.array([[5.0, -6.0], [-7.0, 8.0]], dtype=np.float32))
    input_dict = {"input": input_tensor.numpy(), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Complex128 input with negative values
    input_tensor = tf.complex(np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float64), np.array([[5.0, -6.0], [-7.0, 8.0]], dtype=np.float64))
    input_dict = {"input": input_tensor.numpy(), "name": "test_ifft2d_neg"}
    list_of_inputs.append(copy.deepcopy(input_dict))

     # Input 7: Complex64 with zeros
    input_tensor = tf.complex(np.array([[0.0, 2.0], [3.0, 0.0]], dtype=np.float32), np.array([[5.0, 0.0], [0.0, 8.0]], dtype=np.float32))
    input_dict = {"input": input_tensor.numpy(), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Complex128 with zeros
    input_tensor = tf.complex(np.array([[0.0, 2.0], [3.0, 0.0]], dtype=np.float64), np.array([[5.0, 0.0], [0.0, 8.0]], dtype=np.float64))
    input_dict = {"input": input_tensor.numpy(), "name": "test_ifft2d_zero"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Larger complex64 input
    input_tensor = tf.complex(np.random.rand(5, 5).astype(np.float32), np.random.rand(5, 5).astype(np.float32))
    input_dict = {"input": input_tensor.numpy(), "name": None}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Larger complex128 input
    input_tensor = tf.complex(np.random.rand(5, 5).astype(np.float64), np.random.rand(5, 5).astype(np.float64))
    input_dict = {"input": input_tensor.numpy(), "name": "test_ifft2d_large"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.signal.ifft2d"] = tf_signal_ifft2d_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.signal.ifft2d' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.signal.ifft2d'.")

check_valid('tf.signal.ifft2d', generated_inputs['tf.signal.ifft2d'], lib="tf", suffix=0)

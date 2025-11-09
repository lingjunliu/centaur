
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
np.random.seed(42)
tf.random.set_seed(42)

def tf_autodiff_ForwardAccumulator_inputs():
    list_of_inputs = []

    # Input 1: scalar float32
    primals = np.array(3.14, dtype=np.float32)
    tangents = np.array(1.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"primals": primals, "tangents": tangents}))

    # Input 2: 1D vector float32 with negatives
    primals = np.array([-2.5, 0.0, 3.5, 7.2], dtype=np.float32)
    tangents = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"primals": primals, "tangents": tangents}))

    # Input 3: 2D matrix float64
    primals = np.array([[1.0, -1.0, 2.0],
                        [3.5, 0.0, -4.2],
                        [5.1, 6.3, -7.7]], dtype=np.float64)
    tangents = np.array([[0.5, 0.5, -0.5],
                         [1.0, -1.0, 1.5],
                         [0.0, 2.0, -2.0]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"primals": primals, "tangents": tangents}))

    # Input 4: 3D tensor float32
    primals = np.random.uniform(-1.0, 1.0, size=(2, 3, 1)).astype(np.float32)
    tangents = np.random.uniform(-0.5, 0.5, size=(2, 3, 1)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"primals": primals, "tangents": tangents}))

    # Input 5: 4D tensor float16
    primals = (np.random.randn(2, 1, 3, 4)).astype(np.float16)
    tangents = (np.random.randn(2, 1, 3, 4)).astype(np.float16)
    list_of_inputs.append(copy.deepcopy({"primals": primals, "tangents": tangents}))

    # Input 6: large 1D vector float32
    primals = np.linspace(-10, 10, 100).astype(np.float32)
    tangents = np.ones_like(primals, dtype=np.float32) * 0.01
    list_of_inputs.append(copy.deepcopy({"primals": primals, "tangents": tangents}))

    # Input 7: complex64 vector
    primals = (np.array([1+2j, -3+0.5j, 0-1j], dtype=np.complex64))
    tangents = (np.array([0.1-0.2j, -0.3+0.4j, 0.5+0.6j], dtype=np.complex64))
    list_of_inputs.append(copy.deepcopy({"primals": primals, "tangents": tangents}))

    # Input 8: 2D zeros tensor float32, non-zero tangents
    primals = np.zeros((5, 5), dtype=np.float32)
    tangents = np.full((5, 5), 2.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"primals": primals, "tangents": tangents}))

    # Input 9: arange reshaped float64
    primals = np.arange(10, dtype=np.float64).reshape(2, 5)
    tangents = np.ones((2, 5), dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"primals": primals, "tangents": tangents}))

    # Input 10: 3D tensor float32 with negatives
    primals = np.array([[[-1.0, -2.0], [3.0, -4.0]],
                        [[5.0, -6.0], [7.0, -8.0]]], dtype=np.float32)
    tangents = np.random.uniform(-1.0, 1.0, size=(2, 2, 2)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"primals": primals, "tangents": tangents}))

    # Input 11: very small numbers float64
    primals = np.array([1e-12, -2e-12, 3e-12, -4e-12], dtype=np.float64)
    tangents = np.array([1e-6, -1e-6, 2e-6, -2e-6], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"primals": primals, "tangents": tangents}))

    # Input 12: 2D non-contiguous view float32
    base = np.random.randn(6, 6).astype(np.float32)
    primals = base[::2, ::2]
    tangents = np.random.randn(*primals.shape).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"primals": primals, "tangents": tangents}))

    return list_of_inputs

generated_inputs["tf.autodiff.ForwardAccumulator"] = tf_autodiff_ForwardAccumulator_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.autodiff.ForwardAccumulator' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.autodiff.ForwardAccumulator'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.autodiff.ForwardAccumulator', generated_inputs['tf.autodiff.ForwardAccumulator'], lib="tf", suffix=0)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)
np.random.seed(42)

def tf_raw_ops_DiagPart_inputs():
    list_of_inputs = []

    # Input 1: 2D int32 square matrix
    input_arr = np.array(
        [[1, 0, 0, 0],
         [0, 2, 0, 0],
         [0, 0, 3, 0],
         [0, 0, 0, 4]], dtype=np.int32
    )
    input_dict = {"input": input_arr, "name": "case_2d_int32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float64 matrix with negatives
    input_arr = np.array(
        [[3.0, -1.2, 0.0],
         [2.3, -5.5, 7.8],
         [9.1, 4.2, -8.3]], dtype=np.float64
    )
    input_dict = {"input": input_arr, "name": "case_2d_float64_neg"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 4D float32 tensor [2,3,2,3]
    input_arr = np.arange(2*3*2*3, dtype=np.float32).reshape(2, 3, 2, 3)
    input_dict = {"input": input_arr, "name": "case_4d_float32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 4D int64 tensor [1,5,1,5] with random integers (including negatives)
    input_arr = np.random.randint(-50, 50, size=(1, 5, 1, 5), dtype=np.int64)
    input_dict = {"input": input_arr, "name": "case_4d_int64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 6D complex64 tensor [2,1,3,2,1,3]
    real = np.random.randn(2, 1, 3, 2, 1, 3).astype(np.float32)
    imag = np.random.randn(2, 1, 3, 2, 1, 3).astype(np.float32)
    input_arr = (real + 1j * imag).astype(np.complex64)
    input_dict = {"input": input_arr, "name": "case_6d_complex64"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D float16 1x1 tensor
    input_arr = np.array([[7.5]], dtype=np.float16)
    input_dict = {"input": input_arr, "name": "case_2d_float16_1x1"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D complex128 tensor [3,3,3,3]
    real = np.random.randn(3, 3, 3, 3)
    imag = np.random.randn(3, 3, 3, 3)
    input_arr = (real + 1j * imag).astype(np.complex128)
    input_dict = {"input": input_arr, "name": "case_4d_complex128"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D float32 empty square matrix [0,0]
    input_arr = np.empty((0, 0), dtype=np.float32)
    input_dict = {"input": input_arr, "name": "case_2d_empty_float32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 8D float32 tensor [2,2,1,3,2,2,1,3]
    input_arr = np.random.randn(2, 2, 1, 3, 2, 2, 1, 3).astype(np.float32)
    input_dict = {"input": input_arr, "name": "case_8d_float32"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 6D int32 tensor [4,2,1,4,2,1] with negatives
    input_arr = (np.arange(4*2*1*4*2*1, dtype=np.int32) - 20).reshape(4, 2, 1, 4, 2, 1)
    input_dict = {"input": input_arr, "name": "case_6d_int32_neg"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 2D int64 square matrix with mixed values
    input_arr = np.array(
        [[-10, 2, 3],
         [4, 0, -6],
         [7, 8, 15]], dtype=np.int64
    )
    input_dict = {"input": input_arr, "name": "case_2d_int64_mixed"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: 4D float64 tensor [1,1,1,1]
    input_arr = np.array([[[[42.0]]]], dtype=np.float64)
    input_dict = {"input": input_arr, "name": "case_4d_float64_singleton"}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.DiagPart"] = tf_raw_ops_DiagPart_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.DiagPart' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DiagPart'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.DiagPart', generated_inputs['tf.raw_ops.DiagPart'], lib="tf", suffix=0)

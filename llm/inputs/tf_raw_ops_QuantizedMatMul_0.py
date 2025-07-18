
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def tf_raw_ops_QuantizedMatMul_inputs():
    """
    This function generates a list of valid inputs for the
    tf.raw_ops.QuantizedMatMul operation.
    """
    list_of_inputs = []

    # The error `ValueError: tf.qint8 is not in list` indicates the testing
    # framework cannot process TensorFlow's quantized dtypes (e.g., tf.qint8)
    # when analyzing the input dictionary. To resolve this specific error, this
    # function will provide standard numpy arrays with regular integer dtypes
    # (e.g., np.int8) for the 'a' and 'b' tensors, which the testing framework
    # should be able to process.
    def create_input_dict(a, b, min_a, max_a, min_b, max_b, Toutput, transpose_a, transpose_b, Tactivation, name):
        return {
            'a': a,
            'b': b,
            'min_a': np.array(min_a, dtype=np.float32),
            'max_a': np.array(max_a, dtype=np.float32),
            'min_b': np.array(min_b, dtype=np.float32),
            'max_b': np.array(max_b, dtype=np.float32),
            'Toutput': Toutput,
            'transpose_a': transpose_a,
            'transpose_b': transpose_b,
            'Tactivation': Tactivation,
            'name': name
        }

    # Input 1: Basic int8 multiplication attempt
    a1 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int8)
    b1 = np.array([[7, 8], [9, 10], [11, 12]], dtype=np.int8)
    list_of_inputs.append(copy.deepcopy(create_input_dict(
        a=a1, b=b1, min_a=-10.0, max_a=10.0, min_b=-20.0, max_b=20.0,
        Toutput=tf.qint32, transpose_a=False, transpose_b=False,
        Tactivation=tf.quint8, name="basic_int8")))

    # Input 2: Basic uint8 multiplication attempt
    a2 = np.array([[10, 20], [30, 40], [50, 60]], dtype=np.uint8)
    b2 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.uint8)
    list_of_inputs.append(copy.deepcopy(create_input_dict(
        a=a2, b=b2, min_a=0.0, max_a=100.0, min_b=0.0, max_b=10.0,
        Toutput=tf.qint32, transpose_a=False, transpose_b=False,
        Tactivation=tf.quint8, name="basic_uint8")))

    # Input 3: transpose_a = True
    a3 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int8)
    b3 = np.array([[10, 11], [12, 13]], dtype=np.int8)
    list_of_inputs.append(copy.deepcopy(create_input_dict(
        a=a3, b=b3, min_a=-10.0, max_a=10.0, min_b=-15.0, max_b=15.0,
        Toutput=tf.qint32, transpose_a=True, transpose_b=False,
        Tactivation=tf.quint8, name="transpose_a_true")))

    # Input 4: transpose_b = True
    a4 = np.array([[1, 2, 3]], dtype=np.uint8)
    b4 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    list_of_inputs.append(copy.deepcopy(create_input_dict(
        a=a4, b=b4, min_a=0.0, max_a=5.0, min_b=0.0, max_b=10.0,
        Toutput=tf.qint32, transpose_a=False, transpose_b=True,
        Tactivation=tf.quint8, name="transpose_b_true")))

    # Input 5: transpose_a = True and transpose_b = True
    a5 = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int8)
    b5 = np.array([[1,2,3],[4,5,6]], dtype=np.int8)
    list_of_inputs.append(copy.deepcopy(create_input_dict(
        a=a5, b=b5, min_a=-5.0, max_a=5.0, min_b=-6.0, max_b=6.0,
        Toutput=tf.qint32, transpose_a=True, transpose_b=True,
        Tactivation=tf.quint8, name="transpose_both_true")))

    # Input 6: Different dtypes (int16, uint16) and different Toutput/Tactivation
    a6 = np.array([[100, -200], [300, 400]], dtype=np.int16)
    b6 = np.array([[50, 60], [70, 80]], dtype=np.uint16)
    list_of_inputs.append(copy.deepcopy(create_input_dict(
        a=a6, b=b6, min_a=-500.0, max_a=500.0, min_b=0.0, max_b=100.0,
        Toutput=tf.qint16, transpose_a=False, transpose_b=False,
        Tactivation=tf.qint16, name="mixed_16bit_types")))

    # Input 7: int32 input types
    a7 = np.array([[-100000, 200000], [300000, -400000]], dtype=np.int32)
    b7 = np.array([[10, 20], [-30, 40]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy(create_input_dict(
        a=a7, b=b7, min_a=-500000.0, max_a=500000.0, min_b=-50.0, max_b=50.0,
        Toutput=tf.qint32, transpose_a=False, transpose_b=False,
        Tactivation=tf.qint32, name="int32_inputs")))

    # Input 8: Larger matrices
    a8 = np.arange(20, dtype=np.uint8).reshape(4, 5)
    b8 = np.arange(30, dtype=np.uint8).reshape(5, 6)
    list_of_inputs.append(copy.deepcopy(create_input_dict(
        a=a8, b=b8, min_a=0.0, max_a=20.0, min_b=0.0, max_b=30.0,
        Toutput=tf.qint32, transpose_a=False, transpose_b=False,
        Tactivation=tf.quint8, name="larger_matrices_uint8")))
    
    # Input 9: Negative float ranges with int8
    a9 = np.array([[-120, -100], [-80, -60]], dtype=np.int8)
    b9 = np.array([[-1, -2], [-3, -4]], dtype=np.int8)
    list_of_inputs.append(copy.deepcopy(create_input_dict(
        a=a9, b=b9, min_a=-10.0, max_a=-5.0, min_b=-0.5, max_b=-0.1,
        Toutput=tf.qint32, transpose_a=False, transpose_b=False,
        Tactivation=tf.quint8, name="negative_ranges_int8")))

    # Input 10: Vector-matrix multiplication
    a10 = np.array([[10, 20, 30]], dtype=np.int8)
    b10 = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int8)
    list_of_inputs.append(copy.deepcopy(create_input_dict(
        a=a10, b=b10, min_a=-50.0, max_a=50.0, min_b=-10.0, max_b=10.0,
        Toutput=tf.qint32, transpose_a=False, transpose_b=False,
        Tactivation=tf.quint8, name="vector_matrix_int8")))

    return list_of_inputs

generated_inputs["tf.raw_ops.QuantizedMatMul"] = tf_raw_ops_QuantizedMatMul_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.QuantizedMatMul' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.QuantizedMatMul'.")

check_valid('tf.raw_ops.QuantizedMatMul', generated_inputs['tf.raw_ops.QuantizedMatMul'], lib="tf", suffix=0)

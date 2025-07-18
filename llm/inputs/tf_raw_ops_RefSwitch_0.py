
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

# Assume generated_inputs is pre-initialized
# generated_inputs = {}

def tf_raw_ops_refswitch_inputs():
    """
    Generates a list of valid inputs for the tf.raw_ops.RefSwitch operation.
    Note: This op is designed for TensorFlow's graph mode and will raise a
    RuntimeError in eager execution, as it requires a 'RefTensor' which is
    a graph mode concept. The provided inputs are structurally valid for the
    op's signature but are expected to fail in an eager execution environment.
    """
    list_of_inputs = []

    # Input 1: pred=True, 1D float32 data
    input_dict_1 = {
        'data': np.array([1.0, 2.5, -3.0], dtype=np.float32),
        'pred': np.array(True, dtype=np.bool_),
        'name': 'case1_pred_true_float32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2: pred=False, 2D int32 data
    input_dict_2 = {
        'data': np.array([[10, -20], [30, 40]], dtype=np.int32),
        'pred': np.array(False, dtype=np.bool_),
        'name': 'case2_pred_false_int32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3: pred=True, 3D float64 data
    input_dict_3 = {
        'data': np.random.randn(2, 3, 1).astype(np.float64),
        'pred': np.array(True, dtype=np.bool_),
        'name': 'case3_pred_true_float64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4: pred=False, scalar int64 data
    input_dict_4 = {
        'data': np.array(9876543210, dtype=np.int64),
        'pred': np.array(False, dtype=np.bool_),
        'name': 'case4_pred_false_scalar_int64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5: pred=True, scalar float16 data
    input_dict_5 = {
        'data': np.array(3.14, dtype=np.float16),
        'pred': np.array(True, dtype=np.bool_),
        'name': 'case5_pred_true_scalar_float16'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6: pred=False, complex64 data
    input_dict_6 = {
        'data': np.array([1+2j, 3+4j], dtype=np.complex64),
        'pred': np.array(False, dtype=np.bool_),
        'name': 'case6_pred_false_complex64'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7: pred=True, complex128 data
    input_dict_7 = {
        'data': np.array([[10+20j, -5-5j]], dtype=np.complex128),
        'pred': np.array(True, dtype=np.bool_),
        'name': 'case7_pred_true_complex128'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8: pred=False, empty tensor with shape (0,)
    input_dict_8 = {
        'data': np.array([], dtype=np.float32),
        'pred': np.array(False, dtype=np.bool_),
        'name': 'case8_pred_false_empty'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))
    
    # Input 9: pred=True, int8 data
    input_dict_9 = {
        'data': np.zeros((4, 1), dtype=np.int8),
        'pred': np.array(True, dtype=np.bool_),
        'name': 'case9_pred_true_int8'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10: pred=False, uint32 data
    input_dict_10 = {
        'data': np.arange(10, dtype=np.uint32) * 1000,
        'pred': np.array(False, dtype=np.bool_),
        'name': 'case10_pred_false_uint32'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    # Input 11: pred=True, data is all zeros
    input_dict_11 = {
        'data': np.zeros((2, 5), dtype=np.float32),
        'pred': np.array(True, dtype=np.bool_),
        'name': 'case11_pred_true_zeros'
    }
    list_of_inputs.append(copy.deepcopy(input_dict_11))

    return list_of_inputs

generated_inputs["tf.raw_ops.RefSwitch"] = tf_raw_ops_refswitch_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.RefSwitch' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.RefSwitch'.")

check_valid('tf.raw_ops.RefSwitch', generated_inputs['tf.raw_ops.RefSwitch'], lib="tf", suffix=0)

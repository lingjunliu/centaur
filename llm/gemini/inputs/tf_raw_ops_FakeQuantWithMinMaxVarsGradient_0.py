
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import tensorflow as tf

def tf_raw_ops_FakeQuantWithMinMaxVarsGradient_inputs():
    list_of_inputs = []
    
    # Input 1
    input_dict_1 = {
        'num_bits': 8,
        'narrow_range': False,
        'name': "quant_grad_1",
        'gradients': np.array([1.0, 2.0], dtype=np.float32),
        'inputs': np.array([1.5, 2.5], dtype=np.float32),
        'min': np.array(0.0, dtype=np.float32),
        'max': np.array(6.0, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_1))

    # Input 2
    input_dict_2 = {
        'num_bits': 4,
        'narrow_range': True,
        'name': "quant_grad_2",
        'gradients': np.array([[1.0, -1.0], [0.5, -0.5]], dtype=np.float32),
        'inputs': np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        'min': np.array(-1.0, dtype=np.float32),
        'max': np.array(1.0, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_2))

    # Input 3
    input_dict_3 = {
        'num_bits': 2,
        'narrow_range': False,
        'name': "quant_grad_3",
        'gradients': np.array([[[1.0]]], dtype=np.float32),
        'inputs': np.array([[[0.5]]], dtype=np.float32),
        'min': np.array(-2.0, dtype=np.float32),
        'max': np.array(2.0, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_3))

    # Input 4
    input_dict_4 = {
        'num_bits': 8,
        'narrow_range': False,
        'name': "quant_grad_4",
        'gradients': np.zeros((2, 3, 4), dtype=np.float32),
        'inputs': np.random.uniform(-5.0, 5.0, size=(2, 3, 4)).astype(np.float32),
        'min': np.array(-5.0, dtype=np.float32),
        'max': np.array(5.0, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_4))

    # Input 5
    input_dict_5 = {
        'num_bits': 7,
        'narrow_range': True,
        'name': "quant_grad_5",
        'gradients': np.ones((5,), dtype=np.float32),
        'inputs': np.array([-10.0, -5.0, 0.0, 5.0, 10.0], dtype=np.float32),
        'min': np.array(-3.0, dtype=np.float32),
        'max': np.array(3.0, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_5))

    # Input 6
    input_dict_6 = {
        'num_bits': 6,
        'narrow_range': False,
        'name': "quant_grad_6",
        'gradients': np.random.normal(size=(10,)).astype(np.float32),
        'inputs': np.random.normal(size=(10,)).astype(np.float32),
        'min': np.array(-1.5, dtype=np.float32),
        'max': np.array(1.5, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_6))

    # Input 7
    input_dict_7 = {
        'num_bits': 3,
        'narrow_range': True,
        'name': "quant_grad_7",
        'gradients': np.array([0.0], dtype=np.float32),
        'inputs': np.array([0.0], dtype=np.float32),
        'min': np.array(-0.5, dtype=np.float32),
        'max': np.array(0.5, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_7))

    # Input 8
    input_dict_8 = {
        'num_bits': 5,
        'narrow_range': False,
        'name': "quant_grad_8",
        'gradients': np.array([10.0, 20.0, 30.0], dtype=np.float32),
        'inputs': np.array([1.0, 2.0, 3.0], dtype=np.float32),
        'min': np.array(1.0, dtype=np.float32),
        'max': np.array(3.0, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_8))

    # Input 9
    input_dict_9 = {
        'num_bits': 8,
        'narrow_range': True,
        'name': "quant_grad_9",
        'gradients': np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        'inputs': np.array([[1.1, 1.2, 1.3], [1.4, 1.5, 1.6]], dtype=np.float32),
        'min': np.array(1.0, dtype=np.float32),
        'max': np.array(2.0, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_9))

    # Input 10
    input_dict_10 = {
        'num_bits': 8,
        'narrow_range': False,
        'name': "quant_grad_10",
        'gradients': np.random.uniform(-1.0, 1.0, size=(2, 2, 2, 2)).astype(np.float32),
        'inputs': np.random.uniform(-1.0, 1.0, size=(2, 2, 2, 2)).astype(np.float32),
        'min': np.array(-1.0, dtype=np.float32),
        'max': np.array(1.0, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict_10))

    return list_of_inputs

generated_inputs["tf.raw_ops.FakeQuantWithMinMaxVarsGradient"] = tf_raw_ops_FakeQuantWithMinMaxVarsGradient_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.FakeQuantWithMinMaxVarsGradient' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.FakeQuantWithMinMaxVarsGradient'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.FakeQuantWithMinMaxVarsGradient', generated_inputs['tf.raw_ops.FakeQuantWithMinMaxVarsGradient'], lib="tf", suffix=0)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import tensorflow as tf
import copy

def tf_raw_ops_fake_quant_with_min_max_vars_per_channel_inputs():
    list_of_inputs = []

    rs = np.random.RandomState(123)

    inputs = np.array([-1.2, 0.0, 0.7], dtype=np.float32)
    min_v = np.array([-1.0, 0.0, -0.5], dtype=np.float32)
    max_v = np.array([1.0, 2.0, 0.75], dtype=np.float32)
    input_dict = {"num_bits": 8, "narrow_range": False, "name": "fqpc_1", "inputs": inputs, "min": min_v, "max": max_v}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inputs = np.array([[-2.0, -1.0, 0.0, 1.0],
                       [2.0, 3.0, -3.0, 4.0]], dtype=np.float32)
    min_v = np.array([-1.0, 0.1, -0.5, 2.0], dtype=np.float32)
    max_v = np.array([0.0, 2.0, 1.0, 3.0], dtype=np.float32)
    input_dict = {"num_bits": 2, "narrow_range": True, "name": "fqpc_2", "inputs": inputs, "min": min_v, "max": max_v}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inputs = np.arange(12, dtype=np.float32).reshape(1, 2, 2, 3) / 3.0 - 2.0
    min_v = np.array([-1.0, 0.2, 0.0], dtype=np.float32)
    max_v = np.array([0.9, 2.0, 3.5], dtype=np.float32)
    input_dict = {"num_bits": 6, "narrow_range": False, "name": "fqpc_3", "inputs": inputs, "min": min_v, "max": max_v}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inputs = rs.randn(3, 4, 4, 1).astype(np.float32)
    min_v = np.array([-2.0], dtype=np.float32)
    max_v = np.array([-0.5], dtype=np.float32)
    input_dict = {"num_bits": 16, "narrow_range": False, "name": "fqpc_4", "inputs": inputs, "min": min_v, "max": max_v}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inputs = np.array([[10.0, -10.0],
                       [0.5, -0.25],
                       [2.0, -3.0],
                       [-1.0, 1.0]], dtype=np.float32)
    min_v = np.array([-1.0, -2.0], dtype=np.float32)
    max_v = np.array([0.0, 0.0], dtype=np.float32)
    input_dict = {"num_bits": 12, "narrow_range": True, "name": "fqpc_5", "inputs": inputs, "min": min_v, "max": max_v}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inputs = np.array([0.75], dtype=np.float32)
    min_v = np.array([-0.1], dtype=np.float32)
    max_v = np.array([0.25], dtype=np.float32)
    input_dict = {"num_bits": 7, "narrow_range": False, "name": "fqpc_6", "inputs": inputs, "min": min_v, "max": max_v}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inputs = rs.randn(2, 3, 4, 5).astype(np.float32) * 2.0
    min_v = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    max_v = np.array([0.0, 1.0, 2.0, 2.5, 3.0], dtype=np.float32)
    input_dict = {"num_bits": 9, "narrow_range": False, "name": "fqpc_7", "inputs": inputs, "min": min_v, "max": max_v}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inputs = (np.arange(35, dtype=np.float32).reshape(5, 7) - 17.0) / 5.0
    min_v = np.array([-1.5, -1.0, -0.5, 0.0, 0.1, 0.2, 0.3], dtype=np.float32)
    max_v = np.array([0.0, 0.5, 0.6, 1.0, 1.5, 2.0, 2.5], dtype=np.float32)
    input_dict = {"num_bits": 3, "narrow_range": True, "name": "fqpc_8", "inputs": inputs, "min": min_v, "max": max_v}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inputs = np.linspace(-2.0, 2.0, 8, dtype=np.float32)
    min_v = np.array([-1.0] * 8, dtype=np.float32)
    max_v = np.array([1.0] * 8, dtype=np.float32)
    input_dict = {"num_bits": 10, "narrow_range": False, "name": "fqpc_9", "inputs": inputs, "min": min_v, "max": max_v}
    list_of_inputs.append(copy.deepcopy(input_dict))

    inputs = rs.uniform(-1.0, 1.0, size=(2, 2, 2, 4)).astype(np.float32)
    min_v = np.array([-0.2, -0.1, 0.0, 0.3], dtype=np.float32)
    max_v = np.array([0.2, 0.1, 1.0, 0.9], dtype=np.float32)
    input_dict = {"num_bits": 15, "narrow_range": True, "name": "fqpc_10", "inputs": inputs, "min": min_v, "max": max_v}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.FakeQuantWithMinMaxVarsPerChannel"] = tf_raw_ops_fake_quant_with_min_max_vars_per_channel_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.FakeQuantWithMinMaxVarsPerChannel' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.FakeQuantWithMinMaxVarsPerChannel'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.FakeQuantWithMinMaxVarsPerChannel', generated_inputs['tf.raw_ops.FakeQuantWithMinMaxVarsPerChannel'], lib="tf", suffix=0)

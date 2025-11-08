
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_Elu_inputs():
    list_of_inputs = []

    features = np.array(1.0, dtype=np.float32)
    input_dict = {"name": "elu_case_1", "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array(0.0, dtype=np.float64)
    input_dict = {"name": "elu_case_2", "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array(-1000.0, dtype=np.float32)
    input_dict = {"name": "elu_case_3", "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    input_dict = {"name": "elu_case_4", "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([[-3.5, -0.1, 0.0], [0.1, 2.3, -7.8]], dtype=np.float16)
    input_dict = {"name": "elu_case_5", "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.linspace(-3, 3, num=12).astype(np.float64).reshape(2, 2, 3)
    input_dict = {"name": "elu_case_6", "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([], dtype=np.float32)
    input_dict = {"name": "elu_case_7", "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([np.inf, -np.inf, np.nan, -0.0, 0.0, 3.14], dtype=np.float32)
    input_dict = {"name": "elu_case_8", "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([[-1.5, 0.0, 1.5], [2.5, -2.5, 0.5]], dtype=np.float32).reshape(1, 2, 1, 3)
    input_dict = {"name": "elu_case_9", "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.array([-20.0, -5.0, 5.0, 20.0], dtype=np.float64)
    input_dict = {"name": "elu_case_10", "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = np.arange(-6, 6, dtype=np.float32).reshape(3, 4)[:, ::2]
    input_dict = {"name": "elu_case_11", "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    features = (np.arange(100, dtype=np.float16).reshape(10, 10) - np.float16(50)) / np.float16(10)
    input_dict = {"name": "elu_case_12", "features": features}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.raw_ops.Elu"] = tf_raw_ops_Elu_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Elu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Elu'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Elu', generated_inputs['tf.raw_ops.Elu'], lib="tf", suffix=0)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def tf_raw_ops_elu_inputs():
    list_of_inputs = []
    
    input_dict = {
        "name": "elu_op_1",
        "features": np.array(1.0, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "name": "elu_op_2",
        "features": np.array(0.0, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "name": "elu_op_3",
        "features": np.array(-1000.0, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "name": "elu_op_4",
        "features": np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "name": "elu_op_5",
        "features": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "name": "elu_op_6",
        "features": np.array([[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "name": "elu_op_7",
        "features": np.array([[[1.0, -1.0], [0.0, 2.0]], [[3.0, -3.0], [-2.0, 4.0]]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "name": "elu_op_8",
        "features": np.array([[-5.5, 3.3, 0.0, 7.7]], dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "name": "elu_op_9",
        "features": np.array([[1.5, -0.5], [-1.5, 0.5]], dtype=np.float16)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "name": "elu_op_10",
        "features": np.array([[[[1.0, -1.0], [0.5, -0.5]], [[2.0, -2.0], [1.5, -1.5]]]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.Elu"] = tf_raw_ops_elu_inputs()

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

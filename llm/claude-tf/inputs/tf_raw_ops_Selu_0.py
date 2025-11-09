
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

```python
import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_selu_inputs():
    list_of_inputs = []
    
    features = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {
        "features": features,
        "name": "selu_op_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    features = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    input_dict = {
        "features": features,
        "name": "selu_op_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    features = np.array([-1.5, 0.0, 1.5, -0.5, 2.0], dtype=np.float32)
    input_dict = {
        "features": features,
        "name": "selu_op_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    features = np.array([[1.0, -1.0], [2.0, -2.0]], dtype=np.float32)
    input_dict = {
        "features": features,
        "name": "selu_op_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    features = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    input_dict = {
        "features": features,
        "name": "selu_op_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    features = np.array([1.0, -1.0, 0.0], dtype=np.float64)
    input_dict = {
        "features": features,
        "name": "selu_op_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    features = np.array([0.5, -0.5, 1.0], dtype=np.float16)
    input_dict = {
        "features": features,
        "name": "selu_op_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    features = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_dict = {
        "features": features,
        "name": "selu_op_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    features = np.random.randn(10, 20).astype(np.float32)
    input_dict = {
        "features": features,
        "name": "selu_op_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    features = np.array([[[-2.0, -1.0], [0.0, 1.0]], [[2.0, 3.0], [4.0, 5.0]]], dtype=np.float64)
    input_dict = {
        "features": features,
        "name": "selu_op_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    features = np.array([10.0, -10.0, 100.0, -100.0], dtype=np.float32)
    input_dict = {
        "features": features,
        "name": "selu_op_11"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.Selu

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Selu' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Selu'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Selu', generated_inputs['tf.raw_ops.Selu'], lib="tf", suffix=0)

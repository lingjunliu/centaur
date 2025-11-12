
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

```python
import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_logsoftmax_inputs():
    list_of_inputs = []
    
    logits = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    input_dict = {
        "logits": logits,
        "name": "logsoftmax_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    logits = np.array([[-1.0, -2.0, -3.0], [0.0, 1.0, -1.0]], dtype=np.float64)
    input_dict = {
        "logits": logits,
        "name": "logsoftmax_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    logits = np.array([[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]], dtype=np.float16)
    input_dict = {
        "logits": logits,
        "name": "logsoftmax_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    logits = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    input_dict = {
        "logits": logits,
        "name": "logsoftmax_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    logits = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32)
    input_dict = {
        "logits": logits,
        "name": "logsoftmax_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    logits = np.zeros((2, 3), dtype=np.float32)
    input_dict = {
        "logits": logits,
        "name": "logsoftmax_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    logits = np.array([[10.0, -5.0, 0.0], [-2.0, 3.0, -1.0]], dtype=np.float32)
    input_dict = {
        "logits": logits,
        "name": "logsoftmax_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    logits = np.array([[100.0, 200.0, 300.0], [50.0, 150.0, 250.0]], dtype=np.float64)
    input_dict = {
        "logits": logits,
        "name": "logsoftmax_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    logits = np.array([[0.001, 0.002, 0.003], [0.004, 0.005, 0.006]], dtype=np.float32)
    input_dict = {
        "logits": logits,
        "name": "logsoftmax_9"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    logits = np.array([[-10.0, -20.0], [30.0, 40.0]], dtype=np.float32)
    input_dict = {
        "logits": logits,
        "name": "logsoftmax_10"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

generated_inputs["tf.raw_ops.Log

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.LogSoftmax' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.LogSoftmax'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.LogSoftmax', generated_inputs['tf.raw_ops.LogSoftmax'], lib="tf", suffix=0)

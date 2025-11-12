
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

```python
import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_imag_inputs():
    list_of_inputs = []
    
    input_tensor = np.array([-2.25 + 4.75j, 3.25 + 5.75j], dtype=np.complex64)
    input_dict = {
        "input": input_tensor,
        "Tout": np.float32,
        "name": "imag_op_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[1.0 + 2.0j, 3.0 + 4.0j], [5.0 + 6.0j, 7.0 + 8.0j]], dtype=np.complex64)
    input_dict = {
        "input": input_tensor,
        "Tout": np.float32,
        "name": "imag_op_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([1.5 + 2.5j, -3.5 + 4.5j], dtype=np.complex128)
    input_dict = {
        "input": input_tensor,
        "Tout": np.float64,
        "name": "imag_op_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[[1+1j, 2+2j], [3+3j, 4+4j]], [[5+5j, 6+6j], [7+7j, 8+8j]]], dtype=np.complex64)
    input_dict = {
        "input": input_tensor,
        "Tout": np.float32,
        "name": "imag_op_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([10.0 + 20.0j], dtype=np.complex64)
    input_dict = {
        "input": input_tensor,
        "Tout": np.float32,
        "name": "imag_op_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([1.0 - 2.0j, -3.0 - 4.0j, 5.0 - 6.0j], dtype=np.complex64)
    input_dict = {
        "input": input_tensor,
        "Tout": np.float32,
        "name": "imag_op_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([1.0 + 0.0j, 2.0 + 0.0j], dtype=np.complex64)
    input_dict = {
        "input": input_tensor,
        "Tout": np.float32,
        "name": "imag_op_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([0.0 + 1.0j, 0.0 + 2.0j], dtype=np.complex64)
    input_dict = {
        "input": input_tensor,
        "Tout": np.float32,
        "name": "imag_op_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[[-1-1j, -2-2j]], [[3+4j, -5+6j]]], dtype=np.complex128)
    input_dict = {
        "input": input_tensor,
        "Tout": np.float64,
        "name": "imag_op_9"
    }
    list_of_inputs.appen

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Imag' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Imag'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Imag', generated_inputs['tf.raw_ops.Imag'], lib="tf", suffix=0)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

```python
import numpy as np
import copy

def tf_quantization_fake_quant_with_min_max_vars_per_channel_gradient_inputs():
    list_of_inputs = []
    
    gradients = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    inputs = np.array([0.5, 1.5, 2.5], dtype=np.float32)
    min_val = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    max_val = np.array([3.0, 3.0, 3.0], dtype=np.float32)
    
    input_dict = {
        "gradients": gradients,
        "inputs": inputs,
        "min": min_val,
        "max": max_val,
        "num_bits": 8,
        "narrow_range": False,
        "name": "test1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    gradients = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    inputs = np.array([[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]], dtype=np.float32)
    min_val = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
    max_val = np.array([6.0, 6.0, 6.0], dtype=np.float32)
    
    input_dict = {
        "gradients": gradients,
        "inputs": inputs,
        "min": min_val,
        "max": max_val,
        "num_bits": 8,
        "narrow_range": True,
        "name": "test2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    gradients = np.random.randn(2, 3, 3, 4).astype(np.float32)
    inputs = np.random.randn(2, 3, 3, 4).astype(np.float32)
    min_val = np.array([-2.0, -2.0, -2.0, -2.0], dtype=np.float32)
    max_val = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32)
    
    input_dict = {
        "gradients": gradients,
        "inputs": inputs,
        "min": min_val,
        "max": max_val,
        "num_bits": 8,
        "narrow_range": False,
        "name": "test3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    gradients = np.array([0.1, 0.2], dtype=np.float32)
    inputs = np.array([1.0, 2.0], dtype=np.float32)
    min_val = np.array([0.5, 1.0], dtype=np.float32)
    max_val = np.array([4.0, 5.0], dtype=np.float32)
    
    input_dict = {
        "gradients": gradients,
        "inputs": inputs,
        "min": min_val,
        "max": max_val,
        "num_bits": 4,
        "narrow_range": False,
        "name": "test4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    gradients = np.array([-1.0, -2.0, -3.0, -4.0, -5.0], dtype=np.float32)
    inputs = np.array([-0.5, -1.5, -2.5, -3.5, -4.5], dtype=np.float32)
    

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.quantization.fake_quant_with_min_max_vars_per_channel_gradient' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.quantization.fake_quant_with_min_max_vars_per_channel_gradient'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.quantization.fake_quant_with_min_max_vars_per_channel_gradient', generated_inputs['tf.quantization.fake_quant_with_min_max_vars_per_channel_gradient'], lib="tf", suffix=0)

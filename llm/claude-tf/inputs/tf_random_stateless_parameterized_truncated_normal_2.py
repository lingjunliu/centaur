
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

```python
import numpy as np
import copy

def tf_random_stateless_parameterized_truncated_normal_inputs():
    list_of_inputs = []
    
    input_dict = {
        "shape": [10, 5],
        "seed": [7, 17],
        "means": 0.0,
        "stddevs": 1.0,
        "minvals": -2.0,
        "maxvals": 2.0,
        "name": "truncated_normal_1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "shape": [100],
        "seed": [42, 24],
        "means": 5.0,
        "stddevs": 2.0,
        "minvals": 0.0,
        "maxvals": 10.0,
        "name": "truncated_normal_2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "shape": [5, 3, 4],
        "seed": [1, 2],
        "means": -3.0,
        "stddevs": 0.5,
        "minvals": -5.0,
        "maxvals": -1.0,
        "name": "truncated_normal_3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "shape": [20, 10],
        "seed": [100, 200],
        "means": 0.0,
        "stddevs": 5.0,
        "minvals": -10.0,
        "maxvals": 10.0,
        "name": "truncated_normal_4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "shape": [8, 8, 8],
        "seed": [5, 10],
        "means": 1.0,
        "stddevs": 0.1,
        "minvals": 0.5,
        "maxvals": 1.5,
        "name": "truncated_normal_5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "shape": [2, 3, 4, 5],
        "seed": [99, 88],
        "means": 10.0,
        "stddevs": 3.0,
        "minvals": 5.0,
        "maxvals": 15.0,
        "name": "truncated_normal_6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "shape": [15, 20],
        "seed": [3, 7],
        "means": 2.0,
        "stddevs": 1.5,
        "minvals": -5.0,
        "maxvals": 8.0,
        "name": "truncated_normal_7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "shape": [1000],
        "seed": [123, 456],
        "means": 0.0,
        "stddevs": 1.0,
        "minvals": -3.0,
        "maxvals": 3.0,
        "name": "truncated_normal_8"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "shape": [4, 6],
        "seed": [11, 22],
        "means": 100.0,
        "stddevs": 20.0,
        "minvals": 50.0,
        "maxvals": 150.0,
        "name": "truncated_normal_9"
    

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.random.stateless_parameterized_truncated_normal_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.random.stateless_parameterized_truncated_normal_2'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.random.stateless_parameterized_truncated_normal', generated_inputs['tf.random.stateless_parameterized_truncated_normal_2'], lib="tf", suffix=2)

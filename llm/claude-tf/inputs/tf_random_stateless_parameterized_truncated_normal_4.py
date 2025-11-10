
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

```python
import numpy as np
import copy

def tf_random_stateless_parameterized_truncated_normal_inputs():
    list_of_inputs = []
    
    input_dict = {
        "shape": np.array([5, 3], dtype=np.int32),
        "seed": np.array([7, 17], dtype=np.int32),
        "means": 0.0,
        "stddevs": 1.0,
        "minvals": -2.0,
        "maxvals": 2.0,
        "name": "test1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "shape": np.array([10, 2, 3], dtype=np.int32),
        "seed": np.array([42, 123], dtype=np.int32),
        "means": 5.0,
        "stddevs": 2.0,
        "minvals": 0.0,
        "maxvals": 10.0,
        "name": "test2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "shape": np.array([8, 4], dtype=np.int32),
        "seed": np.array([1, 2], dtype=np.int32),
        "means": -3.0,
        "stddevs": 1.5,
        "minvals": -10.0,
        "maxvals": 0.0,
        "name": "test3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "shape": np.array([100], dtype=np.int32),
        "seed": np.array([99, 88], dtype=np.int32),
        "means": 0.0,
        "stddevs": 0.5,
        "minvals": -1.0,
        "maxvals": 1.0,
        "name": "test4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "shape": np.array([2, 3, 4, 5], dtype=np.int32),
        "seed": np.array([111, 222], dtype=np.int32),
        "means": 10.0,
        "stddevs": 5.0,
        "minvals": -5.0,
        "maxvals": 25.0,
        "name": "test5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "shape": np.array([20, 10], dtype=np.int32),
        "seed": np.array([5, 15], dtype=np.int32),
        "means": 0.0,
        "stddevs": 0.1,
        "minvals": -0.3,
        "maxvals": 0.3,
        "name": "test6"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "shape": np.array([7, 7, 7], dtype=np.int32),
        "seed": np.array([333, 444], dtype=np.int32),
        "means": 100.0,
        "stddevs": 50.0,
        "minvals": -100.0,
        "maxvals": 300.0,
        "name": "test7"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "shape": np.array([15], dtype=np.int32),
        "seed": np.array([0, 1], dtype=np.int32),
        "means": -5.5,
        "stddevs": 3.0,
        "minvals": -15.0,
        "maxvals":

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.random.stateless_parameterized_truncated_normal_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.random.stateless_parameterized_truncated_normal_4'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.random.stateless_parameterized_truncated_normal', generated_inputs['tf.random.stateless_parameterized_truncated_normal_4'], lib="tf", suffix=4)

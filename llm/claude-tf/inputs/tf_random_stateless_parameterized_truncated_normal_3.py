
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
        "means": np.array(0.0, dtype=np.float32),
        "stddevs": np.array(1.0, dtype=np.float32),
        "minvals": np.array(-2.0, dtype=np.float32),
        "maxvals": np.array(2.0, dtype=np.float32),
        "name": "test1"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "shape": np.array([10, 2, 3], dtype=np.int32),
        "seed": np.array([42, 24], dtype=np.int32),
        "means": np.array(0.0, dtype=np.float32),
        "stddevs": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "minvals": np.array([-1.0, -2.0, -1000.0], dtype=np.float32),
        "maxvals": np.array([10.0, 20.0, 1000.0], dtype=np.float32),
        "name": "test2"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "shape": np.array([10, 2, 3], dtype=np.int32),
        "seed": np.array([1, 2], dtype=np.int32),
        "means": np.array(0.0, dtype=np.float32),
        "stddevs": np.array([[1.0], [2.0]], dtype=np.float32),
        "minvals": np.array([-1.0, -2.0, -3.0], dtype=np.float32),
        "maxvals": np.array([[10.0], [5.0]], dtype=np.float32),
        "name": "test3"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "shape": np.array([4, 4], dtype=np.int32),
        "seed": np.array([100, 200], dtype=np.int32),
        "means": np.array(-5.0, dtype=np.float32),
        "stddevs": np.array(2.0, dtype=np.float32),
        "minvals": np.array(-10.0, dtype=np.float32),
        "maxvals": np.array(0.0, dtype=np.float32),
        "name": "test4"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "shape": np.array([8, 8], dtype=np.int32),
        "seed": np.array([50, 150], dtype=np.int32),
        "means": np.array(0.0, dtype=np.float32),
        "stddevs": np.array(10.0, dtype=np.float32),
        "minvals": np.array(-20.0, dtype=np.float32),
        "maxvals": np.array(20.0, dtype=np.float32),
        "name": "test5"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "shape": np.array([100], dtype=np.int32),
        "seed": np.array([25, 75], dtype=np.int32),
        "means": np.array(5.0, dtype=np.float32),
        "stddevs": np.array(1.5, dtype=np.float32),

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.random.stateless_parameterized_truncated_normal_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.random.stateless_parameterized_truncated_normal_3'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.random.stateless_parameterized_truncated_normal', generated_inputs['tf.random.stateless_parameterized_truncated_normal_3'], lib="tf", suffix=3)

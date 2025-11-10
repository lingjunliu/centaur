
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

```python
import numpy as np
import copy

def tf_random_stateless_gamma_inputs():
    list_of_inputs = []
    
    input_dict = {
        "shape": np.array([10, 2], dtype=np.int32),
        "seed": np.array([12, 34], dtype=np.int32),
        "alpha": np.array([0.5, 1.5], dtype=np.float32),
        "beta": np.array([1.0, 2.0], dtype=np.float32),
        "dtype": np.float32,
        "name": "gamma_basic"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "shape": np.array([7, 5, 2], dtype=np.int32),
        "seed": np.array([42, 100], dtype=np.int32),
        "alpha": np.array([0.5, 1.5], dtype=np.float32),
        "beta": np.array([2.0, 3.0], dtype=np.float32),
        "dtype": np.float32,
        "name": "gamma_3d"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "shape": np.array([30, 3, 2], dtype=np.int32),
        "seed": np.array([12, 34], dtype=np.int32),
        "alpha": np.array([[1.], [3.], [5.]], dtype=np.float32),
        "beta": np.array([[3., 4.]], dtype=np.float32),
        "dtype": np.float32,
        "name": "gamma_broadcast"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "shape": np.array([5, 3], dtype=np.int32),
        "seed": np.array([99, 88], dtype=np.int32),
        "alpha": np.array([10.0, 20.0, 50.0], dtype=np.float32),
        "beta": np.array([1.0, 1.0, 1.0], dtype=np.float32),
        "dtype": np.float32,
        "name": "gamma_large_alpha"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "shape": np.array([8, 4], dtype=np.int32),
        "seed": np.array([1, 2], dtype=np.int32),
        "alpha": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        "beta": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        "dtype": np.float32,
        "name": "gamma_small_alpha"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "shape": np.array([10, 2], dtype=np.int32),
        "seed": np.array([55, 66], dtype=np.int32),
        "alpha": np.array([1.0, 2.0], dtype=np.float32),
        "beta": np.array([10.0, 20.0], dtype=np.float32),
        "dtype": np.float32,
        "name": "gamma_large_beta"
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_dict = {
        "shape": np.array([5], dtype=np.int32),
        "seed": np.array([7, 9], dtype=np.int32),
        "alpha": np.array([2.5], dtype=np.float32),
        "beta": np.array([1.5], dtype=np.float

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.random.stateless_gamma_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.random.stateless_gamma_3'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.random.stateless_gamma', generated_inputs['tf.random.stateless_gamma_3'], lib="tf", suffix=3)

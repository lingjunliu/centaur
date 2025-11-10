
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

```python
import tensorflow as tf
import numpy as np
import copy

def tf_random_stateless_gamma_inputs():
    list_of_inputs = []
    
    shape = (10,)
    seed = np.array([12, 34], dtype=np.int32)
    alpha = np.array(1.0, dtype=np.float32)
    beta = np.array(1.0, dtype=np.float32)
    dtype = np.float32
    name = "gamma_sample_1"
    input_dict = {"shape": shape, "seed": seed, "alpha": alpha, "beta": beta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    shape = (10, 2)
    seed = np.array([12, 34], dtype=np.int32)
    alpha = np.array([0.5, 1.5], dtype=np.float32)
    beta = np.array([1.0, 2.0], dtype=np.float32)
    dtype = np.float32
    name = "gamma_sample_2"
    input_dict = {"shape": shape, "seed": seed, "alpha": alpha, "beta": beta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    shape = (7, 5, 2)
    seed = np.array([42, 56], dtype=np.int32)
    alpha = np.array([0.5, 1.5], dtype=np.float32)
    beta = np.array([2.0, 3.0], dtype=np.float32)
    dtype = np.float32
    name = "gamma_sample_3"
    input_dict = {"shape": shape, "seed": seed, "alpha": alpha, "beta": beta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    shape = (30, 3, 2)
    seed = np.array([12, 34], dtype=np.int64)
    alpha = np.array([[1.], [3.], [5.]], dtype=np.float32)
    beta = np.array([[3., 4.]], dtype=np.float32)
    dtype = np.float32
    name = "gamma_sample_4"
    input_dict = {"shape": shape, "seed": seed, "alpha": alpha, "beta": beta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    shape = (5, 3)
    seed = np.array([100, 200], dtype=np.int32)
    alpha = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    beta = np.array([5.0, 10.0, 15.0], dtype=np.float32)
    dtype = np.float32
    name = "gamma_sample_5"
    input_dict = {"shape": shape, "seed": seed, "alpha": alpha, "beta": beta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    shape = (20, 4)
    seed = np.array([7, 14], dtype=np.int32)
    alpha = np.array([0.1, 0.3, 0.5, 0.7], dtype=np.float32)
    beta = np.array([1.0, 1.5, 2.0, 2.5], dtype=np.float32)
    dtype = np.float32
    name = "gamma_sample_6"
    input_dict = {"shape": shape, "seed": seed, "alpha": alpha, "beta": beta, "dtype": dtype, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    shape = (15, 2, 3)
    seed = np.array([99, 88], dtype=np.int32

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.random.stateless_gamma_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.random.stateless_gamma_2'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.random.stateless_gamma', generated_inputs['tf.random.stateless_gamma_2'], lib="tf", suffix=2)

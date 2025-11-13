
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

```python
import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_fractional_max_pool_inputs():
    list_of_inputs = []
    
    value = np.random.rand(1, 10, 10, 3).astype(np.float32)
    pooling_ratio = [1.0, 1.5, 1.5, 1.0]
    pseudo_random = False
    overlapping = False
    deterministic = True
    seed = 0
    seed2 = 0
    name = "test1"
    
    input_dict = {
        "value": value,
        "pooling_ratio": pooling_ratio,
        "pseudo_random": pseudo_random,
        "overlapping": overlapping,
        "deterministic": deterministic,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    value = np.random.rand(2, 20, 20, 4).astype(np.float32)
    pooling_ratio = [1.0, 2.0, 2.0, 1.0]
    pseudo_random = True
    overlapping = False
    deterministic = True
    seed = 42
    seed2 = 100
    name = "test2"
    
    input_dict = {
        "value": value,
        "pooling_ratio": pooling_ratio,
        "pseudo_random": pseudo_random,
        "overlapping": overlapping,
        "deterministic": deterministic,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    value = np.random.rand(1, 15, 15, 1).astype(np.float64)
    pooling_ratio = [1.0, 1.44, 1.73, 1.0]
    pseudo_random = False
    overlapping = True
    deterministic = True
    seed = 10
    seed2 = 20
    name = "test3"
    
    input_dict = {
        "value": value,
        "pooling_ratio": pooling_ratio,
        "pseudo_random": pseudo_random,
        "overlapping": overlapping,
        "deterministic": deterministic,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    value = np.random.rand(3, 12, 16, 2).astype(np.float32)
    pooling_ratio = [1.0, 1.2, 1.6, 1.0]
    pseudo_random = True
    overlapping = True
    deterministic = True
    seed = 7
    seed2 = 14
    name = "test4"
    
    input_dict = {
        "value": value,
        "pooling_ratio": pooling_ratio,
        "pseudo_random": pseudo_random,
        "overlapping": overlapping,
        "deterministic": deterministic,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    value = np.random.randint(0, 100, size=(1, 8, 8, 3)).astype(np.int32)
    pooling_ratio = [1.0, 2.0, 2.0, 1.0]
    pseudo_random = False
    overlapping = False
    deterministic = True
    seed = 5
    seed2 = 10
    name = "test5"
    
    input_dict = {
        "value": value,
        "pooling_ratio": pooling_ratio,
        "pseudo_random": pseudo_random,
        "overlapping": overlapping,
        "deterministic": deterministic,
        "seed": seed,
        "

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.FractionalMaxPool' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.FractionalMaxPool'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.FractionalMaxPool', generated_inputs['tf.raw_ops.FractionalMaxPool'], lib="tf", suffix=0)

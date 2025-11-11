
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

```python
import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_raw_ops_RandomShuffle_inputs():
    list_of_inputs = []
    
    value = np.array([1, 2, 3, 4, 5])
    seed = 0
    seed2 = 0
    name = "shuffle_1d"
    input_dict = {
        "value": value,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    value = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    seed = 42
    seed2 = 0
    name = "shuffle_2d"
    input_dict = {
        "value": value,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    value = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])
    seed = 123
    seed2 = 456
    name = "shuffle_3d"
    input_dict = {
        "value": value,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    value = np.array([100])
    seed = 1
    seed2 = 1
    name = "shuffle_single"
    input_dict = {
        "value": value,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    value = np.arange(100).reshape(50, 2)
    seed = 999
    seed2 = 111
    name = "shuffle_large"
    input_dict = {
        "value": value,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    value = np.array([[-1, -2], [-3, -4], [-5, -6]])
    seed = 0
    seed2 = 10
    name = "shuffle_negative"
    input_dict = {
        "value": value,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    value = np.ones((5, 3, 2, 4))
    seed = 7
    seed2 = 7
    name = "shuffle_4d"
    input_dict = {
        "value": value,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    value = np.array([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]], dtype=np.float64)
    seed = 0
    seed2 = 0
    name = "shuffle_float64"
    input_dict = {
        "value": value,
        "seed": seed,
        "seed2": seed2,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    value = np.array([[10, 20], [30, 40]])
    seed = 55
    seed2 = 66
    name = "shuffle_int"
    input_dict = {
        "value": value,
        "see

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.RandomShuffle' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.RandomShuffle'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.RandomShuffle', generated_inputs['tf.raw_ops.RandomShuffle'], lib="tf", suffix=0)

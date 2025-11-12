
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

```python
import numpy as np
import copy

def tf_raw_ops_bucketize_inputs():
    list_of_inputs = []
    
    input_tensor = np.array([[-5, 10000], [150, 10], [5, 100]], dtype=np.int32)
    boundaries = [0.0, 10.0, 100.0]
    name = "bucketize_1"
    input_dict = {
        "input": input_tensor,
        "boundaries": boundaries,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([1.5, 2.5, 5.5, 10.5], dtype=np.float32)
    boundaries = [0.0, 5.0, 10.0]
    name = "bucketize_2"
    input_dict = {
        "input": input_tensor,
        "boundaries": boundaries,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float64)
    boundaries = [2.5, 5.5]
    name = "bucketize_3"
    input_dict = {
        "input": input_tensor,
        "boundaries": boundaries,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([-100, -50, -10, 0, 10, 50, 100], dtype=np.int64)
    boundaries = [-75.0, -25.0, 25.0, 75.0]
    name = "bucketize_4"
    input_dict = {
        "input": input_tensor,
        "boundaries": boundaries,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([42.0], dtype=np.float32)
    boundaries = [10.0, 20.0, 30.0, 40.0, 50.0]
    name = "bucketize_5"
    input_dict = {
        "input": input_tensor,
        "boundaries": boundaries,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    boundaries = []
    name = "bucketize_6"
    input_dict = {
        "input": input_tensor,
        "boundaries": boundaries,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[0.1, 0.5], [1.1, 1.5]], dtype=np.float64)
    boundaries = [0.5, 1.0, 1.5]
    name = "bucketize_7"
    input_dict = {
        "input": input_tensor,
        "boundaries": boundaries,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    boundaries = [2.0]
    name = "bucketize_8"
    input_dict = {
        "input": input_tensor,
        "boundaries": boundaries,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    input_tensor = np.array([[10, 20], [30, 40]], dtype=np.int64)
    boundaries = [15.0, 25.0, 35

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Bucketize' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Bucketize'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Bucketize', generated_inputs['tf.raw_ops.Bucketize'], lib="tf", suffix=0)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

```python
import tensorflow as tf
import numpy as np
import copy

def tf_raw_ops_floormod_inputs():
    list_of_inputs = []
    
    x = np.array([10, 20, 30], dtype=np.int32)
    y = np.array([3, 7, 4], dtype=np.int32)
    input_dict = {"name": "floormod_op_1", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([-10, -20, 15], dtype=np.int32)
    y = np.array([3, 7, -4], dtype=np.int32)
    input_dict = {"name": "floormod_op_2", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([10.5, 20.3, 30.7], dtype=np.float32)
    y = np.array([3.2, 7.1, 4.5], dtype=np.float32)
    input_dict = {"name": "floormod_op_3", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([100.5, 200.3], dtype=np.float64)
    y = np.array([30.2, 70.1], dtype=np.float64)
    input_dict = {"name": "floormod_op_4", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[10, 20], [30, 40]], dtype=np.int64)
    y = np.array([[3, 7], [4, 9]], dtype=np.int64)
    input_dict = {"name": "floormod_op_5", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[[5, 10], [15, 20]], [[25, 30], [35, 40]]], dtype=np.int32)
    y = np.array([[[2, 3], [4, 5]], [[6, 7], [8, 9]]], dtype=np.int32)
    input_dict = {"name": "floormod_op_6", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([10, 20, 30, 40], dtype=np.int32)
    y = np.array([3], dtype=np.int32)
    input_dict = {"name": "floormod_op_7", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([100, 200, 150], dtype=np.uint8)
    y = np.array([30, 70, 40], dtype=np.uint8)
    input_dict = {"name": "floormod_op_8", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([1000, 2000, 3000], dtype=np.int16)
    y = np.array([300, 700, 400], dtype=np.int16)
    input_dict = {"name": "floormod_op_9", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([5], dtype=np.int8)
    y = np.array([2], dtype=np.int8)
    input_dict = {"name": "floormod_op_10", "x": x, "y": y}
    list

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.FloorMod' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.FloorMod'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.FloorMod', generated_inputs['tf.raw_ops.FloorMod'], lib="tf", suffix=0)

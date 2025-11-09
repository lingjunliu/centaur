
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

```python
import numpy as np
import copy

def tf_raw_ops_maximum_inputs():
    list_of_inputs = []
    
    x = np.array([0., 0., 0., 0.], dtype=np.float32)
    y = np.array([-2., 0., 2., 5.], dtype=np.float32)
    input_dict = {"name": "maximum_op", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([-5., 0., 0., 0.], dtype=np.float32)
    y = np.array([-3.], dtype=np.float32)
    input_dict = {"name": "maximum_broadcast", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    y = np.array([[2, 1, 4], [3, 6, 5]], dtype=np.int32)
    input_dict = {"name": "maximum_2d", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([-10, -5, 0, 5, 10], dtype=np.int64)
    y = np.array([-8, -6, -2, 3, 12], dtype=np.int64)
    input_dict = {"name": "maximum_negative", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[[1.5, 2.3], [3.1, 4.7]], [[5.2, 6.8], [7.4, 8.9]]], dtype=np.float64)
    y = np.array([[[2.0, 2.0], [3.5, 4.0]], [[5.0, 7.0], [7.0, 9.0]]], dtype=np.float64)
    input_dict = {"name": "maximum_3d", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
    y = np.array([2, 3], dtype=np.int32)
    input_dict = {"name": "maximum_broadcast_2d", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([10, 20, 30, 40], dtype=np.uint8)
    y = np.array([15, 18, 35, 38], dtype=np.uint8)
    input_dict = {"name": "maximum_uint8", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([-100, -50, 0, 50, 100], dtype=np.int16)
    y = np.array([-80, -60, -10, 30, 120], dtype=np.int16)
    input_dict = {"name": "maximum_int16", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([5.5, 6.7, 8.2], dtype=np.float32)
    y = np.array([5.0], dtype=np.float32)
    input_dict = {"name": "maximum_scalar", "x": x, "y": y}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([100, 200, 300], dtype=np.

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.Maximum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.Maximum'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.Maximum', generated_inputs['tf.raw_ops.Maximum'], lib="tf", suffix=0)

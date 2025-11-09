
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

```python
import numpy as np
import copy

def tf_math_maximum_inputs():
    list_of_inputs = []
    
    x = np.array([0., 0., 0., 0.], dtype=np.float32)
    y = np.array([-2., 0., 2., 5.], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": "maximum_op"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([-5., 0., 0., 0.], dtype=np.float32)
    y = np.array([-3.], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": "broadcast_maximum"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([1, 5, -3, 8], dtype=np.int32)
    y = np.array([2, 3, -2, 7], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": "int_maximum"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64)
    y = np.array([[2.0, 2.0], [3.0, 5.0]], dtype=np.float64)
    input_dict = {"x": x, "y": y, "name": "matrix_maximum"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int64)
    y = np.array([[[2, 1], [4, 3]], [[6, 5], [8, 7]]], dtype=np.int64)
    input_dict = {"x": x, "y": y, "name": "tensor3d_maximum"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([10, 20, 30, 40], dtype=np.uint8)
    y = np.array([15, 15, 35, 35], dtype=np.uint8)
    input_dict = {"x": x, "y": y, "name": "uint8_maximum"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([-100, -50, 0, 50], dtype=np.int16)
    y = np.array([-75, -60, 10, 40], dtype=np.int16)
    input_dict = {"x": x, "y": y, "name": "int16_maximum"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[1., 2., 3.], [4., 5., 6.]], dtype=np.float32)
    y = np.array([2., 4., 5.], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": "broadcast_2d_1d"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([5.0], dtype=np.float32)
    y = np.array([3.0], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": "scalar_maximum"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([-10, -20, -30], dtype=np.int8)
    y = np.array([-15, -18, -25], dtype=np.int8)
    input_dict = {"x": x, "y": y, "name":

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.maximum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.maximum'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.maximum', generated_inputs['tf.math.maximum'], lib="tf", suffix=0)

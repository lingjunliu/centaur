
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

```python
import numpy as np
import copy

def tf_math_pow_inputs():
    list_of_inputs = []
    
    x = np.array([[2, 2], [3, 3]], dtype=np.int32)
    y = np.array([[8, 16], [2, 3]], dtype=np.int32)
    input_dict = {"x": x, "y": y, "name": "pow_op_1"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[2.5, 3.0], [4.5, 1.5]], dtype=np.float32)
    y = np.array([[2.0, 3.0], [2.0, 4.0]], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": "pow_op_2"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([1.5, 2.5, 3.5], dtype=np.float64)
    y = np.array([2.0, 2.0, 2.0], dtype=np.float64)
    input_dict = {"x": x, "y": y, "name": "pow_op_3"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([-2.0, -3.0, 4.0], dtype=np.float32)
    y = np.array([2.0, 3.0, 2.0], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": "pow_op_4"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([2, 3, 4, 5], dtype=np.int64)
    y = np.array([1, 2, 3, 4], dtype=np.int64)
    input_dict = {"x": x, "y": y, "name": "pow_op_5"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    y = np.array([[[2.0, 2.0], [2.0, 2.0]], [[2.0, 2.0], [2.0, 2.0]]], dtype=np.float32)
    input_dict = {"x": x, "y": y, "name": "pow_op_6"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([5.0], dtype=np.float16)
    y = np.array([3.0], dtype=np.float16)
    input_dict = {"x": x, "y": y, "name": "pow_op_7"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([1+2j, 3+4j], dtype=np.complex64)
    y = np.array([2+0j, 1+1j], dtype=np.complex64)
    input_dict = {"x": x, "y": y, "name": "pow_op_8"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([2+3j, 1+1j], dtype=np.complex128)
    y = np.array([2+0j, 3+0j], dtype=np.complex128)
    input_dict = {"x": x, "y": y, "name": "pow_op_9"}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x = np.array([10, 5, 2], dtype=

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.pow' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.pow'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.pow', generated_inputs['tf.math.pow'], lib="tf", suffix=0)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

```python
import tensorflow as tf
import numpy as np
import copy

def tf_math_nextafter_inputs():
    list_of_inputs = []
    
    x1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x2 = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    name = "nextafter_1"
    input_dict = {"x1": x1, "x2": x2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x1 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    x2 = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    name = "nextafter_2"
    input_dict = {"x1": x1, "x2": x2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    x2 = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
    name = "nextafter_3"
    input_dict = {"x1": x1, "x2": x2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x1 = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    x2 = np.array([-2.0, -3.0, -4.0], dtype=np.float32)
    name = "nextafter_4"
    input_dict = {"x1": x1, "x2": x2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x1 = np.array([1.0, -1.0, 0.0], dtype=np.float64)
    x2 = np.array([-1.0, 1.0, 1.0], dtype=np.float64)
    name = "nextafter_5"
    input_dict = {"x1": x1, "x2": x2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x1 = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    x2 = np.array([[[1.1, 2.1], [3.1, 4.1]], [[5.1, 6.1], [7.1, 8.1]]], dtype=np.float32)
    name = "nextafter_6"
    input_dict = {"x1": x1, "x2": x2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x1 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    x2 = np.array([1.0, -1.0, 0.0], dtype=np.float32)
    name = "nextafter_7"
    input_dict = {"x1": x1, "x2": x2, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    x1 = np.array([100.0, 200.0], dtype=np.float64)
    x2 = np.array([99.0, 199.0], dtype=np.float64)
    name = "nextafter_8"
    input_dict = {"x1": x1, "x2": x2, "name": name}

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.math.nextafter' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.math.nextafter'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.math.nextafter', generated_inputs['tf.math.nextafter'], lib="tf", suffix=0)

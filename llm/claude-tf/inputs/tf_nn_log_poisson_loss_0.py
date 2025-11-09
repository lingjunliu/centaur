
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

```python
import numpy as np
import copy

def tf_nn_log_poisson_loss_inputs():
    list_of_inputs = []
    
    targets = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    log_input = np.array([0.5, 1.0, 1.5], dtype=np.float32)
    compute_full_loss = False
    name = "test1"
    input_dict = {
        "targets": targets,
        "log_input": log_input,
        "compute_full_loss": compute_full_loss,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    targets = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    log_input = np.array([[0.5, 1.0], [1.5, 2.0]], dtype=np.float32)
    compute_full_loss = True
    name = "test2"
    input_dict = {
        "targets": targets,
        "log_input": log_input,
        "compute_full_loss": compute_full_loss,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    targets = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float64)
    log_input = np.array([[[0.5, 1.0], [1.5, 2.0]], [[2.5, 3.0], [3.5, 4.0]]], dtype=np.float64)
    compute_full_loss = False
    name = "test3"
    input_dict = {
        "targets": targets,
        "log_input": log_input,
        "compute_full_loss": compute_full_loss,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    targets = np.array([0.0, 1.0, 2.0, 0.0], dtype=np.float32)
    log_input = np.array([0.1, 0.5, 1.0, 0.2], dtype=np.float32)
    compute_full_loss = True
    name = "test4"
    input_dict = {
        "targets": targets,
        "log_input": log_input,
        "compute_full_loss": compute_full_loss,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    targets = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    log_input = np.array([-1.0, -0.5, 0.0], dtype=np.float32)
    compute_full_loss = False
    name = "test5"
    input_dict = {
        "targets": targets,
        "log_input": log_input,
        "compute_full_loss": compute_full_loss,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    targets = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    log_input = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    compute_full_loss = True
    name = "test6"
    input_dict = {
        "targets": targets,
        "log_input": log_input,
        "compute_full_loss": compute_full_loss,
        "name": name
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    targets = np.array([[5.0]], dtype=np.float32)
    log_input =

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nn.log_poisson_loss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.log_poisson_loss'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.nn.log_poisson_loss', generated_inputs['tf.nn.log_poisson_loss'], lib="tf", suffix=0)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

```python
import numpy as np
import copy

def tf_image_adjust_gamma_inputs():
    list_of_inputs = []
    
    image = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=np.float32)
    gamma = 0.5
    gain = 1.0
    input_dict = {"image": image, "gamma": gamma, "gain": gain}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.array([[[50.0, 100.0, 150.0], [200.0, 250.0, 255.0]]], dtype=np.float32)
    gamma = 2.0
    gain = 1.0
    input_dict = {"image": image, "gamma": gamma, "gain": gain}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.array([[[128.0, 128.0, 128.0]]], dtype=np.float32)
    gamma = 1.5
    gain = 1.0
    input_dict = {"image": image, "gamma": gamma, "gain": gain}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.array([[[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], [[70.0, 80.0, 90.0], [100.0, 110.0, 120.0]]], dtype=np.float32)
    gamma = 1.0
    gain = 2.0
    input_dict = {"image": image, "gamma": gamma, "gain": gain}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.array([[[100.0, 150.0, 200.0]]], dtype=np.float32)
    gamma = 1.0
    gain = 0.5
    input_dict = {"image": image, "gamma": gamma, "gain": gain}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.random.rand(5, 5, 3).astype(np.float32) * 255
    gamma = 0.8
    gain = 1.2
    input_dict = {"image": image, "gamma": gamma, "gain": gain}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.array([[[100, 150, 200], [50, 75, 25]]], dtype=np.uint8)
    gamma = 0.4
    gain = 1.0
    input_dict = {"image": image, "gamma": gamma, "gain": gain}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], dtype=np.float32)
    gamma = 0.1
    gain = 1.0
    input_dict = {"image": image, "gamma": gamma, "gain": gain}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.array([[[100.0, 150.0, 200.0], [50.0, 75.0, 125.0]]], dtype=np.float32)
    gamma = 3.0
    gain = 1.0
    input_dict = {"image": image, "gamma": gamma, "gain": gain}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.array([[[10.0, 20.0, 30.0]]], dtype=np.float32)
    gamma = 0.5
    gain = 2

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.adjust_gamma' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.adjust_gamma'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.adjust_gamma', generated_inputs['tf.image.adjust_gamma'], lib="tf", suffix=0)

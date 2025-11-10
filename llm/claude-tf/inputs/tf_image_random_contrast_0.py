
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

```python
import numpy as np
import copy

def tf_image_random_contrast_inputs():
    list_of_inputs = []
    
    image = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=np.float32)
    lower = 0.2
    upper = 0.5
    seed = 42
    input_dict = {"image": image, "lower": lower, "upper": upper, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.random.rand(2, 32, 32, 3).astype(np.float32)
    lower = 0.5
    upper = 1.5
    seed = 123
    input_dict = {"image": image, "lower": lower, "upper": upper, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.random.rand(64, 64, 1).astype(np.float32)
    lower = 0.8
    upper = 1.2
    seed = 999
    input_dict = {"image": image, "lower": lower, "upper": upper, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.ones((16, 16, 3), dtype=np.float32) * 0.5
    lower = 0.1
    upper = 2.0
    seed = 1
    input_dict = {"image": image, "lower": lower, "upper": upper, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.random.rand(10, 10, 4).astype(np.float32)
    lower = 0.3
    upper = 0.7
    seed = 555
    input_dict = {"image": image, "lower": lower, "upper": upper, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.array([[[0.1, 0.2, 0.3]]], dtype=np.float32)
    lower = 0.0
    upper = 1.0
    seed = 0
    input_dict = {"image": image, "lower": lower, "upper": upper, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.random.rand(128, 128, 3).astype(np.float32)
    lower = 0.9
    upper = 1.1
    seed = 2023
    input_dict = {"image": image, "lower": lower, "upper": upper, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.random.rand(2, 8, 8, 8, 3).astype(np.float32)
    lower = 0.4
    upper = 1.6
    seed = 777
    input_dict = {"image": image, "lower": lower, "upper": upper, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.zeros((5, 5, 3), dtype=np.float32)
    lower = 0.25
    upper = 0.75
    seed = 100
    input_dict = {"image": image, "lower": lower, "upper": upper, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.random.rand(20, 20, 3).astype(np.float32) * 255.0
    lower = 0.6
    upper = 1.4
    seed = 333
    input_dict = {"image": image

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.random_contrast' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.random_contrast'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.random_contrast', generated_inputs['tf.image.random_contrast'], lib="tf", suffix=0)

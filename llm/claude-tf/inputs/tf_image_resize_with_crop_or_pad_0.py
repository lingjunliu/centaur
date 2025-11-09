
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

```python
import numpy as np
import copy

def tf_image_resize_with_crop_or_pad_inputs():
    list_of_inputs = []
    
    image = np.arange(75).reshape(5, 5, 3).astype(np.float32)
    target_height = 3
    target_width = 3
    input_dict = {
        "image": image,
        "target_height": target_height,
        "target_width": target_width
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.arange(1, 28).reshape(3, 3, 3).astype(np.float32)
    target_height = 5
    target_width = 5
    input_dict = {
        "image": image,
        "target_height": target_height,
        "target_width": target_width
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.arange(192).reshape(2, 8, 8, 3).astype(np.float32)
    target_height = 4
    target_width = 4
    input_dict = {
        "image": image,
        "target_height": target_height,
        "target_width": target_width
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.arange(48).reshape(2, 4, 4, 3).astype(np.float32)
    target_height = 6
    target_width = 6
    input_dict = {
        "image": image,
        "target_height": target_height,
        "target_width": target_width
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.arange(60).reshape(5, 4, 3).astype(np.float32)
    target_height = 3
    target_width = 4
    input_dict = {
        "image": image,
        "target_height": target_height,
        "target_width": target_width
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.arange(60).reshape(4, 5, 3).astype(np.float32)
    target_height = 4
    target_width = 3
    input_dict = {
        "image": image,
        "target_height": target_height,
        "target_width": target_width
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.arange(36).reshape(3, 4, 3).astype(np.float32)
    target_height = 5
    target_width = 4
    input_dict = {
        "image": image,
        "target_height": target_height,
        "target_width": target_width
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.arange(36).reshape(4, 3, 3).astype(np.float32)
    target_height = 4
    target_width = 5
    input_dict = {
        "image": image,
        "target_height": target_height,
        "target_width": target_width
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.arange(27).reshape(3, 3, 3).astype(np.float32)
    target_height = 3
    target_width = 3
    input_dict = {
        "image": image,
        "target_height": target_height,
        "target_width": target_width
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.ones((10, 10, 1)).astype(np.float32)
    target_height = 

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.resize_with_crop_or_pad' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.resize_with_crop_or_pad'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.resize_with_crop_or_pad', generated_inputs['tf.image.resize_with_crop_or_pad'], lib="tf", suffix=0)

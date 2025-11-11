
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

```python
import numpy as np
import copy

def tf_image_crop_to_bounding_box_inputs():
    list_of_inputs = []
    
    image = np.random.rand(10, 10, 3).astype(np.float32)
    offset_height = 2
    offset_width = 2
    target_height = 5
    target_width = 5
    input_dict = {
        "image": image,
        "offset_height": offset_height,
        "offset_width": offset_width,
        "target_height": target_height,
        "target_width": target_width
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.random.rand(4, 20, 20, 3).astype(np.float32)
    offset_height = 5
    offset_width = 5
    target_height = 10
    target_width = 10
    input_dict = {
        "image": image,
        "offset_height": offset_height,
        "offset_width": offset_width,
        "target_height": target_height,
        "target_width": target_width
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.random.rand(15, 15, 1).astype(np.float32)
    offset_height = 0
    offset_width = 0
    target_height = 8
    target_width = 8
    input_dict = {
        "image": image,
        "offset_height": offset_height,
        "offset_width": offset_width,
        "target_height": target_height,
        "target_width": target_width
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.random.rand(5, 5, 3).astype(np.float32)
    offset_height = 2
    offset_width = 2
    target_height = 1
    target_width = 1
    input_dict = {
        "image": image,
        "offset_height": offset_height,
        "offset_width": offset_width,
        "target_height": target_height,
        "target_width": target_width
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.random.rand(2, 50, 50, 16).astype(np.float32)
    offset_height = 10
    offset_width = 15
    target_height = 25
    target_width = 20
    input_dict = {
        "image": image,
        "offset_height": offset_height,
        "offset_width": offset_width,
        "target_height": target_height,
        "target_width": target_width
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.random.rand(30, 30, 3).astype(np.float32)
    offset_height = 20
    offset_width = 20
    target_height = 10
    target_width = 10
    input_dict = {
        "image": image,
        "offset_height": offset_height,
        "offset_width": offset_width,
        "target_height": target_height,
        "target_width": target_width
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.random.rand(100, 100, 1).astype(np.float32)
    offset_height = 25
    offset_width = 25
    target_height = 50
    target_width = 50
    input_dict = {
        "image": image,
        "offset_height": offset_height,
        "offset_width": offset_width,
        "target_height": target_height,
        "target_width": target_width
    }
    list_of_inputs.append(copy.deepcopy(input_dict))
    

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.crop_to_bounding_box' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.crop_to_bounding_box'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.crop_to_bounding_box', generated_inputs['tf.image.crop_to_bounding_box'], lib="tf", suffix=0)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

```python
import tensorflow as tf
import numpy as np
import copy

def tf_image_random_jpeg_quality_inputs():
    list_of_inputs = []
    
    image = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.uint8)
    min_jpeg_quality = 75
    max_jpeg_quality = 95
    seed = 42
    input_dict = {"image": image, "min_jpeg_quality": min_jpeg_quality, "max_jpeg_quality": max_jpeg_quality, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.array([[[100], [150]], [[200], [250]]], dtype=np.uint8)
    min_jpeg_quality = 0
    max_jpeg_quality = 100
    seed = 123
    input_dict = {"image": image, "min_jpeg_quality": min_jpeg_quality, "max_jpeg_quality": max_jpeg_quality, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.random.randint(0, 256, size=(10, 10, 3), dtype=np.uint8)
    min_jpeg_quality = 50
    max_jpeg_quality = 80
    seed = 999
    input_dict = {"image": image, "min_jpeg_quality": min_jpeg_quality, "max_jpeg_quality": max_jpeg_quality, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.ones((5, 5, 3), dtype=np.uint8) * 128
    min_jpeg_quality = 10
    max_jpeg_quality = 20
    seed = 1
    input_dict = {"image": image, "min_jpeg_quality": min_jpeg_quality, "max_jpeg_quality": max_jpeg_quality, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    min_jpeg_quality = 90
    max_jpeg_quality = 100
    seed = 555
    input_dict = {"image": image, "min_jpeg_quality": min_jpeg_quality, "max_jpeg_quality": max_jpeg_quality, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.array([[[255, 128, 64]]], dtype=np.uint8)
    min_jpeg_quality = 30
    max_jpeg_quality = 70
    seed = 2022
    input_dict = {"image": image, "min_jpeg_quality": min_jpeg_quality, "max_jpeg_quality": max_jpeg_quality, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.random.randint(0, 256, size=(20, 5, 1), dtype=np.uint8)
    min_jpeg_quality = 25
    max_jpeg_quality = 50
    seed = 789
    input_dict = {"image": image, "min_jpeg_quality": min_jpeg_quality, "max_jpeg_quality": max_jpeg_quality, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.full((15, 15, 3), 200, dtype=np.uint8)
    min_jpeg_quality = 40
    max_jpeg_quality = 60
    seed = 333
    input_dict = {"image": image, "min_jpeg_quality": min_jpeg_quality, "max_jpeg_quality": max_jpeg_quality, "seed": seed}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    image = np.random.randint(50, 200, size

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.random_jpeg_quality' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.random_jpeg_quality'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.random_jpeg_quality', generated_inputs['tf.image.random_jpeg_quality'], lib="tf", suffix=0)

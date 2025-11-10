
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

```python
import numpy as np
import copy

def tf_raw_ops_DrawBoundingBoxes_inputs():
    list_of_inputs = []
    
    images = np.random.rand(1, 100, 100, 3).astype(np.float32)
    boxes = np.array([[[0.1, 0.2, 0.5, 0.9]]]).astype(np.float32)
    name = "draw_boxes_1"
    input_dict = {"images": images, "boxes": boxes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    images = np.random.rand(2, 200, 150, 3).astype(np.float32)
    boxes = np.array([[[0.1, 0.1, 0.9, 0.9], [0.2, 0.2, 0.8, 0.8]], 
                      [[0.3, 0.3, 0.7, 0.7], [0.0, 0.0, 0.5, 0.5]]]).astype(np.float32)
    name = "draw_boxes_2"
    input_dict = {"images": images, "boxes": boxes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    images = np.random.rand(1, 50, 50, 1).astype(np.float16)
    boxes = np.array([[[0.0, 0.0, 1.0, 1.0]]]).astype(np.float32)
    name = "draw_boxes_3"
    input_dict = {"images": images, "boxes": boxes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    images = np.ones((1, 100, 100, 3), dtype=np.float32)
    boxes = np.array([[[0.0, 0.0, 0.5, 0.5]]]).astype(np.float32)
    name = "draw_boxes_4"
    input_dict = {"images": images, "boxes": boxes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    images = np.random.rand(1, 300, 300, 4).astype(np.float32)
    boxes = np.array([[[0.1, 0.1, 0.3, 0.3], 
                       [0.4, 0.4, 0.6, 0.6],
                       [0.7, 0.7, 0.9, 0.9]]]).astype(np.float32)
    name = "draw_boxes_5"
    input_dict = {"images": images, "boxes": boxes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    images = np.random.rand(4, 128, 128, 3).astype(np.float32)
    boxes = np.array([[[0.25, 0.25, 0.75, 0.75]],
                      [[0.1, 0.1, 0.5, 0.5]],
                      [[0.3, 0.3, 0.8, 0.8]],
                      [[0.2, 0.2, 0.6, 0.6]]]).astype(np.float32)
    name = "draw_boxes_6"
    input_dict = {"images": images, "boxes": boxes, "name": name}
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    images = np.zeros((3, 256, 256, 3), dtype=np.float32)
    boxes = np.array([[[0.0, 0.0, 0.3, 0.3]],
                      [[0.4, 0.4, 0.8, 0.8]],
                      [[0.1

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.DrawBoundingBoxes' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DrawBoundingBoxes'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.raw_ops.DrawBoundingBoxes', generated_inputs['tf.raw_ops.DrawBoundingBoxes'], lib="tf", suffix=0)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

tf.config.experimental.enable_op_determinism()
np.random.seed(42)
tf.random.set_seed(42)

def tf_image_transpose_inputs():
    list_of_inputs = []

    # Input 1: 3D float32 with negatives
    image = np.array(
        [[[-1.0, 0.5], [2.5, -3.2], [4.1, 5.5]],
         [[7.0, -8.0], [9.2, 10.3], [-11.4, 12.6]]],
        dtype=np.float32
    )
    list_of_inputs.append(copy.deepcopy({"image": image, "name": "transpose_case_1"}))

    # Input 2: 4D uint8 RGB image batch
    image = np.random.randint(0, 256, size=(2, 4, 3, 3)).astype(np.uint8)
    list_of_inputs.append(copy.deepcopy({"image": image, "name": "transpose_case_2"}))

    # Input 3: 3D int32 non-square dims
    image = (np.arange(5*2*4).reshape(5, 2, 4).astype(np.int32) - 10)
    list_of_inputs.append(copy.deepcopy({"image": image, "name": "transpose_case_3"}))

    # Input 4: 4D float64 with batch=1, height=1
    image = np.linspace(-1, 1, 1*1*5*2).reshape(1, 1, 5, 2).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"image": image, "name": "transpose_case_4"}))

    # Input 5: 3D int32 with larger values
    image = np.array(
        [[[1000, 2000], [3000, 4000]],
         [[5000, 6000], [7000, 8000]],
         [[9000, 10000], [11000, 12000]]],
        dtype=np.int32
    )
    list_of_inputs.append(copy.deepcopy({"image": image, "name": "transpose_case_5"}))

    # Input 6: 4D int32 with negatives
    image = (np.random.randint(-1000, 1000, size=(4, 3, 2, 3))).astype(np.int32)
    list_of_inputs.append(copy.deepcopy({"image": image, "name": "transpose_case_6"}))

    # Input 7: 3D complex64
    real = (np.arange(3*3*2) / 10.0).reshape(3, 3, 2).astype(np.float32)
    imag = (-np.arange(3*3*2) / 10.0).reshape(3, 3, 2).astype(np.float32)
    image = (real + 1j * imag).astype(np.complex64)
    list_of_inputs.append(copy.deepcopy({"image": image, "name": "transpose_case_7"}))

    # Input 8: 4D float32 single channel
    image = np.random.randn(3, 7, 5, 1).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"image": image, "name": "transpose_case_8"}))

    # Input 9: 3D int8 grayscale single channel
    image = np.random.randint(-128, 127, size=(2, 3, 1)).astype(np.int8)
    list_of_inputs.append(copy.deepcopy({"image": image, "name": "transpose_case_9"}))

    # Input 10: 4D float32 with larger batch and channels
    image = (2.0 * np.random.rand(5, 2, 2, 4) - 1.0).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"image": image, "name": "transpose_case_10"}))

    # Input 11: 3D float16 RGB
    image = np.random.rand(6, 6, 3).astype(np.float16)
    list_of_inputs.append(copy.deepcopy({"image": image, "name": "transpose_case_11"}))

    # Input 12: 4D bool batch
    image = (np.random.rand(2, 2, 2, 2) > 0.5)
    list_of_inputs.append(copy.deepcopy({"image": image, "name": "transpose_case_12"}))

    return list_of_inputs

generated_inputs["tf.image.transpose"] = tf_image_transpose_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.transpose' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.transpose'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.transpose', generated_inputs['tf.image.transpose'], lib="tf", suffix=0)

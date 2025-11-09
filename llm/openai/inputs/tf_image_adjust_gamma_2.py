
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import numpy as np
import copy

def tf_image_adjust_gamma_2_inputs():
    list_of_inputs = []

    # Input 1: uint8 grayscale 2D
    rng = np.random.RandomState(0)
    image = rng.randint(0, 256, size=(4, 5), dtype=np.uint8)
    gamma = np.array(0.5, dtype=np.float32)
    gain = 1.0
    list_of_inputs.append(copy.deepcopy({"image": image, "gamma": gamma, "gain": gain}))

    # Input 2: float32 HWC
    rng = np.random.RandomState(1)
    image = rng.rand(3, 4, 3).astype(np.float32)
    gamma = np.array(1.2, dtype=np.float32)
    gain = 1.2
    list_of_inputs.append(copy.deepcopy({"image": image, "gamma": gamma, "gain": gain}))

    # Input 3: uint8 4D batch
    rng = np.random.RandomState(2)
    image = rng.randint(0, 256, size=(2, 3, 4, 3), dtype=np.uint8)
    gamma = np.array(2.2, dtype=np.float32)
    gain = 0.8
    list_of_inputs.append(copy.deepcopy({"image": image, "gamma": gamma, "gain": gain}))

    # Input 4: float64 4D
    rng = np.random.RandomState(3)
    image = rng.rand(1, 2, 2, 4).astype(np.float64)
    gamma = np.array(0.8, dtype=np.float32)
    gain = 0.5
    list_of_inputs.append(copy.deepcopy({"image": image, "gamma": gamma, "gain": gain}))

    # Input 5: int32 1D
    image = np.arange(10, dtype=np.int32)
    gamma = np.array(1.0, dtype=np.float32)
    gain = 2.0
    list_of_inputs.append(copy.deepcopy({"image": image, "gamma": gamma, "gain": gain}))

    # Input 6: float16 5D
    rng = np.random.RandomState(4)
    image = (rng.rand(2, 1, 2, 3, 1) * 2.0).astype(np.float16)
    gamma = np.array(0.9, dtype=np.float32)
    gain = 1.0
    list_of_inputs.append(copy.deepcopy({"image": image, "gamma": gamma, "gain": gain}))

    # Input 7: uint16 single-channel HWC
    rng = np.random.RandomState(5)
    image = rng.randint(0, 1024, size=(4, 4, 1), dtype=np.uint16)
    gamma = np.array(0.3, dtype=np.float32)
    gain = 3.0
    list_of_inputs.append(copy.deepcopy({"image": image, "gamma": gamma, "gain": gain}))

    # Input 8: float32 HWC with values > 1
    rng = np.random.RandomState(6)
    image = (rng.rand(2, 2, 3) * 5.0).astype(np.float32)
    gamma = np.array(0.75, dtype=np.float32)
    gain = 1.0
    list_of_inputs.append(copy.deepcopy({"image": image, "gamma": gamma, "gain": gain}))

    # Input 9: float32 2D with zeros and gamma 0
    rng = np.random.RandomState(7)
    image = rng.rand(3, 3).astype(np.float32)
    image[rng.rand(3, 3) > 0.7] = 0.0
    gamma = np.array(0.0, dtype=np.float32)
    gain = 1.5
    list_of_inputs.append(copy.deepcopy({"image": image, "gamma": gamma, "gain": gain}))

    # Input 10: float32 HWC single-channel
    rng = np.random.RandomState(8)
    image = rng.rand(2, 3, 1).astype(np.float32)
    gamma = np.array(2.0, dtype=np.float32)
    gain = 0.75
    list_of_inputs.append(copy.deepcopy({"image": image, "gamma": gamma, "gain": gain}))

    # Input 11: bool 2D
    rng = np.random.RandomState(9)
    image = rng.rand(5, 5) > 0.5
    gamma = np.array(1.5, dtype=np.float32)
    gain = 0.7
    list_of_inputs.append(copy.deepcopy({"image": image, "gamma": gamma, "gain": gain}))

    return list_of_inputs

generated_inputs["tf.image.adjust_gamma_2"] = tf_image_adjust_gamma_2_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.image.adjust_gamma_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.image.adjust_gamma_2'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.image.adjust_gamma', generated_inputs['tf.image.adjust_gamma_2'], lib="tf", suffix=2)

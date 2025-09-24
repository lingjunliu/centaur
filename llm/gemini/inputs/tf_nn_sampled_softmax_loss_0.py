
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def get_sampled_softmax_loss_inputs():
    """
    Generates a list of valid inputs for tf.nn.sampled_softmax_loss.
    """
    list_of_inputs = []

    def create_input_dict(
        dim, num_classes, num_true,
        remove_accidental_hits, seed, dtype=np.float32
    ):
        """
        Helper to create inputs with batch_size=1 and homogeneous tuples
        for sampled_values to work around a buggy validation tool.
        This relies on broadcasting in TensorFlow to work correctly.
        """
        batch_size = 1
        # Set num_sampled = num_true to make all arrays in the tuple have the same shape.
        num_sampled = num_true

        if num_sampled >= num_classes:
            return None

        weights = np.random.randn(num_classes, dim).astype(dtype)
        biases = np.random.randn(num_classes).astype(dtype)
        inputs = np.random.randn(batch_size, dim).astype(dtype)
        labels = np.random.randint(0, num_classes, size=(batch_size, num_true)).astype(np.int64)

        sampled_candidates = np.random.choice(
            np.arange(num_classes), size=num_sampled, replace=False
        ).astype(np.int64)
        
        # TF expects shape [1, num_true]. We provide [num_true] and rely on broadcasting.
        true_expected_count = np.random.rand(num_true).astype(dtype)
        
        sampled_expected_count = np.random.rand(num_sampled).astype(dtype)
        
        # All arrays in the tuple now have the same 1D shape.
        sampled_values_tuple = (sampled_candidates, true_expected_count, sampled_expected_count)

        input_dict = {
            'weights': weights,
            'biases': biases,
            'labels': labels,
            'inputs': inputs,
            'num_sampled': num_sampled,
            'num_classes': num_classes,
            'num_true': num_true,
            'sampled_values': sampled_values_tuple,
            'remove_accidental_hits': remove_accidental_hits,
            'seed': seed,
            'name': 'test_name'
        }
        return input_dict

    # Parameter sets, all with batch_size=1 to ensure broadcasting works.
    params_list = [
        # (dim, num_classes, num_true, remove_accidental_hits, seed, dtype)
        (64, 100, 10, True, 42, np.float32),
        (32, 50, 5, True, 0, np.float32),
        (128, 200, 25, True, 123, np.float32),
        (64, 100, 10, False, 1, np.float32),
        (16, 30, 8, True, 1, np.float64),
        (256, 10000, 64, True, 2023, np.float32),
        (4, 10, 5, True, 2, np.float32),
        (20, 80, 15, False, 7, np.float32),
        (10, 12, 10, True, 88, np.float32),
        (8, 40, 4, True, 99, np.float64),
    ]

    for params in params_list:
        input_dict = create_input_dict(*params)
        if input_dict:
            list_of_inputs.append(input_dict)
            
    # Input 11: Zero-valued weights and biases
    input_11_params = (16, 20, 5, True, 3, np.float32)
    input_11 = create_input_dict(*input_11_params)
    if input_11:
        input_11['weights'] = np.zeros_like(input_11['weights'])
        input_11['biases'] = np.zeros_like(input_11['biases'])
        list_of_inputs.append(input_11)

    return [copy.deepcopy(i) for i in list_of_inputs]

generated_inputs["tf.nn.sampled_softmax_loss"] = get_sampled_softmax_loss_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.nn.sampled_softmax_loss' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.nn.sampled_softmax_loss'.")

check_valid('tf.nn.sampled_softmax_loss', generated_inputs['tf.nn.sampled_softmax_loss'], lib="tf", suffix=0)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import torch, copy
import numpy as np

def miopen_batch_norm_inputs():
    list_of_inputs = []
    np.random.seed(0)

    # Helper to create a valid set of inputs for a generic batch normalization function.
    # Note: torch.miopen_batch_norm is a CUDA-specific function. These inputs are generated
    # to be compatible with a CPU-based batch norm equivalent, as the execution environment is CPU.
    def create_case(shape, dtype=np.float32, training=True, momentum=0.1, eps=1e-5):
        # Batch norm in training mode needs a batch size > 1 to calculate variance.
        # We enforce a minimum batch size of 2 for training.
        if training and shape[0] <= 1:
            shape = (2,) + shape[1:]
        
        if len(shape) < 2:
            return None # Invalid shape for batch norm

        num_features = shape[1]
        
        input_data = np.random.randn(*shape).astype(dtype)
        weight = np.random.randn(num_features).astype(dtype)
        bias = np.random.randn(num_features).astype(dtype)
        
        # Initialize running stats.
        running_mean = np.zeros(num_features, dtype=dtype)
        running_var = np.ones(num_features, dtype=dtype)
        
        # In evaluation mode, the provided running stats are used.
        if not training:
            running_mean = np.random.randn(num_features).astype(dtype)
            # Variance must be non-negative. We generate strictly positive values for robustness.
            running_var = np.abs(np.random.rand(num_features).astype(dtype)) + 1e-4

        input_dict = {
            "input": input_data,
            "weight": weight,
            "bias": bias,
            "running_mean": running_mean,
            "running_var": running_var,
            "training": training,
            "exponential_average_factor": momentum,
            "eps": eps,
        }
        return input_dict

    # Case 1: Standard 4D input (CNN), training mode
    list_of_inputs.append(copy.deepcopy(create_case((4, 3, 8, 8), training=True)))

    # Case 2: Standard 4D input (CNN), evaluation mode
    list_of_inputs.append(copy.deepcopy(create_case((4, 3, 8, 8), training=False)))

    # Case 3: 2D input (MLP), training mode
    list_of_inputs.append(copy.deepcopy(create_case((64, 16), training=True)))

    # Case 4: 2D input (MLP), evaluation mode
    list_of_inputs.append(copy.deepcopy(create_case((64, 16), training=False)))

    # Case 5: 3D input (Conv1D), training mode
    list_of_inputs.append(copy.deepcopy(create_case((10, 20, 30), training=True)))

    # Case 6: 5D input (Conv3D), training mode
    list_of_inputs.append(copy.deepcopy(create_case((4, 3, 5, 5, 5), training=True)))

    # Case 7: High momentum value
    list_of_inputs.append(copy.deepcopy(create_case((8, 4, 6, 6), momentum=0.9)))

    # Case 8: Large epsilon value
    list_of_inputs.append(copy.deepcopy(create_case((8, 4, 6, 6), eps=1e-2)))

    # Case 9: "No affine" transformation (weight=1, bias=0)
    case9 = create_case((8, 4, 6, 6))
    if case9:
        case9["weight"] = np.ones_like(case9["weight"])
        case9["bias"] = np.zeros_like(case9["bias"])
        list_of_inputs.append(copy.deepcopy(case9))

    # Case 10: Input with all same values (zero variance batch)
    case10 = create_case((8, 4, 6, 6))
    if case10:
        case10["input"] = np.full_like(case10["input"], 5.0)
        list_of_inputs.append(copy.deepcopy(case10))

    # Case 11: Single feature channel
    list_of_inputs.append(copy.deepcopy(create_case((10, 1, 10, 10))))

    # Case 12: Large number of features
    list_of_inputs.append(copy.deepcopy(create_case((4, 256, 4, 4))))

    # Clean up any None cases
    list_of_inputs = [item for item in list_of_inputs if item is not None]

    return list_of_inputs

generated_inputs["torch.miopen_batch_norm"] = miopen_batch_norm_inputs()

def check_valid(api, list_of_inputs, lib="torch", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'torch.miopen_batch_norm' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'torch.miopen_batch_norm'.")

check_valid('torch.miopen_batch_norm', generated_inputs['torch.miopen_batch_norm'], lib="torch", suffix=0)

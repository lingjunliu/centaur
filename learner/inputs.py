import torch
import copy
import time
from utils.api_utils import get_signatures, get_driver
from utils.misc import get_dir_in_root, get_tmp_dir, create_subdir, save_to_new_pkl, read_pkl
from generator.input_generators import get_random_input, get_abstract_input, concretize_input
from eval.oracle import oracle_crash
from llm.valid_inputs import *
import numpy as np
import os

def scatter_inputs():
    list_of_inputs = []
    # Input 1, valid
    src_torch = torch.arange(1, 11).reshape((2, 5))
    src = src_torch.numpy() 
    index = torch.tensor([[0, 1, 2, 0]]).numpy()
    input = torch.zeros(3, 5, dtype=src_torch.dtype).numpy()
    dim = 0

    input_dict = {
        "input": input,
        "dim": dim,
        "src": src, 
        "index": index,         
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, invalid
    index_2 = torch.tensor([[0, 1, 2], [0, 1, 4]])
    input_dict["index"] = index_2.numpy()
    
    # Skipping invalid inputs
    # list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid
    input_dict["dim"] = 1
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, invalid
    input_dict = {
        "input": torch.full((2, 4), 2.).numpy(),
        "dim": 1,
        "src": 1.23, 
        "index": torch.tensor([[2], [3]]).numpy(),         
    }
    
    # Skipping invalid inputs
    # list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, valid
    input_dict = {
        "input": torch.full((2, 4), 2., dtype=int).numpy(),
        "dim": 1,
        "src": torch.tensor([[2], [3]]).numpy(), 
        "index": torch.tensor([[2], [3]]).numpy(),         
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, valid
    input_dict = {
        "input": torch.full((2, 4), 2., dtype=torch.float32).numpy(),
        "dim": 1,
        "src": torch.tensor([[2], [3]], dtype=torch.float32).numpy(), 
        "index": torch.tensor([[2], [3]]).numpy(),         
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

def conv_transpose2d_inputs():
    list_of_inputs = []
    # Input 1, valid
    input = torch.randn(1, 3, 5, 5).numpy()        # [N, C_in, H_in, W_in]
    weight = torch.randn(3, 2, 3, 3).numpy()       # [C_in, C_out, kH, kW]
    stride = 1
    padding = 0
    
    input_dict = {
        "input": input,
        "weight": weight,
        "stride": stride,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, valid
    input = torch.randn(1, 4, 4, 4).numpy()        # [N, C_in, H_in, W_in]
    weight = torch.randn(4, 2, 3, 3).numpy()       # [C_in, C_out, kH, kW]
    stride = 2
    padding = 1
    
    input_dict = {
        "input": input,
        "weight": weight,
        "stride": stride,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid
    input = torch.randn(1, 2, 6, 6).numpy()        # [N, C_in, H_in, W_in]
    weight = torch.randn(2, 1, 5, 5).numpy()       # [C_in, C_out, kH, kW]
    stride = 1
    padding = 0
    
    input_dict = {
        "input": input,
        "weight": weight,
        "stride": stride,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, valid
    input = torch.randn(1, 1, 5, 5).numpy()        # [N, C_in, H_in, W_in]
    weight = torch.randn(1, 1, 3, 3).numpy()       # [C_in, C_out, kH, kW]
    stride = 2
    padding = 1
    
    input_dict = {
        "input": input,
        "weight": weight,
        "stride": stride,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, valid
    input = torch.randn(2, 8, 10, 10).numpy()        # [N, C_in, H_in, W_in]
    weight = torch.randn(8, 16, 4, 4).numpy()       # [C_in, C_out, kH, kW]
    stride = 2
    padding = 1
    
    input_dict = {
        "input": input,
        "weight": weight,
        "stride": stride,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 6, invalid
    input = torch.randn(2, 8, 10, 10, dtype=torch.complex64).numpy()        # [N, C_in, H_in, W_in]
    weight = torch.randn(8, 16, 4, 4).numpy()       # [C_in, C_out, kH, kW]
    stride = 2
    padding = 1
    
    input_dict = {
        "input": input,
        "weight": weight,
        "stride": stride,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 7, invalid
    input = torch.randn(2, 8, 10, 10).numpy()        # [N, C_in, H_in, W_in]
    weight = torch.randint(low=0, high=100, size=(8, 16, 4, 4), dtype=torch.int64).numpy()       # [C_in, C_out, kH, kW]
    stride = 2
    padding = 1
    
    input_dict = {
        "input": input,
        "weight": weight,
        "stride": stride,
        "padding": padding
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

def matmul_inputs():
    list_of_inputs = []
    # Input 1, valid
    input = torch.randn(3, 5).numpy()
    other = torch.randn(5, 2).numpy() 

    input_dict = {
        "input": input,
        "other": other
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.randn(3, 5).numpy()
    other = torch.randn(5).numpy()

    input_dict = {
        "input": input,
        "other": other
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

def add_inputs():
    list_of_inputs = []
    # Input 1, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    other = torch.tensor([0.1, 0.2, 0.3]).numpy()
    alpha = 10.0

    input_dict = {
        "input": input,
        "other": other,
        "alpha": alpha
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([1, 2, 3], dtype=torch.int32).numpy() 
    other = torch.tensor([4, 5, 6], dtype=torch.int32).numpy()
    alpha = 2

    input_dict = {
        "input": input,
        "other": other,
        "alpha": alpha
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

def combinations_inputs():
    list_of_inputs = []
    # Input 1, valid
    input = torch.tensor([10, 20, 30, 40]).numpy()
    r = 3
    with_replacement = False

    input_dict = {
        "input": input,
        "r": r,
        "with_replacement": with_replacement
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.tensor([1.5, 2.5]).numpy()
    r = 2
    with_replacement = True

    input_dict = {
        "input": input,
        "r": r,
        "with_replacement": with_replacement
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

def addcmul_inputs():
    list_of_inputs = []
    # Input 1, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()
    tensor1 = torch.tensor([0.1, 0.2, 0.3]).numpy()
    tensor2 = torch.tensor([10.0, 20.0, 30.0]).numpy()
    value = 2.0  

    input_dict = {
        "input": input,
        "tensor1": tensor1,
        "tensor2": tensor2,
        "value": value
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.ones((2, 3)).numpy()
    tensor1 = torch.tensor([[1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0]]).numpy()
    tensor2 = torch.tensor([[0.1, 0.2, 0.3],
                            [0.4, 0.5, 0.6]]).numpy()
    value = 0.5

    input_dict = {
        "input": input,
        "tensor1": tensor1,
        "tensor2": tensor2,
        "value": value
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    return list_of_inputs

def lp_pool1d_inputs():
    list_of_inputs = []
    # Input 1, valid
    input = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0]]]).numpy()
    norm_type = 2.0
    kernel_size = 2
    stride = 2
    ceil_mode = True
    
    input_dict = {
        "input": input,
        "norm_type": norm_type,
        "kernel_size": kernel_size,
        "stride": stride,
        "ceil_mode": ceil_mode
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, valid
    input = torch.tensor([[[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]]]).numpy()
    norm_type = 1.0
    kernel_size = 3
    stride = 2
    ceil_mode = True
    
    input_dict = {
        "input": input,
        "norm_type": norm_type,
        "kernel_size": kernel_size,
        "stride": stride,
        "ceil_mode": ceil_mode
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

# Add human and LLM defined inputs for APIs that are
# difficult to generate inputs for
inputs_per_api = {
    "scatter": scatter_inputs,      # human start
    "conv_transpose2d": conv_transpose2d_inputs,
    "matmul": matmul_inputs,
    "add": add_inputs,
    "combinations": combinations_inputs,
    "addcmul": addcmul_inputs,
    "lp_pool1d_": lp_pool1d_inputs, # human end
    'full': full_inputs,            # llm start
    'empty_strided': empty_strided_inputs,
    'broadcast_shapes': broadcast_shapes_inputs,
    'rand': rand_inputs,
    'zeros': zeros_inputs,
    'ones': ones_inputs,
    'max_pool1d': max_pool1d_inputs,
    'mean': mean_inputs,
    'rot90': rot90_inputs,
    'max_pool3d': max_pool3d_inputs,
    'var': var_inputs,
    'transpose': transpose_inputs,
    'max_pool2d': max_pool2d_inputs,
    'normalize': normalize_inputs,
    'split': split_inputs,
    'var_mean': var_mean_inputs,
    'std': std_inputs,
    'unsqueeze': unsqueeze_inputs,
    'nansum': nansum_inputs,
    'pixel_shuffle': pixel_shuffle_inputs,
    'logcumsumexp': logcumsumexp_inputs,
    'movedim': movedim_inputs,
    'narrow': narrow_inputs,
    'pad': pad_inputs,
    'reshape': reshape_inputs,
    'std_mean': std_mean_inputs,
    'linspace': linspace_inputs,
    'logspace': logspace_inputs,
    'lstsq': lstsq_inputs,
    'trapz': trapz_inputs,
    'triangular_solve': triangular_solve_inputs,
    'tensordot': tensordot_inputs,
    'multilabel_soft_margin_loss': multilabel_soft_margin_loss_inputs,
    'linear_': linear_inputs,
    'lu_solve': lu_solve_inputs,
    'margin_ranking_loss': margin_ranking_loss_inputs,
    'where': where_inputs,
    'pow': pow_inputs,
    'mm': mm_inputs,
    'mv': mv_inputs,
    'polar': polar_inputs,
    'prelu': prelu_inputs,
    'solve': solve_inputs,
    'sub': sub_inputs,
    'nll_loss': nll_loss_inputs,
    'flatten_': flatten_inputs,
    'interpolate': interpolate_inputs,
    'is_nonzero': is_nonzero_inputs,
    'layer_norm': layer_norm_inputs,
    'scatter_add': scatter_add_inputs,
    'chunk': chunk_inputs,
    'isclose': isclose_inputs,
    'heaviside': heaviside_inputs,
    'gt': gt_inputs,
    'hypot': hypot_inputs,
    'igamma': igamma_inputs,
    'inner': inner_inputs,
    'lcm': lcm_inputs,
    'le': le_inputs,
    'einsum': einsum_inputs,
    'index_select': index_select_inputs,
    'as_strided': as_strided_inputs,
    'broadcast_to': broadcast_to_inputs,
    'cat': cat_inputs,
    'flip': flip_inputs,
    'Flatten': flatten_inputs_2,
    'FractionalMaxPool2d': fractional_max_pool2d_inputs,
    'adaptive_avg_pool2d': adaptive_avg_pool2d_inputs,
    'amax': amax_inputs,
    'atan2': atan2_inputs,
    'cosine_similarity': cosine_similarity_inputs,
    'cross': cross_inputs,
    'cross_entropy': cross_entropy_inputs,
    'dist': dist_inputs,
    'floor_divide': floor_divide_inputs,
    'fmin': fmin_inputs,
    'ger': ger_inputs,
    'GroupNorm': groupnorm_inputs,
    'MaxPool3d': maxpool3d_inputs,
    'PixelShuffle': pixel_shuffle_inputs,
    'adaptive_max_pool2d': adaptive_max_pool2d_inputs,
    'alpha_dropout': alpha_dropout_inputs,
    'bitwise_and': bitwise_and_inputs,
    'LayerNorm': layer_norm_inputs,
    'Linear': linear_inputs,
    'MaxPool2d': maxpool2d_inputs,
    'PReLU_': prelu_inputs,
    'Softmax': softmax_inputs,
    'Softmin': softmin_inputs,
    'bincount': bincount_inputs,
    'bitwise_or': bitwise_or_inputs,
    'bitwise_xor': bitwise_xor_inputs,
    'bmm': bmm_inputs,
    'cdist': cdist_inputs,
    'complex': complex_inputs,
    'copysign': copysign_inputs,
    'dot': dot_inputs,
    'eq': eq_inputs,
    'float_power': float_power_inputs,
    'ge': ge_inputs,
    'allclose': allclose_inputs,
    'L1Loss': l1loss_inputs,
    'MSELoss': MSELoss_inputs,
    'MultiMarginLoss': multimarginloss_inputs,
    'PairwiseDistance': pairwise_distance_inputs,
    'PoissonNLLLoss': poisson_nll_loss_inputs,
    'addmm': addmm_inputs,
    'embedding_bag': embedding_bag_inputs,
    'binaryCrossEntropyWithLogits': binary_cross_entropy_with_logits_inputs,
    'addcdiv': addcdiv_inputs,
    'addmv': addmv_inputs,
    'addr': addr_inputs,
    'MultiLabelSoftMarginLoss': multilabel_soft_margin_loss_inputs,
    'NLLLoss': nllloss_inputs,
    'batch_norm': batch_norm_inputs,
    'LSTMCell': lstm_cell_inputs    # llm end
}

def get_inputs(api, lib="torch", time_budget=30, min_val_inp=5, seed=42):
    # Return human written inputs if available
    if api in inputs_per_api:
        return inputs_per_api[api]()
    
    # Generate valid inputs through random generation otherwise
    api_signature = get_signatures()[api]
    input_file = os.path.join(get_dir_in_root("valid_inputs"), f"{api}.pkl")
    
    list_of_inputs = []
    
    # If there already is a saved file, read from that and concretize
    if os.path.isfile(input_file):
        abstract_inputs = read_pkl(input_file)
        for abs_inp, saved_seed in abstract_inputs:
            rng = np.random.default_rng(saved_seed)
            list_of_inputs.append(concretize_input(abs_inp, api_signature, rng))
    else:   # Generate and save otherwise
        api_driver = get_driver(api, lib=lib)
        valid = 0
        invalid = 0
        abstract_inputs = []
        
        start_time = time.time()
        while (time.time() - start_time < time_budget) and (valid < min_val_inp):
            rng = np.random.default_rng(seed)
            input_dict = get_random_input(api_signature, rng)
            status, exception_message = oracle_crash(api_driver, input_dict, cpu=True)
            if status == "invalid":
                invalid += 1
            else:
                valid += 1
                # Only adding valid inputs
                list_of_inputs.append(input_dict)
                abs_inp = get_abstract_input(input_dict, api_signature)
                # Save the abstract input along with the seed
                abstract_inputs.append((abs_inp, seed))
            
            seed += 1
        
        # Save abstract inputs to file
        save_to_new_pkl(input_file, abstract_inputs)
            
        # Save some stats
        infer_dir = create_subdir(get_tmp_dir(), "infer_results")
        csv_file = os.path.join(infer_dir, f"{api}_{time_budget}.csv")
        with open(csv_file, "w") as f:
            f.write(f"{api},{valid},{invalid},{round(valid*100/(valid+invalid), 4) if (valid+invalid) > 0 else 0}\n")    
    return list_of_inputs

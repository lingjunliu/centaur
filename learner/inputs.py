import torch
import copy
import time
from utils.api_utils import get_signatures, get_driver
from utils.misc import get_dir_in_root, get_tmp_dir, create_subdir, save_to_new_pkl, read_pkl
from generator.input_generators import get_random_input, get_abstract_input, concretize_input
from eval.oracle import oracle_crash
import llm.valid_inputs as valid_inputs
import llm.valid_inputs_old as valid_inputs_old
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
    'full': valid_inputs.full_inputs + valid_inputs_old.full_inputs,    # llm start
    'empty_strided': valid_inputs.empty_strided_inputs + valid_inputs_old.empty_strided_inputs,
    'broadcast_shapes': valid_inputs.broadcast_shapes_inputs + valid_inputs_old.broadcast_shapes_inputs,
    'rand': valid_inputs.rand_inputs + valid_inputs_old.rand_inputs,
    'zeros': valid_inputs.zeros_inputs + valid_inputs_old.zeros_inputs,
    'ones': valid_inputs.ones_inputs + valid_inputs_old.ones_inputs,
    'max_pool1d': valid_inputs.max_pool1d_inputs + valid_inputs_old.max_pool1d_inputs,
    'mean': valid_inputs.mean_inputs + valid_inputs_old.mean_inputs,
    'rot90': valid_inputs.rot90_inputs + valid_inputs_old.rot90_inputs,
    'max_pool3d': valid_inputs.max_pool3d_inputs + valid_inputs_old.max_pool3d_inputs,
    'var': valid_inputs.var_inputs + valid_inputs_old.var_inputs,
    'transpose': valid_inputs.transpose_inputs + valid_inputs_old.transpose_inputs,
    'max_pool2d': valid_inputs.max_pool2d_inputs + valid_inputs_old.max_pool2d_inputs,
    'normalize': valid_inputs.normalize_inputs + valid_inputs_old.normalize_inputs,
    'split': valid_inputs.split_inputs + valid_inputs_old.split_inputs,
    'var_mean': valid_inputs.var_mean_inputs + valid_inputs_old.var_mean_inputs,
    'std': valid_inputs.std_inputs + valid_inputs_old.std_inputs,
    'unsqueeze': valid_inputs.unsqueeze_inputs + valid_inputs_old.unsqueeze_inputs,
    'nansum': valid_inputs.nansum_inputs + valid_inputs_old.nansum_inputs,
    'pixel_shuffle': valid_inputs.pixel_shuffle_inputs + valid_inputs_old.pixel_shuffle_inputs,
    'logcumsumexp': valid_inputs.logcumsumexp_inputs + valid_inputs_old.logcumsumexp_inputs,
    'movedim': valid_inputs.movedim_inputs + valid_inputs_old.movedim_inputs,
    'narrow': valid_inputs.narrow_inputs + valid_inputs_old.narrow_inputs,
    # 'pad': pad_inputs,
    'pad': valid_inputs_old.pad_inputs,
    'reshape': valid_inputs.reshape_inputs + valid_inputs_old.reshape_inputs,
    'std_mean': valid_inputs.std_mean_inputs + valid_inputs_old.std_mean_inputs,
    'linspace': valid_inputs.linspace_inputs + valid_inputs_old.linspace_inputs,
    'logspace': valid_inputs.logspace_inputs + valid_inputs_old.logspace_inputs,
    'lstsq': valid_inputs.lstsq_inputs + valid_inputs_old.lstsq_inputs,
    'trapz': valid_inputs.trapz_inputs + valid_inputs_old.trapz_inputs,
    'triangular_solve': valid_inputs.triangular_solve_inputs + valid_inputs_old.triangular_solve_inputs,
    'tensordot': valid_inputs.tensordot_inputs + valid_inputs_old.tensordot_inputs,
    'multilabel_soft_margin_loss': valid_inputs.multilabel_soft_margin_loss_inputs + valid_inputs_old.multilabel_soft_margin_loss_inputs,
    'linear_': valid_inputs.linear_inputs + valid_inputs_old.linear_inputs,
    'lu_solve': valid_inputs.lu_solve_inputs + valid_inputs_old.lu_solve_inputs,
    'margin_ranking_loss': valid_inputs.margin_ranking_loss_inputs + valid_inputs_old.margin_ranking_loss_inputs,
    'where': valid_inputs.where_inputs + valid_inputs_old.where_inputs,
    'pow': valid_inputs.pow_inputs + valid_inputs_old.pow_inputs,
    'mm': valid_inputs.mm_inputs + valid_inputs_old.mm_inputs,
    'mv': valid_inputs.mv_inputs + valid_inputs_old.mv_inputs,
    'polar': valid_inputs.polar_inputs + valid_inputs_old.polar_inputs,
    'prelu': valid_inputs.prelu_inputs + valid_inputs_old.prelu_inputs,
    'solve': valid_inputs.solve_inputs + valid_inputs_old.solve_inputs,
    'sub': valid_inputs.sub_inputs + valid_inputs_old.sub_inputs,
    'nll_loss': valid_inputs.nll_loss_inputs + valid_inputs_old.nll_loss_inputs,
    'flatten_': valid_inputs.flatten_inputs + valid_inputs_old.flatten_inputs,
    'interpolate': valid_inputs.interpolate_inputs + valid_inputs_old.interpolate_inputs,
    'is_nonzero': valid_inputs.is_nonzero_inputs + valid_inputs_old.is_nonzero_inputs,
    'layer_norm': valid_inputs.layer_norm_inputs + valid_inputs_old.layer_norm_inputs,
    # 'scatter_add': scatter_add_inputs,
    'scatter_add': valid_inputs_old.scatter_add_inputs,
    'chunk': valid_inputs.chunk_inputs + valid_inputs_old.chunk_inputs,
    'isclose': valid_inputs.isclose_inputs + valid_inputs_old.isclose_inputs,
    'heaviside': valid_inputs.heaviside_inputs + valid_inputs_old.heaviside_inputs,
    'gt': valid_inputs.gt_inputs + valid_inputs_old.gt_inputs,
    'hypot': valid_inputs.hypot_inputs + valid_inputs_old.hypot_inputs,
    'igamma': valid_inputs.igamma_inputs + valid_inputs_old.igamma_inputs,
    'inner': valid_inputs.inner_inputs + valid_inputs_old.inner_inputs,
    'lcm': valid_inputs.lcm_inputs + valid_inputs_old.lcm_inputs,
    'le': valid_inputs.le_inputs + valid_inputs_old.le_inputs,
    'einsum': valid_inputs.einsum_inputs + valid_inputs_old.einsum_inputs,
    'index_select': valid_inputs.index_select_inputs + valid_inputs_old.index_select_inputs,
    'as_strided': valid_inputs.as_strided_inputs + valid_inputs_old.as_strided_inputs,
    'broadcast_to': valid_inputs.broadcast_to_inputs + valid_inputs_old.broadcast_to_inputs,
    'cat': valid_inputs.cat_inputs + valid_inputs_old.cat_inputs,
    'flip': valid_inputs.flip_inputs + valid_inputs_old.flip_inputs,
    'Flatten': valid_inputs.flatten_inputs_2 + valid_inputs_old.flatten_inputs_2,
    'FractionalMaxPool2d': valid_inputs.fractional_max_pool2d_inputs + valid_inputs_old.fractional_max_pool2d_inputs,
    'adaptive_avg_pool2d': valid_inputs.adaptive_avg_pool2d_inputs + valid_inputs_old.adaptive_avg_pool2d_inputs,
    'amax': valid_inputs.amax_inputs + valid_inputs_old.amax_inputs,
    'atan2': valid_inputs.atan2_inputs + valid_inputs_old.atan2_inputs,
    'cosine_similarity': valid_inputs.cosine_similarity_inputs + valid_inputs_old.cosine_similarity_inputs,
    'cross': valid_inputs.cross_inputs + valid_inputs_old.cross_inputs,
    'cross_entropy': valid_inputs.cross_entropy_inputs + valid_inputs_old.cross_entropy_inputs,
    'dist': valid_inputs.dist_inputs + valid_inputs_old.dist_inputs,
    'floor_divide': valid_inputs.floor_divide_inputs + valid_inputs_old.floor_divide_inputs,
    'fmin': valid_inputs.fmin_inputs + valid_inputs_old.fmin_inputs,
    'ger': valid_inputs.ger_inputs + valid_inputs_old.ger_inputs,
    'GroupNorm': valid_inputs.groupnorm_inputs + valid_inputs_old.groupnorm_inputs,
    # 'MaxPool3d': maxpool3d_inputs,
    'MaxPool3d': valid_inputs_old.maxpool3d_inputs,
    'PixelShuffle': valid_inputs.pixel_shuffle_inputs + valid_inputs_old.pixel_shuffle_inputs,
    'adaptive_max_pool2d': valid_inputs.adaptive_max_pool2d_inputs + valid_inputs_old.adaptive_max_pool2d_inputs,
    'alpha_dropout': valid_inputs.alpha_dropout_inputs + valid_inputs_old.alpha_dropout_inputs,
    'bitwise_and': valid_inputs.bitwise_and_inputs + valid_inputs_old.bitwise_and_inputs,
    'LayerNorm': valid_inputs.layer_norm_inputs + valid_inputs_old.layer_norm_inputs,
    'Linear': valid_inputs.linear_inputs + valid_inputs_old.linear_inputs,
    # 'MaxPool2d': maxpool2d_inputs,
    'MaxPool2d': valid_inputs_old.maxpool2d_inputs,
    'PReLU_': valid_inputs.prelu_inputs + valid_inputs_old.prelu_inputs,
    'Softmax': valid_inputs.softmax_inputs + valid_inputs_old.softmax_inputs,
    'Softmin': valid_inputs.softmin_inputs + valid_inputs_old.softmin_inputs,
    'bincount': valid_inputs.bincount_inputs + valid_inputs_old.bincount_inputs,
    'bitwise_or': valid_inputs.bitwise_or_inputs + valid_inputs_old.bitwise_or_inputs,
    'bitwise_xor': valid_inputs.bitwise_xor_inputs + valid_inputs_old.bitwise_xor_inputs,
    'bmm': valid_inputs.bmm_inputs + valid_inputs_old.bmm_inputs,
    'cdist': valid_inputs.cdist_inputs + valid_inputs_old.cdist_inputs,
    'complex': valid_inputs.complex_inputs + valid_inputs_old.complex_inputs,
    'copysign': valid_inputs.copysign_inputs + valid_inputs_old.copysign_inputs,
    'dot': valid_inputs.dot_inputs + valid_inputs_old.dot_inputs,
    'eq': valid_inputs.eq_inputs + valid_inputs_old.eq_inputs,
    'float_power': valid_inputs.float_power_inputs + valid_inputs_old.float_power_inputs,
    'ge': valid_inputs.ge_inputs + valid_inputs_old.ge_inputs,
    'allclose': valid_inputs.allclose_inputs + valid_inputs_old.allclose_inputs,
    'L1Loss': valid_inputs.l1loss_inputs + valid_inputs_old.l1loss_inputs,
    'MSELoss': valid_inputs.MSELoss_inputs + valid_inputs_old.MSELoss_inputs,
    'MultiMarginLoss': valid_inputs.multimarginloss_inputs + valid_inputs_old.multimarginloss_inputs,
    'PairwiseDistance': valid_inputs.pairwise_distance_inputs + valid_inputs_old.pairwise_distance_inputs,
    'PoissonNLLLoss': valid_inputs.poisson_nll_loss_inputs + valid_inputs_old.poisson_nll_loss_inputs,
    'addmm': valid_inputs.addmm_inputs + valid_inputs_old.addmm_inputs,
    'embedding_bag': valid_inputs.embedding_bag_inputs + valid_inputs_old.embedding_bag_inputs,
    'binaryCrossEntropyWithLogits': valid_inputs.binary_cross_entropy_with_logits_inputs + valid_inputs_old.binary_cross_entropy_with_logits_inputs,
    'addcdiv': valid_inputs.addcdiv_inputs + valid_inputs_old.addcdiv_inputs,
    'addmv': valid_inputs.addmv_inputs + valid_inputs_old.addmv_inputs,
    'addr': valid_inputs.addr_inputs + valid_inputs_old.addr_inputs,
    'MultiLabelSoftMarginLoss': valid_inputs.multilabel_soft_margin_loss_inputs + valid_inputs_old.multilabel_soft_margin_loss_inputs,
    'NLLLoss': valid_inputs.nll_loss_inputs + valid_inputs_old.nll_loss_inputs,
    'batch_norm': valid_inputs.batch_norm_inputs + valid_inputs_old.batch_norm_inputs,
    # 'LSTMCell': lstm_cell_inputs,
    'LSTMCell': valid_inputs_old.lstm_cell_inputs,
    'MarginRankingLoss': valid_inputs.margin_ranking_loss_inputs + valid_inputs_old.margin_ranking_loss_inputs,# llm end
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

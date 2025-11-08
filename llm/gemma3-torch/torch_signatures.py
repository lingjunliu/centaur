signatures = {}
signatures["torch.addcdiv"] = {
    "args": {
        "input": "tensor",
        "tensor1": "tensor",
        "tensor2": "tensor"
    },
    "kwargs": {
        "value": "float",  # could be integer or float, defaulting to float
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.addmv"] = {
    "args": {
        "input": "tensor",
        "mat": "tensor",
        "vec": "tensor"
    },
    "kwargs": {
        "beta": "float",  # could be integer
        "alpha": "float", # could be integer
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.amin"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"  # could also be tuple of integers
    },
    "kwargs": {
        "keepdim": "boolean",
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.any_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # Should it be tensor or optional tensor? Assuming tensor.
    },
    "inner": {}
}

signatures["torch.any_2"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"  # Or tuple of integers.
    },
    "kwargs": {
        "keepdim": "boolean",
        "out": "tensor"  # Should it be tensor or optional tensor? Assuming tensor.
    },
    "inner": {}
}
signatures["torch.are_deterministic_algorithms_enabled"] = {
    "args": {},
    "kwargs": {},
    "inner": {},
}
signatures["torch.argsort"] = {
    "args": {
        "input": "tensor",
        "dim": "integer",  # could be int or None
        "descending": "boolean"
    },
    "kwargs": {
        "stable": "boolean"
    },
    "inner": {}
}
signatures["torch.asin"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could be None as well, but tensor is more likely
    },
    "inner": {},
}
signatures["torch.bincount"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "weights": "tensor",  # Could be None, but documented as tensor
        "minlength": "integer",
    },
    "inner": {},
}
signatures["torch.bitwise_left_shift"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.cartesian_prod"] = {
    "args": {
        "inputs": "tensor_list" # Could also be a tuple of tensors
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.clip"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "min": "tensor",  # could also be float or integer
        "max": "tensor",  # could also be float or integer
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.complex"] = {
    "args": {
        "real": "tensor",  # could be half, float, or double, but all are tensors
        "imag": "tensor"  # must be the same dtype as real (tensor)
    },
    "kwargs": {
        "out": "tensor"  # depends on dtype of input tensors
    },
    "inner": {}
}
signatures["torch.conj_physical"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.cumprod"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {
        "dtype": "dtype",
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.dequantize_1"] = {
    "args": {
        "tensor": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["torch.dequantize_2"] = {
    "args": {
        "tensors": "tensor_list"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.diag_embed"] = {
    "args": {
        "input": "tensor",
        "offset": "integer" # could also be tensor, but integer is more common
    },
    "kwargs": {
        "dim1": "integer",
        "dim2": "integer"
    },
    "inner": {},
}
signatures["torch.empty_like"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dtype": "dtype", # Could be torch.dtype instead
        "layout": "string", # Could be torch.layout instead
        "device": "string", #device not included
        "requires_grad": "boolean"
    },
    "inner": {},
}
signatures["torch.equal"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.fix"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # Could also be None, but tensor is more specific.
    },
    "inner": {},
}
signatures["torch.flipud"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.float_power"] = {
    "args": {
        "input": "tensor",
        "exponent": "float" # Could also be tensor, but float seems more common
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.floor"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could be None, but tensor is more specific
    },
    "inner": {}
}
signatures["torch.floor_divide"] = {
    "args": {
        "input": "tensor",
        "other": "tensor" # or integer
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.floor_divide_1"] = {
    "args": {
        "input": "tensor",
        "other": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.ge"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"  # or float
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.gradient"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "spacing": "float",  # could also be list of floats or list of tensors
        "dim": "integer",  # could also be list of integers
        "edge_order": "integer"
    },
    "inner": {}
}
signatures["torch.imag"] = {
    "args": {
        "input": "tensor"  # Expecting a tensor as input
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.is_floating_point"] = {
    "args": {
        "input": "tensor"  # Could be tensor or tensor_list
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.is_grad_enabled"] = {
    "args": {},
    "kwargs": {},
    "inner": {}
}
signatures["torch.is_nonzero"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dim": "integer",  # Could also be None
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.isreal"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.is_storage"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.lcm"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.linalg.svdvals"] = {
    "args": {
        "A": "tensor"
    },
    "kwargs": {
        "driver": "string",  # could be None
        "out": "tensor"  # could be None
    },
    "inner": {}
}
signatures["torch.log1p"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could be None, but tensor is more specific
    },
    "inner": {}
}
signatures["torch.logsumexp"] = {
    "args": {
        "input": "tensor",
        "dim": "integer",  # or tuple of ints
    },
    "kwargs": {
        "keepdim": "boolean",
        "out": "tensor",
    },
    "inner": {},
}
signatures["torch.lu_solve"] = {
    "args": {
        "A": "tensor",
        "b": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.lu_unpack"] = {
    "args": {
        "lu": "tensor",
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.matrix_power"] = {
    "args": {
        "input": "tensor",
        "n": "integer"  # Could be float, but integer seems more appropriate for power
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.median_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.median_2"] = {
    "args": {
        "input": "tensor",
        "dim": "integer" # Could also be None
    },
    "kwargs": {
        "keepdim": "boolean",
        "out": "tuple" # tuple of tensors
    },
    "inner": {}
}
signatures["torch.minimum"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could be None, but tensor is more specific
    },
    "inner": {}
}
signatures["torch.msort"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could be None as well
    },
    "inner": {}
}
signatures["torch.nansum_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["torch.nansum_2"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"  # or tuple of ints
    },
    "kwargs": {
        "keepdim": "boolean",
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["torch.nextafter"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could also be None, but tensor is more specific
    },
    "inner": {}
}
signatures["torch.nn.AvgPool1d"] = {
    "args": {
        "kernel_size": "integer", # or tuple[int]
        "stride": "integer", # or tuple[int], default is kernel_size
        "padding": "integer", # or tuple[int], default is 0
    },
    "kwargs": {
        "ceil_mode": "boolean",
        "count_include_pad": "boolean",
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.AvgPool2d"] = {
    "args": {
        "kernel_size": "tuple",  # Can also be an integer
        "stride": "tuple",  # Can also be an integer, defaults to kernel_size
        "padding": "tuple",  # Can also be an integer, defaults to 0
    },
    "kwargs": {
        "ceil_mode": "boolean",
        "count_include_pad": "boolean",
        "divisor_override": "integer",
    },
    "inner": {
        "args": {
            "input": "tensor",
        },
        "kwargs": {},
    },
}
signatures["torch.nn.BatchNorm1d"] = {
    "args": {
        "num_features": "integer",
    },
    "kwargs": {
        "eps": "float",
        "momentum": "float",  # could be None
        "affine": "boolean",
        "track_running_stats": "boolean",
        "dtype": "dtype"
    },
    "inner": {},
}
signatures["torch.nn.BCEWithLogitsLoss"] = {
    "args": {},
    "kwargs": {
        "weight": "tensor",
        "size_average": "boolean",  # deprecated
        "reduce": "boolean",  # deprecated
        "reduction": "string",
        "pos_weight": "tensor"
    },
    "inner": {
        "args": {
            "input": "tensor",
            "target": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.FeatureAlphaDropout"] = {
    "args": {},
    "kwargs": {
        "p": "float", # could also be interpreted as a probability
        "inplace": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Fold"] = {
    "args": {
        "output_size": "tuple",  # Could be int or tuple
        "kernel_size": "tuple",  # Could be int or tuple
    },
    "kwargs": {
        "dilation": "tuple",  # Could be int or tuple, default 1
        "padding": "tuple",  # Could be int or tuple, default 0
        "stride": "tuple",  # Could be int or tuple, default 1
    },
    "inner": {},
}
signatures["torch.nn.functional.binary_cross_entropy_with_logits"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "weight": "tensor", # Could be None
        "size_average": "boolean", # Deprecated
        "reduce": "boolean",
        "pos_weight": "tensor" # Could be None
    },
    "inner": {},
}
signatures["torch.nn.functional.celu"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "alpha": "float",
        "inplace": "boolean"  # Could also be None, but boolean is more likely given documentation
    },
    "inner": {}
}
signatures["torch.nn.functional.dropout"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "p": "float",  # Could also be interpreted as a number
        "training": "boolean",
        "inplace": "boolean"
    },
    "inner": {}
}
signatures["torch.nn.functional.gumbel_softmax"] = {
    "args": {
        "input": "tensor",
        "tau": "float",
        "hard": "boolean",
        "dim": "integer"
    },
    "kwargs": {
        "eps": "float"  # Could also be a tensor.
    },
    "inner": {},
}
signatures["torch.nn.functional.hardshrink"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "lambd": "float" # could also be an integer
    },
    "inner": {},
}
signatures["torch.nn.functional.hardswish"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "inplace": "boolean"  # could also be None, but boolean seems more appropriate
    },
    "inner": {}
}
signatures["torch.nn.functional.kl_div"] = {
    "args": {
        "input": "tensor",
        "target": "tensor",
    },
    "kwargs": {
        "log_target": "boolean", # Could be tensor too, but documentation is not clear
        "reduction": "string", # Could be 'none', 'sum', 'mean', 'batchmean'
        "log_sum_exp": "boolean",
    },
    "inner": {},
}
signatures["torch.nn.functional.margin_ranking_loss"] = {
    "args": {
        "input1": "tensor",
        "input2": "tensor",
        "label": "tensor"  # could be integer as well
    },
    "kwargs": {
        "margin": "float",
        "reduction": "string"  # could also be integer
    },
    "inner": {},
}
signatures["torch.nn.functional.max_pool1d"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "integer",
        "stride": "integer",
        "padding": "integer",
        "dilation": "integer",
        "return_indices": "boolean",
        "ceil_mode": "boolean"
    },
    "kwargs": {
        "dtype": "dtype"  # Could be optional, but documented as the desired output type
    },
    "inner": {},
}
signatures["torch.nn.functional.multilabel_margin_loss"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "margin": "float",  # Should it be integer as well?
        "reduction": "string"  # Could be "mean", "sum", or "none"
    },
    "inner": {}
}
signatures["torch.nn.functional.pdist"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "p": "float",  # Could also be integer
    },
    "inner": {},
}
signatures["torch.nn.functional.prelu"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor"  # could also be a scalar float
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.functional.relu6"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "inplace": "boolean"  # could also be None
    },
    "inner": {},
}
signatures["torch.nn.functional.selu"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "inplace": "boolean"  # could also be None
    },
    "inner": {}
}
signatures["torch.nn.HuberLoss"] = {
    "args": {},
    "kwargs": {
        "reduction": "string",  # Could be 'none', 'mean', or 'sum'
        "delta": "float"
    },
    "inner": {
        "args": {
            "input": "tensor",
            "target": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.init.constant_"] = {
    "args": {
        "tensor": "tensor",
        "val": "float" # Could also be integer
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.LogSigmoid"] = {
    "args": {},
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor" # Could also be tensor_list, but documentation says any number of dimensions so tensor is more general
        },
        "kwargs": {}
    }
}
signatures["torch.nn.LogSoftmax"] = {
    "args": {
        "dim": "integer" # Could also be None, but integer is more specific
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.MaxPool2d"] = {
    "args": {
        "kernel_size": "tuple",  # could also be integer
        "stride": "tuple",  # could also be integer, defaults to kernel_size
        "padding": "tuple",  # could also be integer, defaults to 0
        "dilation": "tuple",  # could also be integer, defaults to 1
    },
    "kwargs": {
        "return_indices": "boolean",
        "ceil_mode": "boolean",
    },
    "inner": {},
}
signatures["torch.nn.MaxPool2d_1"] = {
    "args": {
        "kernel_size": "integer",
        "stride": "tuple",  # could also be integer, defaults to kernel_size
        "padding": "tuple",  # could also be integer, defaults to 0
        "dilation": "tuple",  # could also be integer, defaults to 1
    },
    "kwargs": {
        "return_indices": "boolean",
        "ceil_mode": "boolean",
    },
    "inner": {},
}
signatures["torch.nn.MaxPool2d_2"] = {
    "args": {
        "kernel_size": "integer",
        "stride": "integer",
        "padding": "tuple",  # could also be integer, defaults to 0
        "dilation": "tuple",  # could also be integer, defaults to 1
    },
    "kwargs": {
        "return_indices": "boolean",
        "ceil_mode": "boolean",
    },
    "inner": {},
}
signatures["torch.nn.MaxPool2d_3"] = {
    "args": {
        "kernel_size": "integer",
        "stride": "integer",
        "padding": "integer",
        "dilation": "tuple",  # could also be integer, defaults to 1
    },
    "kwargs": {
        "return_indices": "boolean",
        "ceil_mode": "boolean",
    },
    "inner": {},
}
signatures["torch.nn.MaxPool2d_4"] = {
    "args": {
        "kernel_size": "tuple",  # could also be integer
        "stride": "integer",
        "padding": "tuple",  # could also be integer, defaults to 0
        "dilation": "tuple",  # could also be integer, defaults to 1
    },
    "kwargs": {
        "return_indices": "boolean",
        "ceil_mode": "boolean",
    },
    "inner": {},
}
signatures["torch.nn.MaxPool2d_5"] = {
    "args": {
        "kernel_size": "tuple",  # could also be integer
        "stride": "integer",
        "padding": "integer",
        "dilation": "tuple",  # could also be integer, defaults to 1
    },
    "kwargs": {
        "return_indices": "boolean",
        "ceil_mode": "boolean",
    },
    "inner": {},
}
signatures["torch.nn.MaxPool3d"] = {
    "args": {
        "kernel_size": "tuple",  # or integer
        "stride": "tuple",  # or integer, defaults to kernel_size
        "padding": "tuple",  # or integer, defaults to 0
        "dilation": "tuple",  # or integer, defaults to 1
    },
    "kwargs": {
        "return_indices": "boolean",
        "ceil_mode": "boolean",
    },
    "inner": {},
}
signatures["torch.nn.MultiLabelMarginLoss"] = {
    "args": {},
    "kwargs": {
        "size_average": "boolean",  # Could be deprecated, consider removing
        "reduce": "boolean",  # Could be deprecated, consider removing
        "reduction": "string"
    },
    "inner": {
        "args": {
            "input": "tensor",
            "target": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.MultiMarginLoss"] = {
    "args": {
        "p": "integer",
        "margin": "float",
    },
    "kwargs": {
        "weight": "tensor",  # Could be None, but treating as tensor for simplicity
        "size_average": "boolean",  # Deprecated
        "reduce": "boolean",  # Deprecated
        "reduction": "string"
    },
    "inner": {
        "args": {
            "input": "tensor",
            "target": "tensor"  # or integer
        },
        "kwargs": {}
    }
}
signatures["torch.nn.ReflectionPad2d_1"] = {
    "args": {
        "padding": "integer"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor" # could also be tensor_list
        },
        "kwargs": {}
    }
}
signatures["torch.nn.ReflectionPad2d_2"] = {
    "args": {
        "padding": "tuple"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor" # could also be tensor_list
        },
        "kwargs": {}
    }
}
signatures["torch.nn.ReplicationPad3d_1"] = {
    "args": {
        "padding": "integer" # could also be tuple
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.ReplicationPad3d_2"] = {
    "args": {
        "padding": "tuple"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.RReLU"] = {
    "args": {},
    "kwargs": {
        "lower": "float",
        "upper": "float",
        "inplace": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"  # Input is a tensor
        },
        "kwargs": {}
    }
}
signatures["torch.nn.SiLU"] = {
    "args": {},
    "kwargs": {
        "inplace": "boolean"  # Could also be None, but boolean seems more appropriate
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.SmoothL1Loss"] = {
    "args": {},
    "kwargs": {
        "beta": "float",  # could also be integer
        "reduction": "string",
        "size_average": "boolean", # deprecated
        "reduce": "boolean" # deprecated
    },
    "inner": {
        "args": {
            "input": "tensor",
            "target": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Softmax"] = {
    "args": {},
    "kwargs": {
        "dim": "integer" # Could also be None, but integer is more specific
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Softmin"] = {
    "args": {
        "input": "tensor" # Could also be tensor_list
    },
    "kwargs": {
        "dim": "integer"
    },
    "inner": {}
}
signatures["torch.nn.Softshrink"] = {
    "args": {
        "lambd": "float"  # Should it be lambda instead of lambd?
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.utils.clip_grad_norm_"] = {
    "args": {
        "parameters": "tensor_list",
        "max_norm": "float",
        "norm_type": "integer" # Could also be float, depending on the value of p
    },
    "kwargs": {
        "eps": "float",
        "norm_type": "integer" # Could also be float, depending on the value of p
    },
    "inner": {},
}
signatures["torch.not_equal"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.numel"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.permute"] = {
    "args": {
        "input": "tensor",
        "dims": "tuple"  # Could also be list of integers
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.positive"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.reciprocal"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could be None, but tensor is more specific
    },
    "inner": {}
}
signatures["torch.reshape"] = {
    "args": {
        "input": "tensor",
        "shape": "tuple"  # Could also be a list of integers
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.round"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "decimals": "integer",  # could also be float
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.set_num_interop_threads"] = {
    "args": {
        "num_threads": "integer"  # Could be float, but integer seems more likely
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.signbit"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could be None, but tensor is more specific
    },
    "inner": {}
}
signatures["torch.sin"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could be None, but tensor is more specific
    },
    "inner": {}
}
signatures["torch.special.erfc"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.special.i0e"] = {
    "args": {
        "x": "tensor"  # Could be tensor_list as well
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.special.i1e"] = {
    "args": {
        "x": "tensor"  # could also be tensor_list
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.special.polygamma"] = {
    "args": {
        "n": "integer",  # Could also be float, but integer seems more common for the order
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.special.sinc"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.special.xlog1py"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "threshold": "float" # Could also be an integer
    },
    "inner": {},
}
signatures["torch.sqrt"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # could also be None, but tensor is more specific
    },
    "inner": {}
}
signatures["torch.take"] = {
    "args": {
        "input": "tensor",
        "index": "tensor"  # should be LongTensor, but tensor is closest
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.tril"] = {
    "args": {
        "input": "tensor",
        "diagonal": "integer"  # could also be float, but integer is more common
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.unique_consecutive"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "return_inverse": "boolean",
        "return_counts": "boolean"
    },
    "inner": {}
}

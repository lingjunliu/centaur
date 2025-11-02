signatures = {}
signatures["torch.fix"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.is_grad_enabled"] = {
    "args": {},
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.functional.hardshrink"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "lambd": "float"
    },
    "inner": {},
}
signatures["torch.reciprocal"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": " tensor"
    },
    "inner": {}
}
signatures["torch.nn.functional.celu"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "alpha": "float",
        "inplace": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.MaxPool2d_1"] = {
    "args": {
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "integer",
        "dilation": "integer",
        "return_indices": "boolean",
        "ceil_mode": "boolean"
    },
    "inner": {}
}

signatures["torch.nn.MaxPool2d_2"] = {
    "args": {
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "tuple",
        "dilation": "tuple",
        "return_indices": "boolean",
        "ceil_mode": "boolean"
    },
    "inner": {}
}

signatures["torch.nn.MaxPool2d_3"] = {
    "args": {
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "tuple",
        "dilation": "tuple",
        "return_indices": "boolean",
        "ceil_mode": "boolean"
    },
    "inner": {}
}

signatures["torch.nn.MaxPool2d_4"] = {
    "args": {
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "integer",
        "dilation": "integer",
        "return_indices": "boolean",
        "ceil_mode": "boolean"
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
        "dim": "integer"
    },
    "kwargs": {
        "keepdim": "boolean",
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["torch.cartesian_prod"] = {
    "args": {
        " tensors": "tensor_list"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nextafter"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.tril_1"] = {
    "args": {
        "input": "tensor",
        "diagonal": "integer"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.nn.functional.max_pool1d_1"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "integer",
        "dilation": "integer",
        "ceil_mode": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.Softmax"] = {
    "args": {
        "dim": "integer"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.are_deterministic_algorithms_enabled"] = {
    "args": {},
    "kwargs": {},
    "inner": {}
}
signatures["torch.numel"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.ReflectionPad2d_1"] = {
    "args": {
        "padding": "integer"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
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
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.minimum_1"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}

signatures["torch.minimum_2"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.set_num_interop_threads"] = {
    "args": {
        "num_threads": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.AvgPool2d_1"] = {
    "args": {
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "integer",
        "ceil_mode": "boolean",
        "count_include_pad": "boolean",
        "divisor_override": "integer"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}

signatures["torch.nn.AvgPool2d_2"] = {
    "args": {
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "tuple",
        "ceil_mode": "boolean",
        "count_include_pad": "boolean",
        "divisor_override": "integer"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.clip_1"] = {
    "args": {
        "input": "tensor",
        "min": "tensor",
        "max": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.functional.pdist"] = {
    "args": {
        "input": "tensor",
        "p": "float"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.not_equal"] = {
    "args": {
        "input": "tensor",
        " other": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.amin_1"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {
        "keepdim": "boolean",
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.amin_2"] = {
    "args": {
        "input": "tensor",
        "dim": "tuple"
    },
    "kwargs": {
        "keepdim": "boolean",
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
signatures["torch.nn.LogSigmoid"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.functional.binary_cross_entropy_with_logits"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "weight": "tensor",
        "size_average": "boolean",
        "reduce": "boolean",
        "reduction": "string"
    },
    "inner": {}
}
signatures["torch.nn.functional.kl_div_1"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        " reduction": "string",
        " log_target": "boolean"
    },
    "inner": {},
}

signatures["torch.nn.functional.kl_div_2"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        " reduction": "string",
        " log_target": "boolean"
    },
    "inner": {},
}
signatures["torch.unique_consecutive"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {
        "return_inverse": "boolean",
        "return_counts": "boolean"
    },
    "inner": {},
}
signatures["torch.is_nonzero"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.SiLU"] = {
    "args": {
        "inplace": "boolean"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.bitwise_left_shift"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.conj_physical"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.take_1"] = {
    "args": {
        "input": "tensor",
        "index": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.round"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "decimals": "integer",
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
signatures["torch.nn.utils.clip_grad_norm__1"] = {
    "args": {
        "parameters": "tensor_list"
    },
    "kwargs": {
        "max_norm": "float",
        "norm_type": "float"
    },
    "inner": {}
}
signatures["torch.nn.RReLU"] = {
    "args": {
        "lower": "float",
        "upper": "float"
    },
    "kwargs": {
        "inplace": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.ge"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.special.i1e"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.diag_embed"] = {
    "args": {
        "input": "tensor",
        "k": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.linalg.svdvals_1"] = {
    "args": {
        "A": "tensor"
    },
    "kwargs": {
        "driver": "string",
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.nn.MaxPool3d"] = {
    "args": {
        "kernel_size": "tuple",
        "stride": "tuple"
    },
    "kwargs": {
        "padding": "tuple",
        "dilation": "tuple",
        "return_indices": "boolean",
        "ceil_mode": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.asin"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.addcdiv_1"] = {
    "args": {
        "input": "tensor",
        "tensor1": "tensor",
        "tensor2": "tensor"
    },
    "kwargs": {
        "value": "float",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.msort"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.nn.Softshrink"] = {
    "args": {
        "lambd": "float"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.bincount_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "weights": "tensor",
        "minlength": "integer"
    },
    "inner": {},
}

signatures["torch.bincount_2"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "weights": "tensor",
        "minlength": "integer"
    },
    "inner": {},
}
signatures["torch.nn.functional.multilabel_margin_loss"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "weight": "tensor",
        "reduction": "string"
    },
    "inner": {},
}
signatures["torch.special.xlog1py"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.is_floating_point"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.SmoothL1Loss"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "size_average": "boolean",
        "reduce": "boolean",
        "reduction": "string",
        "beta": "float"
    },
    "inner": {}
}
signatures["torch.nn.Fold_1"] = {
    "args": {
        "output_size": "tuple",
        "kernel_size": "tuple"
    },
    "kwargs": {
        "dilation": "tuple",
        "padding": "tuple",
        "stride": "tuple"
    },
    "inner": {}
}
signatures["torch.nn.Fold_2"] = {
    "args": {
        "output_size": "integer",
        "kernel_size": "integer"
    },
    "kwargs": {
        "dilation": "integer",
        "padding": "integer",
        "stride": "integer"
    },
    "inner": {}
}
signatures["torch.nn.init.constant__1"] = {
    "args": {
        "tensor": "tensor",
        "val": "float"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.special.sinc"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.matrix_power"] = {
    "args": {
        "input": "tensor",
        "n": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.functional.dropout_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "p": "float",
        "training": "boolean",
        "inplace": "boolean"
    },
    "inner": {}
}
signatures["torch.addmv_1"] = {
    "args": {
        "input": "tensor",
        "mat": "tensor",
        "vec": "tensor"
    },
    "kwargs": {
        "beta": "float",
        "alpha": "float"
    },
    "inner": {}
}
signatures["torch.special.i0e"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.gradient"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "spacing": "tensor_list",
        "dim": "integer",
        "edge_order": "integer"
    },
    "inner": {}
}
signatures["torch.nn.BCEWithLogitsLoss_1"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "weight": "tensor",
        "size_average": "boolean",
        "reduce": "boolean",
        "reduction": "string",
        "pos_weight": "tensor"
    },
    "inner": {}
}
signatures["torch.nn.BCEWithLogitsLoss_2"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "weight": "tensor",
        "reduction": "string",
        "pos_weight": "tensor"
    },
    "inner": {}
}
signatures["torch.lu_solve"] = {
    "args": {
        "b": "tensor",
        "LU": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.permute"] = {
    "args": {
        "input": "tensor",
        "dims": "tuple"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.imag"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.MultiLabelMarginLoss"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "size_average": "boolean",
        "reduce": "boolean",
        "reduction": "string"
    },
    "inner": {}
}
signatures["torch.argsort"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {
        "descending": "boolean",
        "stable": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.gumbel_softmax_1"] = {
    "args": {
        "input": "tensor",
        "tau": "float",
        "hard": "boolean"
    },
    "kwargs": {
        "dim": "integer",
        "keepdim": "boolean"
    },
    "inner": {},
}
signatures["torch.special.erfc"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.Softmin"] = {
    "args": {
        "dim": "integer"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.floor_divide"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
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
    "inner": {}
}
signatures["torch.nn.MultiMarginLoss_1"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "p": "integer",
        "margin": "float",
        "weight": "tensor",
        "size_average": "boolean",
        "reduce": "boolean",
        "reduction": "string"
    },
    "inner": {}
}
signatures["torch.equal"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.lu_unpack"] = {
    "args": {
        "LU": "tensor",
        "pivot": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.log1p"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.is_storage"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.reshape_1"] = {
    "args": {
        "input": "tensor",
        "shape": "tuple"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.AvgPool1d"] = {
    "args": {
        "kernel_size": "integer",
        "stride": "integer"
    },
    "kwargs": {
        "padding": "integer",
        "ceil_mode": "boolean",
        "count_include_pad": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.HuberLoss"] = {
    "args": {
        "reduction": "string",
        "delta": "float"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor",
            "target": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.any_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}

signatures["torch.any_2"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {
        "keepdim": "boolean",
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.nn.FeatureAlphaDropout"] = {
    "args": {
        "p": "float",
        "inplace": "boolean"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.BatchNorm1d_1"] = {
    "args": {
        "num_features": "integer"
    },
    "kwargs": {
        "eps": "float",
        "momentum": "float",
        "affine": "boolean",
        "track_running_stats": "boolean",
        "device": "string",
        "dtype": "string"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.special.polygamma"] = {
    "args": {
        "n": "integer",
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.logsumexp"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {
        "keepdim": "boolean",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.sqrt"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": " tensor"
    },
    "inner": {}
}
signatures["torch.nn.functional.margin_ranking_loss"] = {
    "args": {
        "input1": "tensor",
        "input2": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "margin": "float",
        "reduction": "string"
    },
    "inner": {},
}
signatures["torch.flipud_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.float_power"] = {
    "args": {
        "input": "tensor",
        "exponent": "tensor"
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
signatures["torch.nn.functional.relu6"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "inplace": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.ReplicationPad3d_1"] = {
    "args": {
        "padding": "integer"
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
signatures["torch.signbit"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.nn.functional.selu"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "inplace": "boolean"
    },
    "inner": {},
}
signatures["torch.complex"] = {
    "args": {
        "real": "tensor",
        "imag": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.nn.LogSoftmax"] = {
    "args": {
        "dim": "integer"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.floor"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.nn.functional.prelu"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.sin"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
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
        "dim": "integer"
    },
    "kwargs": {
        "keepdim": "boolean",
        "out": "tuple"
    },
    "inner": {}
}
signatures["torch.positive_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.empty_like"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dtype": "dtype",
        "layout": "string",
        "device": "string",
        "requires_grad": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.hardswish"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "inplace": "boolean"
    },
    "inner": {}
}

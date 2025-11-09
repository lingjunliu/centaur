signatures = {}
signatures["torch.addcdiv"] = {
    "args": {
        "input": "tensor",
        "tensor1": "tensor",
        "tensor2": "tensor"
    },
    "kwargs": {
        "value": "float",
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
        "beta": "float",  # Can be Number (real or integer), using float as closest match
        "alpha": "float",  # Can be Number (real or integer), using float as closest match
        "out": "tensor"
    },
    "inner": {}
}
# torch.amin has two signatures based on dim parameter type
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

signatures["torch.any_3"] = {
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
signatures["torch.are_deterministic_algorithms_enabled"] = {
    "args": {},
    "kwargs": {},
    "inner": {},
}
signatures["torch.argsort"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dim": "integer",
        "descending": "boolean",
        "stable": "boolean"
    },
    "inner": {}
}
signatures["torch.asin"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.bincount"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "weights": "tensor",
        "minlength": "integer"
    },
    "inner": {}
}
signatures["torch.bitwise_left_shift"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.cartesian_prod"] = {
    "args": {},
    "kwargs": {},
    "inner": {}
}
signatures["torch.clip"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "min": "float",
        "max": "float",
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.complex"] = {
    "args": {
        "real": "tensor",
        "imag": "tensor"
    },
    "kwargs": {
        "out": "tensor"
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
    "inner": {},
}
signatures["torch.dequantize_2"] = {
    "args": {
        "tensors": "tensor_list"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.diag_embed"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "offset": "integer",
        "dim1": "integer",
        "dim2": "integer"
    },
    "inner": {}
}
signatures["torch.empty_like"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dtype": "dtype",
        "layout": "string",
        "requires_grad": "boolean",
        "memory_format": "string"
    },
    "inner": {}
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
        "out": "tensor"
    },
    "inner": {}
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
        "exponent": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
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
signatures["torch.floor_divide_1"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.floor_divide_2"] = {
    "args": {
        "input": "tensor",
        "other": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.floor_divide_3"] = {
    "args": {
        "input": "tensor",
        "other": "float"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.ge_1"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}

signatures["torch.ge_2"] = {
    "args": {
        "input": "tensor",
        "other": "float"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.gradient_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "spacing": "float",
        "dim": "integer",
        "edge_order": "integer"
    },
    "inner": {}
}

signatures["torch.gradient_2"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "spacing": "list",
        "dim": "integer",
        "edge_order": "integer"
    },
    "inner": {}
}

signatures["torch.gradient_3"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "spacing": "tuple",
        "dim": "integer",
        "edge_order": "integer"
    },
    "inner": {}
}

signatures["torch.gradient_4"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "spacing": "float",
        "dim": "list",
        "edge_order": "integer"
    },
    "inner": {}
}

signatures["torch.gradient_5"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "spacing": "list",
        "dim": "list",
        "edge_order": "integer"
    },
    "inner": {}
}

signatures["torch.gradient_6"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "spacing": "tuple",
        "dim": "list",
        "edge_order": "integer"
    },
    "inner": {}
}
signatures["torch.imag"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.is_floating_point"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.is_grad_enabled"] = {
    "args": {},
    "kwargs": {},
    "inner": {},
}
signatures["torch.is_nonzero"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
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
        "obj": "tensor"  # accepts any object, but most commonly tensor/storage
    },
    "kwargs": {},
    "inner": {}
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
signatures["torch.linalg.svdvals"] = {
    "args": {
        "A": "tensor"
    },
    "kwargs": {
        "driver": "string",
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.log1p"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.logsumexp_1"] = {
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

signatures["torch.logsumexp_2"] = {
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
signatures["torch.lu_solve"] = {
    "args": {
        "input": "tensor",
        "LU_data": "tensor",
        "LU_pivots": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.lu_unpack"] = {
    "args": {
        "LU_data": "tensor",
        "LU_pivots": "tensor"
    },
    "kwargs": {
        "unpack_data": "boolean",
        "unpack_pivots": "boolean"
    },
    "inner": {}
}
signatures["torch.matrix_power"] = {
    "args": {
        "input": "tensor",
        "n": "integer"
    },
    "kwargs": {},
    "inner": {}
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
        "input": "tensor"
    },
    "kwargs": {
        "dim": "integer",
        "keepdim": "boolean",
        "out": "tuple"
    },
    "inner": {}
}
signatures["torch.minimum"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.msort"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
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

signatures["torch.nansum_3"] = {
    "args": {
        "input": "tensor",
        "dim": "tuple"
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
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.nn.AvgPool1d_1"] = {
    "args": {
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "integer",
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
signatures["torch.nn.AvgPool1d_2"] = {
    "args": {
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "tuple",
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

signatures["torch.nn.AvgPool2d_3"] = {
    "args": {
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "tuple",
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

signatures["torch.nn.AvgPool2d_4"] = {
    "args": {
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "tuple",
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

signatures["torch.nn.AvgPool2d_5"] = {
    "args": {
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "integer",
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

signatures["torch.nn.AvgPool2d_6"] = {
    "args": {
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "integer",
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

signatures["torch.nn.AvgPool2d_7"] = {
    "args": {
        "kernel_size": "integer"
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

signatures["torch.nn.AvgPool2d_8"] = {
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
    }
}
signatures["torch.nn.BatchNorm1d"] = {
    "args": {
        "num_features": "integer"
    },
    "kwargs": {
        "eps": "float",
        "momentum": "float",
        "affine": "boolean",
        "track_running_stats": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.BCEWithLogitsLoss"] = {
    "args": {},
    "kwargs": {
        "weight": "tensor",
        "size_average": "boolean",
        "reduce": "boolean",
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
        "p": "float",
        "inplace": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}

signatures["torch.nn.Fold_1"] = {
    "args": {
        "output_size": "integer"
    },
    "kwargs": {
        "kernel_size": "integer",
        "dilation": "integer",
        "padding": "integer",
        "stride": "integer"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Fold_2"] = {
    "args": {
        "output_size": "tuple"
    },
    "kwargs": {
        "kernel_size": "integer",
        "dilation": "integer",
        "padding": "integer",
        "stride": "integer"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Fold_3"] = {
    "args": {
        "output_size": "integer"
    },
    "kwargs": {
        "kernel_size": "tuple",
        "dilation": "integer",
        "padding": "integer",
        "stride": "integer"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Fold_4"] = {
    "args": {
        "output_size": "tuple"
    },
    "kwargs": {
        "kernel_size": "tuple",
        "dilation": "integer",
        "padding": "integer",
        "stride": "integer"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Fold_5"] = {
    "args": {
        "output_size": "integer"
    },
    "kwargs": {
        "kernel_size": "integer",
        "dilation": "tuple",
        "padding": "integer",
        "stride": "integer"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Fold_6"] = {
    "args": {
        "output_size": "tuple"
    },
    "kwargs": {
        "kernel_size": "tuple",
        "dilation": "tuple",
        "padding": "integer",
        "stride": "integer"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Fold_7"] = {
    "args": {
        "output_size": "integer"
    },
    "kwargs": {
        "kernel_size": "integer",
        "dilation": "integer",
        "padding": "tuple",
        "stride": "integer"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Fold_8"] = {
    "args": {
        "output_size": "tuple"
    },
    "kwargs": {
        "kernel_size": "tuple",
        "dilation": "tuple",
        "padding": "tuple",
        "stride": "integer"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Fold_9"] = {
    "args": {
        "output_size": "integer"
    },
    "kwargs": {
        "kernel_size": "integer",
        "dilation": "integer",
        "padding": "integer",
        "stride": "tuple"
    },
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
        "reduction": "string",
        "pos_weight": "tensor"
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
    "inner": {}
}
signatures["torch.nn.functional.dropout"] = {
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
signatures["torch.nn.functional.gumbel_softmax"] = {
    "args": {
        "logits": "tensor",
    },
    "kwargs": {
        "tau": "float",
        "hard": "boolean",
        "eps": "float",
        "dim": "integer"
    },
    "inner": {},
}
signatures["torch.nn.functional.hardshrink"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "lambd": "float"
    },
    "inner": {}
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
signatures["torch.nn.functional.kl_div"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "size_average": "boolean",
        "reduce": "boolean",
        "reduction": "string",
        "log_target": "boolean"
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
        "size_average": "boolean",
        "reduce": "boolean",
        "reduction": "string"
    },
    "inner": {}
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
        "ceil_mode": "boolean",
        "return_indices": "boolean"
    },
    "inner": {}
}

signatures["torch.nn.functional.max_pool1d_2"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "tuple",
        "dilation": "tuple",
        "ceil_mode": "boolean",
        "return_indices": "boolean"
    },
    "inner": {}
}
signatures["torch.nn.functional.multilabel_margin_loss"] = {
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
signatures["torch.nn.functional.pdist"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "p": "float"  # accepts integer or float values, but float is more general
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
signatures["torch.nn.functional.relu6"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "inplace": "boolean"
    },
    "inner": {}
}
signatures["torch.nn.functional.selu"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "inplace": "boolean"
    },
    "inner": {}
}
signatures["torch.nn.HuberLoss"] = {
    "args": {},
    "kwargs": {
        "reduction": "string",
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
        "val": "float"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.LogSigmoid"] = {
    "args": {},
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.LogSoftmax"] = {
    "args": {},
    "kwargs": {
        "dim": "integer"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
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
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
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
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.MaxPool3d_1"] = {
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
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}

signatures["torch.nn.MaxPool3d_2"] = {
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
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.MultiLabelMarginLoss"] = {
    "args": {},
    "kwargs": {
        "size_average": "boolean",
        "reduce": "boolean",
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
    "args": {},
    "kwargs": {
        "p": "integer",
        "margin": "float",
        "weight": "tensor",
        "size_average": "boolean",
        "reduce": "boolean",
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
signatures["torch.nn.RReLU"] = {
    "args": {},
    "kwargs": {
        "lower": "float",
        "upper": "float",
        "inplace": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.SiLU"] = {
    "args": {},
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
signatures["torch.nn.SmoothL1Loss"] = {
    "args": {},
    "kwargs": {
        "size_average": "boolean",
        "reduce": "boolean",
        "reduction": "string",
        "beta": "float"
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
        "dim": "integer"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Softmin"] = {
    "args": {},
    "kwargs": {
        "dim": "integer"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Softshrink"] = {
    "args": {},
    "kwargs": {
        "lambd": "float"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.utils.clip_grad_norm_"] = {
    "args": {
        "parameters": "tensor_list",  # iterable of parameters or a single tensor
        "max_norm": "float",
    },
    "kwargs": {
        "norm_type": "float",
        "error_if_nonfinite": "boolean",
        "foreach": "boolean",
    },
    "inner": {},
}
signatures["torch.not_equal"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
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
        "dims": "tuple"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.positive"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.reciprocal"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.reshape"] = {
    "args": {
        "input": "tensor",
        "shape": "tuple"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.round"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "decimals": "integer",
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.set_num_interop_threads"] = {
    "args": {
        "num": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.signbit"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.sin"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.special.erfc"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.special.i0e"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.special.i1e"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.special.polygamma"] = {
    "args": {
        "n": "integer",
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.special.sinc"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.special.xlog1py"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.sqrt"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.take"] = {
    "args": {
        "input": "tensor",
        "index": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.tril"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "diagonal": "integer",
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
        "return_counts": "boolean",
        "dim": "integer"
    },
    "inner": {}
}

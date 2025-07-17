signatures = {}
signatures["torch.nn.Flatten"] = {
    "args": {},
    "kwargs": {
        "start_dim": "integer",
        "end_dim": "integer"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.FractionalMaxPool2d_1"] = {
    "args": {
        "kernel_size": "integer",
    },
    "kwargs": {
        "output_size": "tuple",
        "output_ratio": "tuple",
        "return_indices": "boolean",
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.FractionalMaxPool2d_2"] = {
    "args": {
        "kernel_size": "tuple",
    },
    "kwargs": {
        "output_size": "tuple",
        "output_ratio": "tuple",
        "return_indices": "boolean",
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.GroupNorm"] = {
    "args": {
        "num_groups": "integer",
        "num_channels": "integer"
    },
    "kwargs": {
        "eps": "float",
        "affine": "boolean",
        "dtype": "dtype"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.Hardshrink"] = {
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
signatures["torch.nn.Hardswish"] = {
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
signatures["torch.nn.Hardtanh"] = {
    "args": {},
    "kwargs": {
        "min_val": "float",
        "max_val": "float",
        "inplace": "boolean",
        "min_value": "float", #deprecated
        "max_value": "float" #deprecated
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.InstanceNorm1d"] = {
    "args": {
        "num_features": "integer"
    },
    "kwargs": {
        "eps": "float",
        "momentum": "float",
        "affine": "boolean",
        "track_running_stats": "boolean",
        "dtype": "dtype"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.InstanceNorm2d"] = {
    "args": {
        "num_features": "integer"
    },
    "kwargs": {
        "eps": "float",
        "momentum": "float", # Optional[float] but can only be float
        "affine": "boolean",
        "track_running_stats": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.InstanceNorm3d"] = {
    "args": {
        "num_features": "integer",
    },
    "kwargs": {
        "eps": "float",
        "momentum": "float", # Optional[float] is still a float
        "affine": "boolean",
        "track_running_stats": "boolean",
        "dtype": "dtype"
    },
    "inner": {
        "args": {
            "input": "tensor",
        },
        "kwargs": {}
    },
}
signatures["torch.nn.L1Loss"] = {
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
    },
}
signatures["torch.nn.LPPool1d"] = {
    "args": {
        "norm_type": "integer",
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "integer",
        "ceil_mode": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.LPPool2d_1"] = {
    "args": {
        "norm_type": "float",
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "integer",
        "ceil_mode": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.LPPool2d_2"] = {
    "args": {
        "norm_type": "float",
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "tuple",
        "ceil_mode": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.LSTMCell"] = {
    "args": {
        "input_size": "integer",
        "hidden_size": "integer"
    },
    "kwargs": {
        "bias": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor",
            "h_0": "tensor",
            "c_0": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.LayerNorm_1"] = {
    "args": {
        "normalized_shape": "integer"
    },
    "kwargs": {
        "eps": "float",
        "elementwise_affine": "boolean",
        "bias": "boolean",
        "dtype": "dtype"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.LayerNorm_2"] = {
    "args": {
        "normalized_shape": "list" # Could also be tuple, but 'list' seems more general
    },
    "kwargs": {
        "eps": "float",
        "elementwise_affine": "boolean",
        "bias": "boolean",
        "dtype": "dtype"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.LayerNorm_3"] = {
    "args": {
        "normalized_shape": "tuple" # Could also be list, but 'tuple' seems more general
    },
    "kwargs": {
        "eps": "float",
        "elementwise_affine": "boolean",
        "bias": "boolean",
        "dtype": "dtype"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.LeakyReLU"] = {
    "args": {},
    "kwargs": {
        "negative_slope": "float",
        "inplace": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Linear"] = {
    "args": {
        "in_features": "integer",
        "out_features": "integer"
    },
    "kwargs": {
        "bias": "boolean",
        "dtype": "dtype"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
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
signatures["torch.nn.MSELoss"] = {
    "args": {},
    "kwargs": {
        "size_average": "boolean", # deprecated
        "reduce": "boolean", # deprecated
        "reduction": "string"
    },
    "inner": {
        "args": {
            "input": "tensor",
            "target": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.MarginRankingLoss"] = {
    "args": {},
    "kwargs": {
        "margin": "float",
        "size_average": "boolean",
        "reduce": "boolean",
        "reduction": "string"
    },
    "inner": {
        "args": {
            "input1": "tensor",
            "input2": "tensor",
            "target": "tensor"
        },
        "kwargs": {}
    },
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
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.MaxPool2d_4"] = {
    "args": {
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "integer",
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
signatures["torch.nn.MaxPool2d_5"] = {
    "args": {
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "integer",
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
signatures["torch.nn.MaxPool2d_6"] = {
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
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.MaxPool2d_7"] = {
    "args": {
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "tuple",
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
signatures["torch.nn.MaxPool2d_8"] = {
    "args": {
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "tuple",
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
signatures["torch.nn.MaxPool2d_9"] = {
    "args": {
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "integer",
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
signatures["torch.nn.MaxPool2d_10"] = {
    "args": {
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "tuple",
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
signatures["torch.nn.MaxPool2d_11"] = {
    "args": {
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "integer",
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
signatures["torch.nn.MaxPool2d_12"] = {
    "args": {
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "integer",
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
signatures["torch.nn.MaxPool2d_13"] = {
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
signatures["torch.nn.MaxUnpool2d_1"] = {
    "args": {
        "kernel_size": "integer",
    },
    "kwargs": {
        "stride": "integer",
        "padding": "integer"
    },
    "inner": {
        "args": {
            "input": "tensor",
            "indices": "tensor",
            "output_size": "tuple" #Could also be a tensor? Unclear.
        },
        "kwargs": {}
    },
}
signatures["torch.nn.MaxUnpool2d_2"] = {
    "args": {
        "kernel_size": "tuple",
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "tuple"
    },
    "inner": {
        "args": {
            "input": "tensor",
            "indices": "tensor",
            "output_size": "tuple" #Could also be a tensor? Unclear.
        },
        "kwargs": {}
    },
}
signatures["torch.nn.MultiLabelSoftMarginLoss"] = {
    "args": {},
    "kwargs": {
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
    },
}
signatures["torch.nn.NLLLoss"] = {
    "args": {},
    "kwargs": {
        "weight": "tensor",
        "size_average": "boolean",
        "ignore_index": "integer",
        "reduce": "boolean",
        "reduction": "string"
    },
    "inner": {
        "args": {
            "input": "tensor",
            "target": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.PReLU"] = {
    "args": {},
    "kwargs": {
        "num_parameters": "integer",
        "init": "float",
        "dtype": "dtype" # Could also be None, not sure how to represent this better
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.PairwiseDistance"] = {
    "args": {},
    "kwargs": {
        "p": "float",
        "eps": "float",
        "keepdim": "boolean"
    },
    "inner": {
        "args": {
            "input1": "tensor",
            "input2": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.PixelShuffle"] = {
    "args": {
        "upscale_factor": "integer"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.PoissonNLLLoss"] = {
    "args": {},
    "kwargs": {
        "log_input": "boolean",
        "full": "boolean",
        "size_average": "boolean",
        "eps": "float",
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
signatures["torch.nn.ReLU6"] = {
    "args": {},
    "kwargs": {
        "inplace": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.ReLU"] = {
    "args": {},
    "kwargs": {
        "inplace": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor" # Input tensor
        },
        "kwargs": {}
    }
}
signatures["torch.nn.ReflectionPad1d_1"] = {
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
signatures["torch.nn.ReflectionPad1d_2"] = {
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
signatures["torch.nn.ReplicationPad1d_1"] = {
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
signatures["torch.nn.ReplicationPad1d_2"] = {
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
signatures["torch.nn.SELU"] = {
    "args": {},
    "kwargs": {
        "inplace": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor" # The input tensor
        },
        "kwargs": {}
    },
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
signatures["torch.nn.Sigmoid"] = {
    "args": {},
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
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
    },
}
signatures["torch.nn.Softmax2d"] = {
    "args": {},
    "kwargs": {},
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
    },
}
signatures["torch.nn.Softplus"] = {
    "args": {},
    "kwargs": {
        "beta": "float",
        "threshold": "float"
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
    },
}
signatures["torch.nn.Softsign"] = {
    "args": {},
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.abs"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.acos"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.acosh"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.nn.functional.adaptive_avg_pool1d_1"] = {
    "args": {
        "input": "tensor",
        "output_size": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.functional.adaptive_avg_pool1d_2"] = {
    "args": {
        "input": "tensor",
        "output_size": "tuple"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.functional.adaptive_avg_pool2d_1"] = {
    "args": {
        "input": "tensor",
        "output_size": "tuple"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.functional.adaptive_avg_pool2d_2"] = {
    "args": {
        "input": "tensor",
        "output_size": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.functional.adaptive_max_pool1d_1"] = {
    "args": {
        "input": "tensor",
        "output_size": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.functional.adaptive_max_pool1d_2"] = {
    "args": {
        "input": "tensor",
        "output_size": "tuple"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.functional.adaptive_max_pool2d_1"] = {
    "args": {
        "input": "tensor",
        "output_size": "tuple"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.functional.adaptive_max_pool2d_2"] = {
    "args": {
        "input": "tensor",
        "output_size": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.add_1"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "alpha": "float",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.add_2"] = {
    "args": {
        "input": "tensor",
        "other": "float"
    },
    "kwargs": {
        "alpha": "float",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.addbmm"] = {
    "args": {
        "input": "tensor",
        "batch1": "tensor",
        "batch2": "tensor"
    },
    "kwargs": {
        "beta": "float", # Could also be integer, but float is more general for FloatTensor and DoubleTensor
        "alpha": "float", # Could also be integer, but float is more general for FloatTensor and DoubleTensor
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.addcdiv"] = {
    "args": {
        "input": "tensor",
        "tensor1": "tensor",
        "tensor2": "tensor"
    },
    "kwargs": {
        "value": "float",  # Could also be integer depending on input type
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.addcmul"] = {
    "args": {
        "input": "tensor",
        "tensor1": "tensor",
        "tensor2": "tensor"
    },
    "kwargs": {
        "value": "float", #Could also be an integer, but float is more general
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.addmm"] = {
    "args": {
        "input": "tensor",
        "mat1": "tensor",
        "mat2": "tensor"
    },
    "kwargs": {
        "beta": "float", # Could also be integer, but float seems more general
        "alpha": "float", # Could also be integer, but float seems more general
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.addmv"] = {
    "args": {
        "input": "tensor",
        "mat": "tensor",
        "vec": "tensor"
    },
    "kwargs": {
        "beta": "float", # Should this be Number?
        "alpha": "float", # Should this be Number?
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.addr"] = {
    "args": {
        "input": "tensor",
        "vec1": "tensor",
        "vec2": "tensor"
    },
    "kwargs": {
        "beta": "float", # Number should be float
        "alpha": "float", # Number should be float
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.allclose"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "rtol": "float",
        "atol": "float",
        "equal_nan": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.alpha_dropout"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "p": "float",
        "training": "boolean",
        "inplace": "boolean"
    },
    "inner": {},
}
signatures["torch.amax_1"] = {
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
signatures["torch.amax_2"] = {
    "args": {
        "input": "tensor",
        "dim": "tuple"
    },
    "kwargs": {
        "keepdim": "boolean",
        "out": "tensor"
    },
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
    "inner": {},
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
    "inner": {},
}
signatures["torch.angle"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.arange"] = {
    "args": {
        "start": "float", # Could also be integer
        "end": "float",
        "step": "float" # Could also be integer
    },
    "kwargs": {
        "out": "tensor",
        "dtype": "dtype",
        "layout": "string",
        "requires_grad": "boolean"
    },
    "inner": {},
}
signatures["torch.argmax_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.argmax_2"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {
        "keepdim": "boolean"
    },
    "inner": {},
}
signatures["torch.argmin"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "dim": "integer",
        "keepdim": "boolean"
    },
    "inner": {},
}
signatures["torch.argsort"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "dim": "integer",
        "descending": "boolean",
        "stable": "boolean"
    },
    "inner": {},
}
signatures["torch.as_strided"] = {
    "args": {
        "input": "tensor",
        "size": "tuple",
        "stride": "tuple",
        "storage_offset": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.as_tensor_1"] = {
    "args": {
        "data": "list" # Could also be tuple, but list seems more general
    },
    "kwargs": {
        "dtype": "dtype",
    },
    "inner": {},
}
signatures["torch.as_tensor_2"] = {
    "args": {
        "data": "tuple" # Could also be list, but tuple seems more general
    },
    "kwargs": {
        "dtype": "dtype",
    },
    "inner": {},
}
signatures["torch.as_tensor_3"] = {
    "args": {
        "data": "tensor"
    },
    "kwargs": {
        "dtype": "dtype",
    },
    "inner": {},
}
signatures["torch.as_tensor_4"] = {
    "args": {
        "data": "integer"
    },
    "kwargs": {
        "dtype": "dtype",
    },
    "inner": {},
}
signatures["torch.as_tensor_5"] = {
    "args": {
        "data": "float"
    },
    "kwargs": {
        "dtype": "dtype",
    },
    "inner": {},
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
signatures["torch.asinh"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.atan"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.atan2"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.atanh"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.atleast_1d_1"] = {
    "args": {
        "tensors": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.atleast_1d_2"] = {
    "args": {
        "tensors": "tensor_list"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.atleast_2d_1"] = {
    "args": {
        "tensors": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.atleast_2d_2"] = {
    "args": {
        "tensors": "tensor_list"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.atleast_3d_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.atleast_3d_2"] = {
    "args": {
        "input": "tensor_list"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.functional.avg_pool1d_1"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "integer",
        "ceil_mode": "boolean",
        "count_include_pad": "boolean",
        "divisor_override": "integer"
    },
    "inner": {},
}
signatures["torch.nn.functional.avg_pool1d_2"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "tuple",
        "ceil_mode": "boolean",
        "count_include_pad": "boolean",
        "divisor_override": "integer"
    },
    "inner": {},
}
signatures["torch.nn.functional.avg_pool2d_1"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "integer",
        "ceil_mode": "boolean",
        "count_include_pad": "boolean",
        "divisor_override": "integer"
    },
    "inner": {},
}
signatures["torch.nn.functional.avg_pool2d_2"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "tuple",
        "ceil_mode": "boolean",
        "count_include_pad": "boolean",
        "divisor_override": "integer"
    },
    "inner": {},
}
signatures["torch.baddbmm"] = {
    "args": {
        "input": "tensor",
        "batch1": "tensor",
        "batch2": "tensor"
    },
    "kwargs": {
        "beta": "float", # Could also be integer
        "alpha": "float", # Could also be integer
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.nn.functional.batch_norm"] = {
    "args": {
        "input": "tensor",
        "running_mean": "tensor",
        "running_var": "tensor",
        "weight": "tensor",
        "bias": "tensor",
        "training": "boolean",
        "momentum": "float",
        "eps": "float"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.bernoulli"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "generator": "torch.Generator", # unsure if this is the correct type
        "out": "tensor"
    },
    "inner": {},
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
    "inner": {},
}
signatures["torch.bincount"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "weights": "tensor",
        "minlength": "integer"
    },
    "inner": {},
}
signatures["torch.bitwise_and"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.bitwise_not"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.bitwise_or"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.bitwise_xor_1"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.bitwise_xor_2"] = {
    "args": {
        "input": "tensor",
        "other": "integer"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.bitwise_xor_3"] = {
    "args": {
        "input": "integer",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.block_diag"] = {
    "args": {
        "tensors": "tensor_list"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.bmm"] = {
    "args": {
        "input": "tensor",
        "mat2": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.broadcast_shapes"] = {
    "args": {
        "shape1": "tuple",
        "shape2": "tuple"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.broadcast_tensors"] = {
    "args": {
        "tensors": "tensor_list"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.broadcast_to"] = {
    "args": {
        "input": "tensor",
        "size": "tuple"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.bucketize"] = {
    "args": {
        "input": "tensor",
        "boundaries": "tensor"
    },
    "kwargs": {
        "out_int32": "boolean",
        "right": "boolean",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.cartesian_prod"] = {
    "args": {
        "tensors": "tensor_list"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.cat"] = {
    "args": {
        "tensors": "tensor_list",
        "dim": "integer"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.cdist"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {
        "p": "float",
        "compute_mode": "string"
    },
    "inner": {},
}
signatures["torch.ceil"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
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
signatures["torch.chain_matmul"] = {
    "args": {
        "matrices": "tensor_list"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.cholesky"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "upper": "boolean",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.cholesky_inverse"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.cholesky_solve"] = {
    "args": {
        "input": "tensor",
        "L": "tensor"
    },
    "kwargs": {
        "upper": "boolean",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.chunk"] = {
    "args": {
        "input": "tensor",
        "chunks": "integer",
        "dim": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.clamp_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "min": "float",
        "max": "float",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.clamp_2"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "min": "tensor",
        "max": "float",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.clamp_3"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "min": "float",
        "max": "tensor",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.clamp_4"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "min": "tensor",
        "max": "tensor",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.nn.utils.clip_grad_norm__1"] = {
    "args": {
        "parameters": "tensor_list",
        "max_norm": "float"
    },
    "kwargs": {
        "norm_type": "float",
        "error_if_nonfinite": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.utils.clip_grad_norm__2"] = {
    "args": {
        "parameters": "tensor",
        "max_norm": "float"
    },
    "kwargs": {
        "norm_type": "float",
        "error_if_nonfinite": "boolean"
    },
    "inner": {},
}
signatures["torch.combinations"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "r": "integer",
        "with_replacement": "boolean"
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
signatures["torch.conj"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.functional.conv_transpose2d_1"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor",
    },
    "kwargs": {
        "bias": "tensor",
        "stride": "integer", # Could be tuple
        "padding": "integer", # Could be tuple
        "output_padding": "integer", # Could be tuple
        "groups": "integer",
        "dilation": "integer" # Could be tuple
    },
    "inner": {},
}
signatures["torch.nn.functional.conv_transpose2d_2"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor",
    },
    "kwargs": {
        "bias": "tensor",
        "stride": "tuple",
        "padding": "tuple",
        "output_padding": "tuple",
        "groups": "integer",
        "dilation": "tuple"
    },
    "inner": {},
}
signatures["torch.copysign"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"  # Or "float" if other is a Number. Assuming tensor because it says tensor or number and tensor is more specific
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.cos"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.cosh"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.nn.functional.cosine_similarity"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor",
    },
    "kwargs": {
        "dim": "integer",
        "eps": "float",
    },
    "inner": {},
}
signatures["torch.count_nonzero"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dim": "integer"
    },
    "inner": {},
}
signatures["torch.cross"] = {
    "args": {
        "input": "tensor",
        "other": "tensor",
    },
    "kwargs": {
        "dim": "integer",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.nn.functional.cross_entropy_1"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "weight": "tensor",
        "ignore_index": "integer",
        "reduction": "string",
        "label_smoothing": "float"
    },
    "inner": {},
}
signatures["torch.nn.functional.cross_entropy_2"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "weight": "tensor",
        "ignore_index": "integer",
        "reduction": "string",
        "label_smoothing": "float"
    },
    "inner": {},
}
signatures["torch.nn.functional.cross_entropy_3"] = {
    "args": {
        "input": "tensor",
        "target": "tensor" # Could be integer
    },
    "kwargs": {
        "weight": "tensor",
        "ignore_index": "integer",
        "reduction": "string",
        "label_smoothing": "float"
    },
    "inner": {},
}
signatures["torch.cummax"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {
        "out": "tuple" # Should this be tuple of tensors? Or just tensor? It seems to expect a tuple of two tensors (values, indices)
    },
    "inner": {},
}
signatures["torch.cummin"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {
        "out": "tuple" # Should be a tuple of tensors, but can't specify that
    },
    "inner": {},
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
    "inner": {},
}
signatures["torch.cumsum"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {
        "dtype": "dtype",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.deg2rad"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.det"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.diag"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "diagonal": "integer",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.diag_embed"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "offset": "integer",
        "dim1": "integer",
        "dim2": "integer"
    },
    "inner": {},
}
signatures["torch.diagflat"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "offset": "integer",
    },
    "inner": {},
}
signatures["torch.diagonal"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "offset": "integer",
        "dim1": "integer",
        "dim2": "integer"
    },
    "inner": {},
}
signatures["torch.digamma"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.dist"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "p": "float"
    },
    "inner": {},
}
signatures["torch.div_1"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "rounding_mode": "string",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.div_2"] = {
    "args": {
        "input": "tensor",
        "other": "float"  # Could also be integer, but float seems more general based on the description
    },
    "kwargs": {
        "rounding_mode": "string",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.dot"] = {
    "args": {
        "input": "tensor",
        "tensor": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.dstack"] = {
    "args": {
        "tensors": "tensor_list"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.linalg.eig"] = {
    "args": {
        "A": "tensor"
    },
    "kwargs": {
        "out": "tuple"
    },
    "inner": {},
}
signatures["torch.einsum_1"] = {
    "args": {
        "equation": "string",
        "*operands": "tensor_list"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.einsum_2"] = {
    "args": {
        "operands": "list" # list containing tensors and sublists (list of integers)
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.functional.embedding"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.embedding_bag_1"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor",
        "offsets": "tensor"
    },
    "kwargs": {
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "mode": "string",
        "sparse": "boolean",
        "per_sample_weights": "tensor",
        "include_last_offset": "boolean"
    },
    "inner": {}
}
signatures["torch.nn.functional.embedding_bag_2"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor"
    },
    "kwargs": {
        "offsets": "tensor",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "mode": "string",
        "sparse": "boolean",
        "per_sample_weights": "tensor",
        "include_last_offset": "boolean"
    },
    "inner": {}
}
signatures["torch.nn.functional.embedding_bag_3"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor",
        "indices": "tensor",
        "offsets": "tensor"
    },
    "kwargs": {
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "mode": "string",
        "sparse": "boolean",
        "per_sample_weights": "tensor",
        "include_last_offset": "boolean"
    },
    "inner": {}
}
signatures["torch.nn.functional.embedding_bag_4"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor"
    },
    "kwargs": {
        "indices": "tensor",
        "offsets": "tensor",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "mode": "string",
        "sparse": "boolean",
        "per_sample_weights": "tensor",
        "include_last_offset": "boolean"
    },
    "inner": {}
}
signatures["torch.empty_like"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dtype": "dtype",
        "layout": "string", # Best match, but maybe should be enum
        "requires_grad": "boolean",
        "memory_format": "string" # Best match, but maybe should be enum
    },
    "inner": {},
}
signatures["torch.empty_strided"] = {
    "args": {
        "size": "tuple",
        "stride": "tuple",
        "dtype": "dtype"
    },
    "kwargs": {
        "layout": "string", #string representing layout. Could be enum
        "pin_memory": "boolean",
        "requires_grad": "boolean"
    },
    "inner": {},
}
signatures["torch.eq"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.eq_2"] = {
    "args": {
        "input": "tensor",
        "other": "float"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.erf"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.erfc"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.special.erfinv"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.exp"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.exp2"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.expm1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.eye"] = {
    "args": {
        "n": "integer",
        "m": "integer"
    },
    "kwargs": {
        "out": "tensor",
        "dtype": "dtype",
        "layout": "string",
        "requires_grad": "boolean"
    },
    "inner": {},
}
signatures["torch.flatten"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "start_dim": "integer",
        "end_dim": "integer"
    },
    "inner": {}
}
signatures["torch.flip"] = {
    "args": {
        "input": "tensor",
        "dims": "list" # Could also be a tuple, creating another signature is not necessary in this case
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.fliplr"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.flipud"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.float_power"] = {
    "args": {
        "input": "tensor",
        "exponent": "float"  # or "tensor" but I'm assuming float
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
    "inner": {},
}
signatures["torch.floor_divide_1"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.floor_divide_2"] = {
    "args": {
        "input": "tensor",
        "other": "float" # or integer? Should there be one signature for each? Assuming float
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.fmin"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.frac"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.full"] = {
    "args": {
        "size": "tuple",
        "fill_value": "float" # could also be integer, but float seems more general
    },
    "kwargs": {
        "out": "tensor",
        "dtype": "dtype",
        "layout": "string",
        "requires_grad": "boolean"
    },
    "inner": {},
}
signatures["torch.full_like"] = {
    "args": {
        "input": "tensor",
        "fill_value": "float" # Can also be integer, but float is more general
    },
    "kwargs": {
        "dtype": "dtype",
        "layout": "string",
        "requires_grad": "boolean",
        "memory_format": "string"
    },
    "inner": {}
}
signatures["torch.nn.functional.hardsigmoid"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "inplace": "boolean"
    },
    "inner": {},
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
signatures["torch.ge"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"  # Could also be a float, but defaulting to tensor for consistency
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.nn.functional.gelu"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "approximate": "string"
    },
    "inner": {}
}
signatures["torch.outer"] = {
    "args": {
        "input": "tensor",
        "vec2": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.nn.GRUCell"] = {
    "args": {
        "input_size": "integer",
        "hidden_size": "integer",
    },
    "kwargs": {
        "bias": "boolean",
        "dtype": "dtype",
    },
    "inner": {
        "args": {
            "input": "tensor",
            "hidden": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.gt_1"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.gt_2"] = {
    "args": {
        "input": "tensor",
        "other": "float"
    },
    "kwargs": {
        "out": "tensor"
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
    "inner": {},
}
signatures["torch.nn.functional.hardswish"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "inplace": "boolean"
    },
    "inner": {},
}
signatures["torch.heaviside"] = {
    "args": {
        "input": "tensor",
        "values": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.histc"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "bins": "integer",
        "min": "float", # Could also be an integer, but float is more general for scalars.
        "max": "float", # Could also be an integer, but float is more general for scalars.
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.hstack"] = {
    "args": {
        "tensors": "tensor_list"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.hypot"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.i0"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.special.gammainc"] = {
    "args": {
        "a": "tensor",
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.imag"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.index_select"] = {
    "args": {
        "input": "tensor",
        "dim": "integer",
        "index": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.inner"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.nn.functional.interpolate_1"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "size": "integer",
        "scale_factor": "float",
        "mode": "string",
        "align_corners": "boolean",
        "recompute_scale_factor": "boolean",
        "antialias": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.interpolate_2"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "size": "tuple",
        "scale_factor": "float",
        "mode": "string",
        "align_corners": "boolean",
        "recompute_scale_factor": "boolean",
        "antialias": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.interpolate_3"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "size": "tuple",
        "scale_factor": "tuple",
        "mode": "string",
        "align_corners": "boolean",
        "recompute_scale_factor": "boolean",
        "antialias": "boolean"
    },
    "inner": {},
}
signatures["torch.inverse"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
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
signatures["torch.is_tensor"] = {
    "args": {
        "obj": "tensor" # Can be any object actually.
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.isclose"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "rtol": "float",
        "atol": "float",
        "equal_nan": "boolean"
    },
    "inner": {},
}
signatures["torch.isfinite"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.isinf"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.isnan"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.isneginf"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.isposinf"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.isreal"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.kron"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.kthvalue"] = {
    "args": {
        "input": "tensor",
        "k": "integer",
    },
    "kwargs": {
        "dim": "integer",
        "keepdim": "boolean",
        "out": "tuple"
    },
    "inner": {},
}
signatures["torch.nn.functional.l1_loss"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "reduction": "string"
    },
    "inner": {},
}
signatures["torch.nn.functional.layer_norm"] = {
    "args": {
        "input": "tensor",
        "normalized_shape": "tuple", # Could be a list or int as well, but tuple seems like the closest match
        "weight": "tensor",
        "bias": "tensor"
    },
    "kwargs": {
        "eps": "float",
        "elementwise_affine": "boolean"
    },
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
signatures["torch.le"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.le_2"] = {
    "args": {
        "input": "tensor",
        "other": "float" # Assuming scalar is a float
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.nn.functional.leaky_relu"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "negative_slope": "float",
        "inplace": "boolean"
    },
    "inner": {},
}
signatures["torch.lerp"] = {
    "args": {
        "input": "tensor",
        "end": "tensor",
        "weight": "float"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.lerp_1"] = {
    "args": {
        "input": "tensor",
        "end": "tensor",
        "weight": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.lgamma"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.nn.functional.linear"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor"
    },
    "kwargs": {
        "bias": "tensor"
    },
    "inner": {},
}
signatures["torch.linspace"] = {
    "args": {
        "start": "float",
        "end": "float",
        "steps": "integer"
    },
    "kwargs": {
        "out": "tensor",
        "dtype": "dtype",
        "layout": "string", # There are only a few options for layout, but string is the closest type.
        "requires_grad": "boolean"
    },
    "inner": {},
}
signatures["torch.log"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.log10"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
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
signatures["torch.log2"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
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
    },
}
signatures["torch.nn.functional.log_softmax_1"] = {
    "args": {
        "input": "tensor",
        "dim": "integer",
    },
    "kwargs": {
        "dtype": "dtype",
    },
    "inner": {},
}
signatures["torch.nn.functional.log_softmax_2"] = {
    "args": {
        "input": "tensor",
        "dim": "integer",
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.logaddexp"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.logaddexp2"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.logcumsumexp"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.logdet"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.logical_and"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.logical_not"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.logical_or"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.logical_xor"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.logit"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "eps": "float",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.nn.functional.logsigmoid"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.logspace"] = {
    "args": {
        "start": "float",
        "end": "float",
        "steps": "integer"
    },
    "kwargs": {
        "base": "float",
        "out": "tensor",
        "dtype": "dtype",
        "layout": "string", #string because it's torch.layout enum
        "requires_grad": "boolean"
    },
    "inner": {},
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
    "inner": {},
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
    "inner": {},
}
signatures["torch.logsumexp_3"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dim": "integer",
        "keepdim": "boolean",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.logsumexp_4"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dim": "tuple",
        "keepdim": "boolean",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.logsumexp_5"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dim": "list", #should be None
        "keepdim": "boolean",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.nn.functional.lp_pool1d"] = {
    "args": {
        "input": "tensor",
        "norm_type": "float",
        "kernel_size": "integer",
        "stride": "integer"
    },
    "kwargs": {
        "ceil_mode": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.lp_pool2d"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "tuple",
        "stride": "tuple",
        "padding": "tuple",
        "ceil_mode": "boolean",
        "p": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.linalg.lstsq"] = {
    "args": {
        "A": "tensor",
        "B": "tensor"
    },
    "kwargs": {
        "rcond": "float",
        "driver": "string"
    },
    "inner": {},
}
signatures["torch.lt"] = {
    "args": {
        "input": "tensor",
        "other": "tensor" # or "float", creating another signature for float
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.lt_2"] = {
    "args": {
        "input": "tensor",
        "other": "float"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.lu_solve"] = {
    "args": {
        "b": "tensor",
        "LU_data": "tensor",
        "LU_pivots": "tensor"
    },
    "kwargs": {},
    "inner": {},
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
    "inner": {},
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
signatures["torch.masked_select"] = {
    "args": {
        "input": "tensor",
        "mask": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.matmul"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.matrix_exp"] = {
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
signatures["torch.linalg.matrix_rank_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "tol": "float",
        "rtol": "float",
        "hermitian": "boolean"
    },
    "inner": {},
}
signatures["torch.linalg.matrix_rank_2"] = {
    "args": {
        "input": "tensor",
        "tol": "tensor"
    },
    "kwargs": {
        "hermitian": "boolean"
    },
    "inner": {},
}
signatures["torch.max_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.max_2"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {
        "keepdim": "boolean",
        "out": "tuple"
    },
    "inner": {},
}
signatures["torch.max_3"] = {
    "args": {
        "input": "tensor",
        "dim": "tuple"
    },
    "kwargs": {
        "keepdim": "boolean",
        "out": "tuple"
    },
    "inner": {},
}
signatures["torch.max_4"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
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
        "ceil_mode": "boolean"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.functional.max_pool2d_1"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "integer",
        "dilation": "integer",
        "return_indices": "boolean",
        "ceil_mode": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.max_pool2d_2"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "tuple",
        "dilation": "integer",
        "return_indices": "boolean",
        "ceil_mode": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.max_pool2d_3"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "tuple",
        "dilation": "integer",
        "return_indices": "boolean",
        "ceil_mode": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.max_pool2d_4"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "integer",
        "dilation": "integer",
        "return_indices": "boolean",
        "ceil_mode": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.max_pool3d_1"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "integer",
        "dilation": "integer",
        "return_indices": "boolean",
        "ceil_mode": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.max_pool3d_2"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "tuple",
        "dilation": "tuple",
        "return_indices": "boolean",
        "ceil_mode": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.max_unpool2d"] = {
    "args": {
        "input": "tensor",
        "indices": "tensor",
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "tuple",
        "output_size": "tuple"
    },
    "inner": {},
}
signatures["torch.maximum"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.mean_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {},
}
signatures["torch.mean_2"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {
        "keepdim": "boolean",
        "dtype": "dtype",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.mean_3"] = {
    "args": {
        "input": "tensor",
        "dim": "tuple"
    },
    "kwargs": {
        "keepdim": "boolean",
        "dtype": "dtype",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.median_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.median_2"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {
        "keepdim": "boolean",
        "out": "tuple" # Should be a tuple of tensors, but approximating with "tuple"
    },
    "inner": {},
}
signatures["torch.min_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.min_2"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {
        "keepdim": "boolean",
        "out": "tuple"
    },
    "inner": {},
}
signatures["torch.min_3"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.minimum"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.mm"] = {
    "args": {
        "input": "tensor",
        "mat2": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.movedim_1"] = {
    "args": {
        "input": "tensor",
        "source": "integer",
        "destination": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.movedim_2"] = {
    "args": {
        "input": "tensor",
        "source": "tuple",
        "destination": "tuple"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.functional.mse_loss"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "size_average": "boolean", # deprecated
        "reduce": "boolean", # deprecated
        "reduction": "string"
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
signatures["torch.mul_1"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.mul_2"] = {
    "args": {
        "input": "tensor",
        "other": "float" # Number could also be integer?
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.mul_3"] = {
    "args": {
        "input": "tensor",
        "other": "integer" # Number could also be integer?
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.nn.MultiheadAttention"] = {
    "args": {
        "embed_dim": "integer",
        "num_heads": "integer",
    },
    "kwargs": {
        "dropout": "float",
        "bias": "boolean",
        "add_bias_kv": "boolean",
        "add_zero_attn": "boolean",
        "kdim": "integer",
        "vdim": "integer",
        "batch_first": "boolean",
        "dtype": "dtype",
    },
    "inner": {
        "args": {
            "query": "tensor",
            "key": "tensor",
            "value": "tensor",
        },
        "kwargs": {
            "key_padding_mask": "tensor",
            "need_weights": "boolean",
            "attn_mask": "tensor",
            "average_attn_weights": "boolean",
            "is_causal": "boolean",
        }
    },
}
signatures["torch.nn.functional.multilabel_soft_margin_loss"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "weight": "tensor",
        "size_average": "boolean", #deprecated
        "reduce": "boolean", #deprecated
        "reduction": "string"
    },
    "inner": {},
}
signatures["torch.multinomial"] = {
    "args": {
        "input": "tensor",
        "num_samples": "integer"
    },
    "kwargs": {
        "replacement": "boolean",
        "generator": "torch.Generator", # This seems like an object instead of a type
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.mv"] = {
    "args": {
        "input": "tensor",
        "vec": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.special.multigammaln"] = {
    "args": {
        "x": "tensor",
        "p": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nanmedian_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nanmedian_2"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {
        "keepdim": "boolean",
        "out": "tuple" # unsure if this should be tuple, should be optional(Tensor, Tensor)
    },
    "inner": {},
}
signatures["torch.nansum_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {},
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
    "inner": {},
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
    "inner": {},
}
signatures["torch.narrow"] = {
    "args": {
        "input": "tensor",
        "dim": "integer",
        "start": "integer",
        "length": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.narrow_2"] = {
    "args": {
        "input": "tensor",
        "dim": "integer",
        "start": "tensor", # Should this be integer? But documentation says 0-dim integral Tensor
        "length": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.ne"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.ne_2"] = {
    "args": {
        "input": "tensor",
        "other": "float"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.neg"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
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
signatures["torch.nn.functional.nll_loss_1"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "weight": "tensor",
        "size_average": "boolean",
        "ignore_index": "integer",
        "reduce": "boolean",
        "reduction": "string"
    },
    "inner": {},
}
signatures["torch.nn.functional.nll_loss_2"] = {
    "args": {
        "input": "tensor",
        "target": "tensor",
        "log_target": "tensor"
    },
    "kwargs": {
        "weight": "tensor",
        "size_average": "boolean",
        "ignore_index": "integer",
        "reduce": "boolean",
        "reduction": "string"
    },
    "inner": {},
}
signatures["torch.nonzero"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor",
        "as_tuple": "boolean"
    },
    "inner": {},
}
signatures["torch.norm_1"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "p": "string", # Could also be int or float, but string ('fro', 'nuc') is also possible
        "dim": "tuple", # Could also be int or list
        "keepdim": "boolean",
        "out": "tensor",
        "dtype": "dtype"
    },
    "inner": {},
}
signatures["torch.norm_2"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "p": "float", # Could also be int or string, but string ('fro', 'nuc') is also possible
        "dim": "tuple", # Could also be int or list
        "keepdim": "boolean",
        "out": "tensor",
        "dtype": "dtype"
    },
    "inner": {},
}
signatures["torch.norm_3"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "p": "integer", # Could also be float or string, but string ('fro', 'nuc') is also possible
        "dim": "tuple", # Could also be int or list
        "keepdim": "boolean",
        "out": "tensor",
        "dtype": "dtype"
    },
    "inner": {},
}
signatures["torch.norm_4"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "p": "float", # inf
        "dim": "tuple", # Could also be int or list
        "keepdim": "boolean",
        "out": "tensor",
        "dtype": "dtype"
    },
    "inner": {},
}
signatures["torch.norm_5"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "p": "float", # -inf
        "dim": "tuple", # Could also be int or list
        "keepdim": "boolean",
        "out": "tensor",
        "dtype": "dtype"
    },
    "inner": {},
}
signatures["torch.norm_6"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "p": "string", # Could also be int or float, but string ('fro', 'nuc') is also possible
        "dim": "integer", # Could also be tuple or list
        "keepdim": "boolean",
        "out": "tensor",
        "dtype": "dtype"
    },
    "inner": {},
}
signatures["torch.norm_7"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "p": "float", # Could also be int or string, but string ('fro', 'nuc') is also possible
        "dim": "integer", # Could also be tuple or list
        "keepdim": "boolean",
        "out": "tensor",
        "dtype": "dtype"
    },
    "inner": {},
}
signatures["torch.norm_8"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "p": "integer", # Could also be float or string, but string ('fro', 'nuc') is also possible
        "dim": "integer", # Could also be tuple or list
        "keepdim": "boolean",
        "out": "tensor",
        "dtype": "dtype"
    },
    "inner": {},
}
signatures["torch.norm_9"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "p": "float", # inf
        "dim": "integer", # Could also be tuple or list
        "keepdim": "boolean",
        "out": "tensor",
        "dtype": "dtype"
    },
    "inner": {},
}
signatures["torch.norm_10"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "p": "float", # -inf
        "dim": "integer", # Could also be tuple or list
        "keepdim": "boolean",
        "out": "tensor",
        "dtype": "dtype"
    },
    "inner": {},
}
signatures["torch.norm_11"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "p": "string", # Could also be int or float, but string ('fro', 'nuc') is also possible
        "dim": "list", # Could also be tuple or int
        "keepdim": "boolean",
        "out": "tensor",
        "dtype": "dtype"
    },
    "inner": {},
}
signatures["torch.norm_12"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "p": "float", # Could also be int or string, but string ('fro', 'nuc') is also possible
        "dim": "list", # Could also be tuple or int
        "keepdim": "boolean",
        "out": "tensor",
        "dtype": "dtype"
    },
    "inner": {},
}
signatures["torch.norm_13"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "p": "integer", # Could also be float or string, but string ('fro', 'nuc') is also possible
        "dim": "list", # Could also be tuple or int
        "keepdim": "boolean",
        "out": "tensor",
        "dtype": "dtype"
    },
    "inner": {},
}
signatures["torch.norm_14"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "p": "float", # inf
        "dim": "list", # Could also be tuple or int
        "keepdim": "boolean",
        "out": "tensor",
        "dtype": "dtype"
    },
    "inner": {},
}
signatures["torch.norm_15"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "p": "float", # -inf
        "dim": "list", # Could also be tuple or int
        "keepdim": "boolean",
        "out": "tensor",
        "dtype": "dtype"
    },
    "inner": {},
}
signatures["torch.nn.functional.normalize"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "p": "float",
        "dim": "integer", # or tuple of integers, but creating separate signature below
        "eps": "float",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.numel"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.ones_1"] = {
    "args": {
        "size": "integer"
    },
    "kwargs": {
        "out": "tensor",
        "dtype": "dtype",
        "layout": "torch.layout", # Ideally this should be string but the options are limited
        "requires_grad": "boolean"
    },
    "inner": {},
}
signatures["torch.ones_2"] = {
    "args": {
        "size": "tuple"
    },
    "kwargs": {
        "out": "tensor",
        "dtype": "dtype",
        "layout": "torch.layout", # Ideally this should be string but the options are limited
        "requires_grad": "boolean"
    },
    "inner": {},
}
signatures["torch.ones_3"] = {
    "args": {
        "size": "list"
    },
    "kwargs": {
        "out": "tensor",
        "dtype": "dtype",
        "layout": "torch.layout", # Ideally this should be string but the options are limited
        "requires_grad": "boolean"
    },
    "inner": {},
}
signatures["torch.ones_like"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dtype": "dtype",
        "layout": "string", # Could also be enum
        "device": "string", # Should skip this, since this is explicitly stated.
        "requires_grad": "boolean",
        "memory_format": "string" # Could also be enum
    },
    "inner": {},
}
signatures["torch.outer"] = {
    "args": {
        "input": "tensor",
        "vec2": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.nn.functional.pad"] = {
    "args": {
        "input": "tensor",
        "pad": "tuple"
    },
    "kwargs": {
        "mode": "string",
        "value": "float"
    },
    "inner": {},
}
signatures["torch.nn.utils.rnn.pad_sequence"] = {
    "args": {
        "sequences": "tensor_list"
    },
    "kwargs": {
        "batch_first": "boolean",
        "padding_value": "float",
        "enforce_sorted": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.pairwise_distance"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {
        "p": "float",
        "eps": "float",
        "keepdim": "boolean"
    },
    "inner": {},
}
signatures["torch.pinverse"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "rcond": "float" # could also be a tensor
    },
    "inner": {},
}
signatures["torch.nn.functional.pixel_shuffle"] = {
    "args": {
        "input": "tensor",
        "upscale_factor": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.poisson"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "generator": "torch.Generator" # Ideally this would be a more specific generator type if available
    },
    "inner": {},
}
signatures["torch.nn.functional.poisson_nll_loss"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "log_input": "boolean",
        "full": "boolean",
        "eps": "float",
        "reduction": "string"
    },
    "inner": {},
}
signatures["torch.polar"] = {
    "args": {
        "abs": "tensor",
        "angle": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.polygamma"] = {
    "args": {
        "n": "integer",
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.pow_1"] = {
    "args": {
        "input": "tensor",
        "exponent": "float"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.pow_2"] = {
    "args": {
        "input": "tensor",
        "exponent": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.pow_3"] = {
    "args": {
        "self": "float",
        "exponent": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.nn.functional.prelu"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.prod_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {},
}
signatures["torch.prod_2"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {
        "keepdim": "boolean",
        "dtype": "dtype"
    },
    "inner": {},
}
signatures["torch.promote_types"] = {
    "args": {
        "type1": "dtype",
        "type2": "dtype"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.qr"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "some": "boolean",
        "out": "tuple" # could also be a tensor or list of tensors, but choosing tuple to match the example
    },
    "inner": {},
}
signatures["torch.rad2deg"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.rand"] = {
    "args": {
        "size": "integer" # Can also be a sequence of integers (list or tuple), but defining size as an integer.
    },
    "kwargs": {
        "generator": "torch.Generator", # Type torch.Generator is not in the list, but it's a specific type.
        "out": "tensor",
        "dtype": "dtype",
        "layout": "torch.layout", # Type torch.layout is not in the list, but it's a specific type.
        "requires_grad": "boolean",
        "pin_memory": "boolean"
    },
    "inner": {},
}
signatures["torch.device_1"] = {
    "args": {
        "type": "string"
    },
    "kwargs": {
        "index": "integer"
    },
    "inner": {},
}
signatures["torch.device_2"] = {
    "args": {
        "obj": "string" # Could potentially be a device object as well but string is more likely according to the documentation
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.randperm"] = {
    "args": {
        "n": "integer"
    },
    "kwargs": {
        "generator": "torch.Generator", # unclear, could also be tensor?
        "out": "tensor",
        "dtype": "dtype",
        "layout": "torch.layout", # unclear, could also be string?
        "requires_grad": "boolean",
        "pin_memory": "boolean"
    },
    "inner": {},
}
signatures["torch.range"] = {
    "args": {
        "end": "float"
    },
    "kwargs": {
        "start": "float",
        "step": "float",
        "out": "tensor",
        "dtype": "dtype",
        "layout": "string", # Could also be torch.layout, but sticking to string based on instructions
        "requires_grad": "boolean"
    },
    "inner": {},
}
signatures["torch.ravel"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.real"] = {
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
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.nn.functional.relu"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "inplace": "boolean"
    },
    "inner": {},
}
signatures["torch.remainder"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.repeat_interleave_1"] = {
    "args": {
        "input": "tensor",
        "repeats": "integer"
    },
    "kwargs": {
        "dim": "integer",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.repeat_interleave_2"] = {
    "args": {
        "input": "tensor",
        "repeats": "tensor"
    },
    "kwargs": {
        "dim": "integer",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.reshape"] = {
    "args": {
        "input": "tensor",
        "shape": "tuple"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.result_type_1"] = {
    "args": {
        "tensor1": "tensor",
        "tensor2": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.result_type_2"] = {
    "args": {
        "tensor1": "tensor",
        "tensor2": "tensor",
        "others": "tensor_list"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.roll_1"] = {
    "args": {
        "input": "tensor",
        "shifts": "integer",
    },
    "kwargs": {
        "dims": "integer"
    },
    "inner": {},
}
signatures["torch.roll_2"] = {
    "args": {
        "input": "tensor",
        "shifts": "tuple",
    },
    "kwargs": {
        "dims": "tuple"
    },
    "inner": {},
}
signatures["torch.rot90"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "k": "integer",
        "dims": "tuple" # could also be list. keeping tuple as the doc says tuple or list
    },
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
signatures["torch.nn.functional.rrelu"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "lower": "float",
        "upper": "float",
        "training": "boolean",
        "inplace": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.rrelu__1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "lower": "float",
        "upper": "float",
        "training": "boolean",
        "inplace": "boolean"
    },
    "inner": {}
}
signatures["torch.rsqrt"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.scatter"] = {
    "args": {
        "input": "tensor",
        "dim": "integer",
        "index": "tensor",
        "src": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.scatter_add"] = {
    "args": {
        "input": "tensor",
        "dim": "integer",
        "index": "tensor",
        "src": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.searchsorted"] = {
    "args": {
        "sorted_sequence": "tensor",
        "values": "tensor" # Could also be scalar, but tensor is more general
    },
    "kwargs": {
        "out_int32": "boolean",
        "right": "boolean",
        "side": "string",
        "out": "tensor",
        "sorter": "tensor"
    },
    "inner": {},
}
signatures["torch.nn.functional.selu"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "inplace": "boolean"
    },
    "inner": {},
}
signatures["torch.set_flush_denormal"] = {
    "args": {
        "mode": "boolean"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.sgn"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.sigmoid"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.sign"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
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
signatures["torch.nn.functional.silu"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "inplace": "boolean"
    },
    "inner": {},
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
signatures["torch.sinh"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.slogdet"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.functional.softmax"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {},
}
signatures["torch.nn.functional.softmin"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {},
}
signatures["torch.nn.functional.softplus"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "beta": "float",
        "threshold": "float"
    },
    "inner": {},
}
signatures["torch.nn.functional.softshrink"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "lambd": "float" # Should have been float
    },
    "inner": {},
}
signatures["torch.nn.functional.softsign"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.linalg.solve"] = {
    "args": {
        "A": "tensor",
        "B": "tensor"
    },
    "kwargs": {
        "left": "boolean",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.sort"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dim": "integer",
        "descending": "boolean",
        "stable": "boolean",
        "out": "tuple"
    },
    "inner": {},
}
signatures["torch.sparse_coo_tensor_1"] = {
    "args": {
        "indices": "tensor",
        "values": "tensor",
        "size": "tuple"
    },
    "kwargs": {
        "dtype": "dtype",
        "requires_grad": "boolean"
    },
    "inner": {},
}
signatures["torch.sparse_coo_tensor_2"] = {
    "args": {
        "indices": "tensor",
        "values": "tensor"
    },
    "kwargs": {
        "size": "tuple",
        "dtype": "dtype",
        "requires_grad": "boolean"
    },
    "inner": {},
}
signatures["torch.sparse_coo_tensor_3"] = {
    "args": {
        "size": "tuple"
    },
    "kwargs": {
        "dtype": "dtype",
        "requires_grad": "boolean"
    },
    "inner": {},
}
signatures["torch.split_1"] = {
    "args": {
        "tensor": "tensor",
        "split_size_or_sections": "integer",
        "dim": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.split_2"] = {
    "args": {
        "tensor": "tensor",
        "split_size_or_sections": "list",
        "dim": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.sqrt"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.square"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.squeeze_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.squeeze_2"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.squeeze_3"] = {
    "args": {
        "input": "tensor",
        "dim": "tuple"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.stack"] = {
    "args": {
        "tensors": "tensor_list",
    },
    "kwargs": {
        "dim": "integer",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.std_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dim": "integer",
        "correction": "integer",
        "keepdim": "boolean",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.std_2"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dim": "tuple",
        "correction": "integer",
        "keepdim": "boolean",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.std_mean_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "unbiased": "boolean",
        "keepdim": "boolean",
        "out": "tuple"
    },
    "inner": {},
}
signatures["torch.std_mean_2"] = {
    "args": {
        "input": "tensor",
        "dim": "list" # Could also be "integer" or "tuple"
    },
    "kwargs": {
        "unbiased": "boolean",
        "keepdim": "boolean",
        "out": "tuple"
    },
    "inner": {},
}
signatures["torch.std_mean_3"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {
        "unbiased": "boolean",
        "keepdim": "boolean",
        "out": "tuple"
    },
    "inner": {},
}
signatures["torch.std_mean_4"] = {
    "args": {
        "input": "tensor",
        "dim": "tuple"
    },
    "kwargs": {
        "unbiased": "boolean",
        "keepdim": "boolean",
        "out": "tuple"
    },
    "inner": {},
}
signatures["torch.sub_1"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "alpha": "float",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.sub_2"] = {
    "args": {
        "input": "tensor",
        "other": "float"
    },
    "kwargs": {
        "alpha": "float",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.sum_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {},
}
signatures["torch.sum_2"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {
        "keepdim": "boolean",
        "dtype": "dtype"
    },
    "inner": {},
}
signatures["torch.sum_3"] = {
    "args": {
        "input": "tensor",
        "dim": "tuple"
    },
    "kwargs": {
        "keepdim": "boolean",
        "dtype": "dtype"
    },
    "inner": {},
}
signatures["torch.svd"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "some": "boolean",
        "compute_uv": "boolean",
        "out": "tuple"
    },
    "inner": {},
}
signatures["torch.linalg.eigh"] = {
    "args": {
        "A": "tensor"
    },
    "kwargs": {
        "UPLO": "string",
        "out": "tuple"
    },
    "inner": {},
}
signatures["torch.t"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.tan"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.tanh"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.tensordot_1"] = {
    "args": {
        "a": "tensor",
        "b": "tensor",
    },
    "kwargs": {
        "dims": "integer",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.tensordot_2"] = {
    "args": {
        "a": "tensor",
        "b": "tensor",
    },
    "kwargs": {
        "dims": "tuple",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.tensordot_3"] = {
    "args": {
        "a": "tensor",
        "b": "tensor",
    },
    "kwargs": {
        "dims": "list",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.tensordot_4"] = {
    "args": {
        "a": "tensor",
        "b": "tensor",
    },
    "kwargs": {
        "dims": "tensor",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.topk"] = {
    "args": {
        "input": "tensor",
        "k": "integer"
    },
    "kwargs": {
        "dim": "integer",
        "largest": "boolean",
        "sorted": "boolean",
        "out": "tuple"
    },
    "inner": {},
}
signatures["torch.transpose"] = {
    "args": {
        "input": "tensor",
        "dim0": "integer",
        "dim1": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.trapz"] = {
    "args": {
        "y": "tensor",
        "x": "tensor"
    },
    "kwargs": {
        "dim": "integer"
    },
    "inner": {},
}
signatures["torch.triangular_solve"] = {
    "args": {
        "input": "tensor",
        "A": "tensor"
    },
    "kwargs": {
        "upper": "boolean",
        "transpose": "boolean",
        "unitriangular": "boolean",
        "out": "tuple"  # It can also be a single tensor according to documentation but keeping tuple to follow examples
    },
    "inner": {},
}
signatures["torch.tril"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "diagonal": "integer",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.tril_indices"] = {
    "args": {
        "row": "integer",
        "col": "integer",
        "offset": "integer"
    },
    "kwargs": {
        "dtype": "dtype",
        "layout": "string", # could be layout type
        "device": "string", # skipped
        "pin_memory": "boolean"
    },
    "inner": {},
}
signatures["torch.triu"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "diagonal": "integer",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.triu_indices"] = {
    "args": {
        "row": "integer",
        "col": "integer",
    },
    "kwargs": {
        "offset": "integer",
        "dtype": "dtype",
        "layout": "string", # Should ideally be torch.layout
    },
    "inner": {},
}
signatures["torch.trunc"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.unbind"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "dim": "integer"
    },
    "inner": {},
}
signatures["torch.unique"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "sorted": "boolean",
        "return_inverse": "boolean",
        "return_counts": "boolean",
        "dim": "integer"
    },
    "inner": {},
}
signatures["torch.unique_consecutive_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "return_inverse": "boolean",
        "return_counts": "boolean",
        "dim": "integer"
    },
    "inner": {},
}
signatures["torch.unique_consecutive_2"] = {
    "args": {
        "input": "tensor",
        "return_inverse": "boolean",
        "return_counts": "boolean",
        "dim": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.unsqueeze"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.var_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dim": "integer",
        "correction": "integer",
        "keepdim": "boolean",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.var_2"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dim": "tuple",
        "correction": "integer",
        "keepdim": "boolean",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.var_mean_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dim": "list", # Can be a single integer or a tuple of integers. Using list as it is more general.
        "unbiased": "boolean",
        "keepdim": "boolean",
        "out": "tuple"
    },
    "inner": {},
}
signatures["torch.var_mean_2"] = {
    "args": {
        "input": "tensor",
        "dim": "list"
    },
    "kwargs": {
        "unbiased": "boolean",
        "keepdim": "boolean",
        "out": "tuple"
    },
    "inner": {},
}
signatures["torch.var_mean_3"] = {
    "args": {
        "input": "tensor",
        "dim": "list",
        "unbiased": "boolean"
    },
    "kwargs": {
        "keepdim": "boolean",
        "out": "tuple"
    },
    "inner": {},
}
signatures["torch.var_mean_4"] = {
    "args": {
        "input": "tensor",
        "unbiased": "boolean"
    },
    "kwargs": {
        "dim": "list",
        "keepdim": "boolean",
        "out": "tuple"
    },
    "inner": {},
}
signatures["torch.where_1"] = {
    "args": {
        "condition": "tensor",
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.where_2"] = {
    "args": {
        "condition": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.zeros_1"] = {
    "args": {
        "size": "integer"
    },
    "kwargs": {
        "out": "tensor",
        "dtype": "dtype",
        "layout": "string", # it's an enum, but string is the closest
        "requires_grad": "boolean"
    },
    "inner": {},
}
signatures["torch.zeros_2"] = {
    "args": {
        "size": "tuple"
    },
    "kwargs": {
        "out": "tensor",
        "dtype": "dtype",
        "layout": "string", # it's an enum, but string is the closest
        "requires_grad": "boolean"
    },
    "inner": {},
}
signatures["torch.zeros_3"] = {
    "args": {
        "size": "list"
    },
    "kwargs": {
        "out": "tensor",
        "dtype": "dtype",
        "layout": "string", # it's an enum, but string is the closest
        "requires_grad": "boolean"
    },
    "inner": {},
}
signatures["torch.zeros_like"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dtype": "dtype",
        "layout": "string", # Could also be a Layout enum, but string is closest
        "requires_grad": "boolean",
        "memory_format": "string" # Could also be a MemoryFormat enum, but string is closest
    },
    "inner": {},
}
signatures["torch.arctan_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.UninitializedParameter"] = {
    "args": {
    },
    "kwargs": {
        "requires_grad": "boolean"
    },
    "inner": {}
}
signatures["torch.get_default_device"] = {
    "args": {},
    "kwargs": {},
    "inner": {},
}
signatures["torch.tile"] = {
    "args": {
        "input": "tensor",
        "dims": "tuple"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.is_autocast_cache_enabled"] = {
    "args": {},
    "kwargs": {},
    "inner": {},
}
signatures["torch.log_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.special.log_ndtr"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.view_as_complex_copy"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.take"] = {
    "args": {
        "input": "tensor",
        "index": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.is_autocast_enabled"] = {
    "args": {},
    "kwargs": {},
    "inner": {},
}
signatures["torch.swapdims"] = {
    "args": {
        "input": "tensor",
        "dim0": "integer",
        "dim1": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.linalg.lu"] = {
    "args": {
        "A": "tensor"
    },
    "kwargs": {
        "pivot": "boolean",
        "out": "tuple"
    },
    "inner": {},
}
signatures["torch.crow_indices_copy"] = {
    "args": {
        "indices": "tensor",
        "row_offsets": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.threshold"] = {
    "args": {
        "input": "tensor",
        "threshold": "float",
        "value": "float"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.fft.fft"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "n": "integer",
        "dim": "integer",
        "norm": "string",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.iinfo"] = {
    "args": {
        "dtype": "dtype"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.celu"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "alpha": "float",
        "inplace": "boolean"
    },
    "inner": {},
}
signatures["torch.set_deterministic_debug_mode"] = {
    "args": {
        "mode": "string"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.special.ndtr"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.special.spherical_bessel_j0"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.special.softmax"] = {
    "args": {
        "input": "tensor",
        "dim": "integer",
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {},
}
signatures["torch.negative_"] = {
    "args": {
        "input": "tensor",
        "out": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.abs_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.arccos"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.rsqrt_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.fft.rfft2"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "s": "tuple",
        "dim": "tuple",
        "norm": "string",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.frexp"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tuple"
    },
    "inner": {},
}
signatures["torch.special.gammainc"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.jit.Error"] = {
    "args": {
        "msg": "string"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.finfo"] = {
    "args": {
        "dtype": "dtype"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.linalg.multi_dot"] = {
    "args": {
        "tensors": "tensor_list"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.ConstantPad1d_1"] = {
    "args": {
        "padding": "integer",
        "value": "float" # Could also be an integer, but float is more general
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.ConstantPad1d_2"] = {
    "args": {
        "padding": "tuple",
        "value": "float" # Could also be an integer, but float is more general
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.cos_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.stft"] = {
    "args": {
        "input": "tensor",
        "n_fft": "integer"
    },
    "kwargs": {
        "hop_length": "integer",
        "win_length": "integer",
        "window": "tensor",
        "center": "boolean",
        "pad_mode": "string",
        "normalized": "boolean",
        "onesided": "boolean",
        "return_complex": "boolean",
        "align_to_window": "boolean" # best guess
    },
    "inner": {},
}
signatures["torch.special.i1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.empty_1"] = {
    "args": {
        "size": "integer"
    },
    "kwargs": {
        "out": "tensor",
        "dtype": "dtype",
        "layout": "string", # Assuming layout is a string representing the layout.
        "requires_grad": "boolean",
        "pin_memory": "boolean",
        "memory_format": "string" # Assuming memory_format is a string representing the format.
    },
    "inner": {},
}
signatures["torch.empty_2"] = {
    "args": {
        "size": "tuple"
    },
    "kwargs": {
        "out": "tensor",
        "dtype": "dtype",
        "layout": "string", # Assuming layout is a string representing the layout.
        "requires_grad": "boolean",
        "pin_memory": "boolean",
        "memory_format": "string" # Assuming memory_format is a string representing the format.
    },
    "inner": {},
}
signatures["torch.empty_3"] = {
    "args": {
        "size": "list"
    },
    "kwargs": {
        "out": "tensor",
        "dtype": "dtype",
        "layout": "string", # Assuming layout is a string representing the layout.
        "requires_grad": "boolean",
        "pin_memory": "boolean",
        "memory_format": "string" # Assuming memory_format is a string representing the format.
    },
    "inner": {},
}
signatures["torch.jit.optimized_execution"] = {
    "args": {
        "enabled": "boolean"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.cosh_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.view_as_complex"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.autocast_increment_nesting"] = {
    "args": {},
    "kwargs": {},
    "inner": {},
}
signatures["torch.linalg.solve_triangular"] = {
    "args": {
        "a": "tensor",
        "b": "tensor"
    },
    "kwargs": {
        "upper": "boolean",
        "transpose": "boolean",
        "unitriangular": "boolean",
        "left": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.AdaptiveAvgPool3d_1"] = {
    "args": {
        "output_size": "integer"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.AdaptiveAvgPool3d_2"] = {
    "args": {
        "output_size": "tuple"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.clear_autocast_cache"] = {
    "args": {},
    "kwargs": {},
    "inner": {},
}
signatures["torch.floor_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.narrow_copy"] = {
    "args": {
        "input": "tensor",
        "dim": "integer",
        "start": "integer",
        "length": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.is_deterministic_algorithms_warn_only_enabled"] = {
    "args": {},
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.CosineSimilarity"] = {
    "args": {},
    "kwargs": {
        "dim": "integer",
        "eps": "float"
    },
    "inner": {
        "args": {
            "input1": "tensor",
            "input2": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.set_autocast_ipu_enabled"] = {
    "args": {
        "enabled": "boolean"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.linalg.eigh"] = {
    "args": {
        "A": "tensor"
    },
    "kwargs": {
        "UPLO": "string",
        "out": "tuple"
    },
    "inner": {},
}
signatures["torch.acosh_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.argwhere"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.fft.fftn"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "s": "tuple",
        "dim": "tuple",
        "norm": "string",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.nn.UninitializedBuffer"] = {
    "args": {
        "size": "tuple"
    },
    "kwargs": {
        "dtype": "dtype",
        "layout": "string", # Could also be a Layout object, but sticking to allowed types
        "requires_grad": "boolean",
        "pin_memory": "boolean"
    },
    "inner": {}
}
signatures["torch.linalg.inv"] = {
    "args": {
        "A": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.reciprocal_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.PixelUnshuffle"] = {
    "args": {
        "downscale_factor": "integer"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.set_autocast_enabled"] = {
    "args": {
        "enabled": "boolean"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.sqrt_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.enable_grad"] = {
    "args": {
        "orig_func": "function" # Could also be NoneType, but function is the closest type available
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.set_default_device"] = {
    "args": {
        "device": "string"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.linalg.svdvals"] = {
    "args": {
        "A": "tensor"
    },
    "kwargs": {
        "driver": "string",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.nn.BatchNorm2d"] = {
    "args": {
        "num_features": "integer",
    },
    "kwargs": {
        "eps": "float",
        "momentum": "float",
        "affine": "boolean",
        "track_running_stats": "boolean",
        "dtype": "dtype",
    },
    "inner": {
        "args": {
            "input": "tensor",
        },
        "kwargs": {}
    },
}
signatures["torch.put"] = {
    "args": {
        "input": "tensor",
        "index": "tensor",
        "source": "tensor",
        "accumulate": "boolean"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.index_put_1"] = {
    "args": {
        "input": "tensor",
        "indices": "tuple",
        "values": "tensor"
    },
    "kwargs": {
        "accumulate": "boolean"
    },
    "inner": {},
}
signatures["torch.index_put_2"] = {
    "args": {
        "input": "tensor",
        "indices": "list",
        "values": "tensor"
    },
    "kwargs": {
        "accumulate": "boolean"
    },
    "inner": {},
}
signatures["torch.conj_physical"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.vander"] = {
    "args": {
        "x": "tensor",
    },
    "kwargs": {
        "N": "integer",
        "increasing": "boolean"
    },
    "inner": {},
}
signatures["torch.absolute"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.dsplit_1"] = {
    "args": {
        "input": "tensor",
        "indices_or_sections": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.dsplit_2"] = {
    "args": {
        "input": "tensor",
        "indices_or_sections": "list"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.dsplit_3"] = {
    "args": {
        "input": "tensor",
        "indices_or_sections": "tuple"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.special.modified_bessel_i1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.prepare_multiprocessing_environment"] = {
    "args": {
        "file_system": "string"
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
    "inner": {},
}
signatures["torch.set_default_tensor_type"] = {
    "args": {
        "t": "string"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.special.gammaln"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.pdist"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "p": "float"
    },
    "inner": {},
}
signatures["torch.get_file_path"] = {
    "args": {
        "url": "string"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.equal"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.get_autocast_xla_dtype"] = {
    "args": {},
    "kwargs": {},
    "inner": {}
}
signatures["torch.fake_quantize_per_channel_affine"] = {
    "args": {
        "input": "tensor",
        "scale": "tensor",
        "zero_point": "tensor",
        "quant_min": "integer",
        "quant_max": "integer",
        "ch_axis": "integer",
    },
    "kwargs": {},
    "inner": {},
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
signatures["torch.nn.KLDivLoss"] = {
    "args": {},
    "kwargs": {
        "size_average": "boolean", # Deprecated, so might be removed in the future
        "reduce": "boolean", # Deprecated, so might be removed in the future
        "reduction": "string",
        "log_target": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor",
            "target": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.ModuleDict_1"] = {
    "args": {},
    "kwargs": {
        "modules": "iterable" # Could be a mapping (dictionary) of (string: module) or an iterable of key-value pairs of type (string, module)
    },
    "inner": {},
}
signatures["torch.nn.ModuleDict_2"] = {
    "args": {
        "modules": "iterable"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.ModuleDict.clear"] = {
    "args": {},
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.ModuleDict.items"] = {
    "args": {},
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.ModuleDict.keys"] = {
    "args": {},
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.ModuleDict.pop"] = {
    "args": {
        "key": "string"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.ModuleDict.update"] = {
    "args": {
        "modules": "iterable"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.ModuleDict.values"] = {
    "args": {},
    "kwargs": {},
    "inner": {},
}
signatures["torch.special.entr"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.atanh_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.MaxUnpool3d_1"] = {
    "args": {
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "integer"
    },
    "inner": {
        "args": {
            "input": "tensor",
            "indices": "tensor"
        },
        "kwargs": {
            "output_size": "tuple"
        }
    },
}
signatures["torch.nn.MaxUnpool3d_2"] = {
    "args": {
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "tuple"
    },
    "inner": {
        "args": {
            "input": "tensor",
            "indices": "tensor"
        },
        "kwargs": {
            "output_size": "tuple"
        }
    },
}
signatures["torch.get_device"] = {
    "args": {
        "obj": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.erf_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.flatten"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "start_dim": "integer",
        "end_dim": "integer"
    },
    "inner": {},
}
signatures["torch.index_add"] = {
    "args": {
        "input": "tensor",
        "dim": "integer",
        "index": "tensor",
        "source": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.is_autocast_cpu_enabled"] = {
    "args": {},
    "kwargs": {},
    "inner": {}
}
signatures["torch.diff"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "n": "integer",
        "dim": "integer",
        "prepend": "tensor",
        "append": "tensor",
        "out": "tensor" # Should this be a return type instead?
    },
    "inner": {},
}
signatures["torch.igammac"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.gather"] = {
    "args": {
        "input": "tensor",
        "dim": "integer",
        "index": "tensor" # Should it be LongTensor?
    },
    "kwargs": {
        "sparse_grad": "boolean",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.hamming_window"] = {
    "args": {
        "window_length": "integer"
    },
    "kwargs": {
        "periodic": "boolean",
        "alpha": "float",
        "beta": "float",
        "dtype": "dtype",
        "layout": "string", # Should be torch.layout, but no such type exists.
        "requires_grad": "boolean"
    },
    "inner": {},
}
signatures["torch.arctanh_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.randn_like"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dtype": "dtype",
        "layout": "string", # unsure what type to put here
        "requires_grad": "boolean",
        "memory_format": "string" # unsure what type to put here
    },
    "inner": {},
}
signatures["torch.negative"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.fft.rfftfreq"] = {
    "args": {
        "n": "integer"
    },
    "kwargs": {
        "d": "float",
        "out": "tensor",
        "dtype": "dtype",
        "layout": "string", # or maybe enum?
        "requires_grad": "boolean"
    },
    "inner": {},
}
signatures["torch.sin_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.special.xlog1py"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.arcsinh"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.nn.LazyInstanceNorm1d"] = {
    "args": {},
    "kwargs": {
        "eps": "float",
        "momentum": "float", # Optional[float] but float is the most suitable type
        "affine": "boolean",
        "track_running_stats": "boolean",
        "dtype": "dtype"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.utils.vector_to_parameters"] = {
    "args": {
        "vector": "tensor",
        "parameters": "tensor_list"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.from_dlpack"] = {
    "args": {
        "x": "tensor" # any object implementing the DLPack protocol
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.GaussianNLLLoss"] = {
    "args": {},
    "kwargs": {
        "full": "boolean",
        "eps": "float",
        "reduction": "string"
    },
    "inner": {
        "args": {
            "input": "tensor",
            "target": "tensor",
            "var": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.native_channel_shuffle"] = {
    "args": {
        "input": "tensor",
        "groups": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.arctan"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.nn.Unfold_1"] = {
    "args": {
        "kernel_size": "integer"
    },
    "kwargs": {
        "dilation": "integer",
        "padding": "integer",
        "stride": "integer"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.Unfold_2"] = {
    "args": {
        "kernel_size": "tuple"
    },
    "kwargs": {
        "dilation": "tuple",
        "padding": "tuple",
        "stride": "tuple"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.fft.fftfreq"] = {
    "args": {
        "n": "integer"
    },
    "kwargs": {
        "d": "float",
        "out": "tensor",
        "dtype": "dtype",
        "layout": "string", # It is actually torch.layout
        "requires_grad": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.TripletMarginLoss"] = {
    "args": {},
    "kwargs": {
        "margin": "float",
        "p": "integer",
        "eps": "float",
        "swap": "boolean",
        "size_average": "boolean", # deprecated
        "reduce": "boolean", # deprecated
        "reduction": "string"
    },
    "inner": {
        "args": {
            "anchor": "tensor",
            "positive": "tensor",
            "negative": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.less_equal_1"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.less_equal_2"] = {
    "args": {
        "input": "tensor",
        "other": "float"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.special.bessel_y0"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.addmv_"] = {
    "args": {
        "input": "tensor",
        "mat": "tensor",
        "vec": "tensor"
    },
    "kwargs": {
        "beta": "float",
        "alpha": "float"
    },
    "inner": {},
}
signatures["torch.get_default_dtype"] = {
    "args": {},
    "kwargs": {},
    "inner": {},
}
signatures["torch.hspmm"] = {
    "args": {
        "mat1": "tensor",
        "mat2": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.bilinear"] = {
    "args": {
        "input1": "tensor",
        "input2": "tensor",
        "weight": "tensor",
        "bias": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.linalg.cholesky_ex"] = {
    "args": {
        "A": "tensor"
    },
    "kwargs": {
        "upper": "boolean",
        "check_errors": "boolean"
    },
    "inner": {},
}
signatures["torch.linalg.tensorinv"] = {
    "args": {
        "A": "tensor",
        "ind": "integer"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
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
    },
}
signatures["torch.set_grad_enabled"] = {
    "args": {
        "mode": "boolean"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.RMSNorm_1"] = {
    "args": {
        "normalized_shape": "integer"
    },
    "kwargs": {
        "eps": "float",
        "elementwise_affine": "boolean",
        "dtype": "dtype"
    },
    "inner": {
        "args": {
            "x": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.RMSNorm_2"] = {
    "args": {
        "normalized_shape": "list"
    },
    "kwargs": {
        "eps": "float",
        "elementwise_affine": "boolean",
        "dtype": "dtype"
    },
    "inner": {
        "args": {
            "x": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.RMSNorm_3"] = {
    "args": {
        "normalized_shape": "tuple"
    },
    "kwargs": {
        "eps": "float",
        "elementwise_affine": "boolean",
        "dtype": "dtype"
    },
    "inner": {
        "args": {
            "x": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.is_inference"] = {
    "args": {},
    "kwargs": {},
    "inner": {},
}
signatures["torch.special.multigammaln"] = {
    "args": {
        "input": "tensor",
        "p": "integer"
    },
    "kwargs": {
        "quiet": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.SmoothL1Loss"] = {
    "args": {},
    "kwargs": {
        "size_average": "boolean", # deprecated
        "reduce": "boolean", # deprecated
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
signatures["torch.nn.Parameter"] = {
    "args": {
        "data": "tensor"
    },
    "kwargs": {
        "requires_grad": "boolean"
    },
    "inner": {}
}
signatures["torch.ldexp_"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.DoubleStorage_1"] = {
    "args": {},
    "kwargs": {},
    "inner": {},
}
signatures["torch.DoubleStorage_2"] = {
    "args": {
        "size": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.DoubleStorage_3"] = {
    "args": {
        "source": "list" # Could also be tuple, but list is the closest option
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.DoubleStorage_4"] = {
    "args": {
        "source": "tuple" # Could also be list, but tuple is closest
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.is_anomaly_check_nan_enabled"] = {
    "args": {},
    "kwargs": {},
    "inner": {},
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
signatures["torch.arcsin_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.square_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.true_divide"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.relu_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.MaxUnpool1d_1"] = {
    "args": {
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "integer"
    },
    "inner": {
        "args": {
            "input": "tensor",
            "indices": "tensor"
        },
        "kwargs": {
            "output_size": "tuple" # Could also be an integer list representing the output size
        }
    }
}
signatures["torch.nn.MaxUnpool1d_2"] = {
    "args": {
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "integer"
    },
    "inner": {
        "args": {
            "input": "tensor",
            "indices": "tensor"
        },
        "kwargs": {
            "output_size": "tuple" # Could also be an integer list representing the output size
        }
    }
}
signatures["torch.nn.MaxUnpool1d_3"] = {
    "args": {
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "integer"
    },
    "inner": {
        "args": {
            "input": "tensor",
            "indices": "tensor"
        },
        "kwargs": {
            "output_size": "tuple" # Could also be an integer list representing the output size
        }
    }
}
signatures["torch.nn.MaxUnpool1d_4"] = {
    "args": {
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "integer"
    },
    "inner": {
        "args": {
            "input": "tensor",
            "indices": "tensor"
        },
        "kwargs": {
            "output_size": "tuple" # Could also be an integer list representing the output size
        }
    }
}
signatures["torch.linalg.pinv"] = {
    "args": {
        "A": "tensor"
    },
    "kwargs": {
        "atol": "float",
        "rtol": "float",
        "hermitian": "boolean",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.special.erfcx"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.manual_seed"] = {
    "args": {
        "seed": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.special.bessel_j1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.subtract"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "alpha": "float",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.parse_schema"] = {
    "args": {
        "schema_string": "string",
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.CrossEntropyLoss"] = {
    "args": {},
    "kwargs": {
        "weight": "tensor",
        "size_average": "boolean", # Deprecated
        "ignore_index": "integer",
        "reduce": "boolean", # Deprecated
        "reduction": "string",
        "label_smoothing": "float"
    },
    "inner": {
        "args": {
            "input": "tensor",
            "target": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.tanh"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.jit.ignore"] = {
    "args": {},
    "kwargs": {
        "drop": "boolean"
    },
    "inner": {}
}
signatures["torch.jit.strict_fusion"] = {
    "args": {},
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.ReflectionPad3d_1"] = {
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
signatures["torch.nn.ReflectionPad3d_2"] = {
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
signatures["torch.nn.LocalResponseNorm"] = {
    "args": {
        "size": "integer"
    },
    "kwargs": {
        "alpha": "float",
        "beta": "float",
        "k": "float"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.unravel_index_1"] = {
    "args": {
        "indices": "tensor",
        "shape": "tuple"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.unravel_index_2"] = {
    "args": {
        "indices": "tensor",
        "shape": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.arctanh"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.special.i0e"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.Sigmoid"] = {
    "args": {},
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.corrcoef"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.fft.ifftshift"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dim": "integer" # Could also be a tuple
    },
    "inner": {},
}
signatures["torch.fft.ifftshift_2"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dim": "tuple"
    },
    "inner": {},
}
signatures["torch.unsafe_split_with_sizes"] = {
    "args": {
        "input": "tensor",
        "split_sizes": "list", # could be tuple
        "dim": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.FractionalMaxPool3d_1"] = {
    "args": {
        "kernel_size": "integer"
    },
    "kwargs": {
        "output_size": "tuple",
        "output_ratio": "tuple",
        "return_indices": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.FractionalMaxPool3d_2"] = {
    "args": {
        "kernel_size": "tuple"
    },
    "kwargs": {
        "output_size": "tuple",
        "output_ratio": "tuple",
        "return_indices": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.FractionalMaxPool3d_3"] = {
    "args": {
        "kernel_size": "integer"
    },
    "kwargs": {
        "output_size": "integer",
        "output_ratio": "tuple",
        "return_indices": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.FractionalMaxPool3d_4"] = {
    "args": {
        "kernel_size": "tuple"
    },
    "kwargs": {
        "output_size": "integer",
        "output_ratio": "tuple",
        "return_indices": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.FractionalMaxPool3d_5"] = {
    "args": {
        "kernel_size": "integer"
    },
    "kwargs": {
        "output_size": "tuple",
        "output_ratio": "float",
        "return_indices": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.FractionalMaxPool3d_6"] = {
    "args": {
        "kernel_size": "tuple"
    },
    "kwargs": {
        "output_size": "tuple",
        "output_ratio": "float",
        "return_indices": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.FractionalMaxPool3d_7"] = {
    "args": {
        "kernel_size": "integer"
    },
    "kwargs": {
        "output_size": "integer",
        "output_ratio": "float",
        "return_indices": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.FractionalMaxPool3d_8"] = {
    "args": {
        "kernel_size": "tuple"
    },
    "kwargs": {
        "output_size": "integer",
        "output_ratio": "float",
        "return_indices": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.expand_copy"] = {
    "args": {
        "self": "tensor",
        "size": "tuple"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.is_autocast_xla_enabled"] = {
    "args": {},
    "kwargs": {},
    "inner": {},
}
signatures["torch.get_autocast_cpu_dtype"] = {
    "args": {},
    "kwargs": {},
    "inner": {}
}
signatures["torch.concat"] = {
    "args": {
        "tensors": "tensor_list",
        "dim": "integer"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.nn.CircularPad1d_1"] = {
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
signatures["torch.nn.CircularPad1d_2"] = {
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
signatures["torch.smm"] = {
    "args": {
        "input": "tensor",
        "mat": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.linalg.solve_ex"] = {
    "args": {
        "A": "tensor",
        "B": "tensor"
    },
    "kwargs": {
        "left": "boolean",
        "check_errors": "boolean"
    },
    "inner": {},
}
signatures["torch.jit.wait"] = {
    "args": {
        "future": "list"  # torch.jit.Future[T] is like a list of futures
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.fake_quantize_per_tensor_affine"] = {
    "args": {
        "input": "tensor",
        "scale": "float",
        "zero_point": "integer",
        "quant_min": "integer",
        "quant_max": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.aminmax"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dim": "integer",
        "keepdim": "boolean",
        "out": "tuple" # Could also be a tuple of tensors
    },
    "inner": {},
}
signatures["torch.nn.Threshold"] = {
    "args": {
        "threshold": "float",
        "value": "float",
    },
    "kwargs": {
        "inplace": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.linalg.vecdot"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "dim": "integer",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.is_storage"] = {
    "args": {
        "obj": "tensor" # or "list" or "tuple"? The documentation says "object", but it seems to be checking if it's a storage associated with a tensor
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.fft.fftshift_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dim": "integer"
    },
    "inner": {},
}
signatures["torch.fft.fftshift_2"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dim": "tuple"
    },
    "inner": {},
}
signatures["torch.nn.LazyInstanceNorm2d"] = {
    "args": {},
    "kwargs": {
        "eps": "float",
        "momentum": "float",
        "affine": "boolean",
        "track_running_stats": "boolean",
        "dtype": "dtype"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.jit.script_if_tracing"] = {
    "args": {
        "fn": "list", # Could be callable, but list seems more appropriate based on context
        "alternative_fn": "list" # Could be callable, but list seems more appropriate based on context
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.greater"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.autocast_decrement_nesting"] = {
    "args": {},
    "kwargs": {},
    "inner": {},
}
signatures["torch.meshgrid"] = {
    "args": {
        "tensors": "tensor_list"
    },
    "kwargs": {
        "indexing": "string"
    },
    "inner": {},
}
signatures["torch.special.hermite_polynomial_he"] = {
    "args": {
        "x": "tensor",
        "n": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.no_grad"] = {
    "args": {
        "orig_func": "function" # Best guess for the type of a function
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.miopen_batch_norm"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor",
        "bias": "tensor",
        "running_mean": "tensor",
        "running_var": "tensor",
        "training": "boolean",
        "exponential_average_factor": "float",
        "eps": "float"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.fmax"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
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
signatures["torch.nn.utils.get_total_norm_1"] = {
    "args": {
        "parameters": "tensor_list",
    },
    "kwargs": {
        "norm_type": "float",
    },
    "inner": {},
}
signatures["torch.nn.utils.get_total_norm_2"] = {
    "args": {
        "parameters": "tensor_list",
    },
    "kwargs": {
        "norm_type": "integer",
    },
    "inner": {},
}
signatures["torch.moveaxis_1"] = {
    "args": {
        "input": "tensor",
        "source": "integer",
        "destination": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.moveaxis_2"] = {
    "args": {
        "input": "tensor",
        "source": "tuple",
        "destination": "tuple"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.jit.ScriptWarning"] = {
    "args": {
        "msg": "string"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.from_numpy"] = {
    "args": {
        "ndarray": "list" # Should it be numpy.ndarray instead of list?
    },
    "kwargs": {},
    "inner": {},
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
    },
}
signatures["torch.nan_to_num"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "nan": "float",
        "posinf": "float",
        "neginf": "float",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.special.modified_bessel_i0"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.ReLU"] = {
    "args": {},
    "kwargs": {
        "inplace": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor" # Input tensor of any shape
        },
        "kwargs": {}
    },
}
signatures["torch.vstack"] = {
    "args": {
        "tensors": "tensor_list"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.nn.Unflatten"] = {
    "args": {
        "dim": "integer",
        "unflattened_size": "tuple"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.sspaddmm"] = {
    "args": {
        "input": "tensor",
        "mat1": "tensor",
        "mat2": "tensor"
    },
    "kwargs": {
        "beta": "float", # Number could be float or integer
        "alpha": "float", # Number could be float or integer
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.jit.set_fusion_strategy"] = {
    "args": {
        "value": "string"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.is_warn_always_enabled"] = {
    "args": {},
    "kwargs": {},
    "inner": {}
}
signatures["torch.linalg.eigvals"] = {
    "args": {
        "A": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.get_deterministic_debug_mode"] = {
    "args": {},
    "kwargs": {},
    "inner": {},
}
signatures["torch.special.gammaincc"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
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
signatures["torch.is_autocast_ipu_enabled"] = {
    "args": {},
    "kwargs": {},
    "inner": {},
}
signatures["torch.select_copy"] = {
    "args": {
        "input": "tensor",
        "dim": "integer",
        "index": "integer",
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.arcsinh_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.utils.remove_weight_norm"] = {
    "args": {
        "module": "tensor", # Should be a module, but no module type exists
        "name": "string",
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.all_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.all_2"] = {
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
signatures["torch.all_3"] = {
    "args": {
        "input": "tensor",
        "dim": "tuple"
    },
    "kwargs": {
        "keepdim": "boolean",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.nn.utils.parameters_to_vector"] = {
    "args": {
        "parameters": "tensor_list"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.Tanhshrink"] = {
    "args": {},
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.set_autocast_cpu_dtype"] = {
    "args": {
        "dtype": "dtype"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.fft.fft2"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "s": "tuple",
        "dim": "tuple",
        "norm": "string",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.nn.ZeroPad1d_1"] = {
    "args": {
        "padding": "integer"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.ZeroPad1d_2"] = {
    "args": {
        "padding": "tuple"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.scatter_reduce_1"] = {
    "args": {
        "input": "tensor",
        "dim": "integer",
        "index": "tensor",
        "src": "tensor",
        "reduce": "string"
    },
    "kwargs": {
        "include_self": "boolean"
    },
    "inner": {},
}
signatures["torch.scatter_reduce_2"] = {
    "args": {
        "input": "tensor",
        "dim": "integer",
        "index": "tensor",
        "src": "tensor",
        "reduce": "string",
        "output_size": "integer" # Or "tuple" if it must be a tuple.
    },
    "kwargs": {
        "include_self": "boolean"
    },
    "inner": {},
}
signatures["torch.scatter_reduce_3"] = {
    "args": {
        "input": "tensor",
        "dim": "integer",
        "index": "tensor",
        "src": "tensor",
        "reduce": "string",
        "output_size": "tuple" # Or "tuple" if it must be a tuple.
    },
    "kwargs": {
        "include_self": "boolean"
    },
    "inner": {},
}
signatures["torch.sym_float"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.special.psi"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.nn.HingeEmbeddingLoss"] = {
    "args": {},
    "kwargs": {
        "margin": "float",
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
signatures["torch.atan_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.is_inference_mode_enabled"] = {
    "args": {},
    "kwargs": {},
    "inner": {},
}
signatures["torch.greater_equal"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.less"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.vdot"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.ceil_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.CELU"] = {
    "args": {},
    "kwargs": {
        "alpha": "float",
        "inplace": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.parse_type_comment"] = {
    "args": {
        "comment": "string"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.jit.enable_onednn_fusion"] = {
    "args": {
        "mode": "boolean"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.hsplit_1"] = {
    "args": {
        "input": "tensor",
        "indices_or_sections": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.hsplit_2"] = {
    "args": {
        "input": "tensor",
        "indices_or_sections": "list"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.hsplit_3"] = {
    "args": {
        "input": "tensor",
        "indices_or_sections": "tuple"
    },
    "kwargs": {},
    "inner": {},
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
signatures["torch.concatenate"] = {
    "args": {
        "tensors": "tensor_list"
    },
    "kwargs": {
        "axis": "integer",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.jit.set_module"] = {
    "args": {
        "mod": "string",
        "new_module": "string" # Should it be module?
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.bitwise_left_shift"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.jit.CompilationUnit"] = {
    "args": {},
    "kwargs": {},
    "inner": {},
}
signatures["torch.get_rng_state"] = {
    "args": {},
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.SyncBatchNorm"] = {
    "args": {
        "num_features": "integer"
    },
    "kwargs": {
        "eps": "float",
        "momentum": "float",
        "affine": "boolean",
        "track_running_stats": "boolean",
        "process_group": "list", # Assuming process_group is some kind of list. Could be a custom object.
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["torch.nn.SyncBatchNorm.convert_sync_batchnorm"] = {
    "args": {
        "module": "tensor" # Assuming module is a tensor of some kind. Could be nn.Module.
    },
    "kwargs": {
        "process_group": "list" # Assuming process_group is some kind of list. Could be a custom object.
    },
    "inner": {}
}
signatures["torch.nn.LazyBatchNorm1d"] = {
    "args": {},
    "kwargs": {
        "eps": "float",
        "momentum": "float", # Optional[float] is represented as float
        "affine": "boolean",
        "track_running_stats": "boolean",
        "dtype": "dtype"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.Tanh"] = {
    "args": {},
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.view_as_real"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.column_stack"] = {
    "args": {
        "tensors": "tensor_list"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.set_autocast_cache_enabled"] = {
    "args": {
        "enabled": "boolean"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.row_stack"] = {
    "args": {
        "tensors": "tensor_list"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.adjoint"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.get_num_threads"] = {
    "args": {},
    "kwargs": {},
    "inner": {},
}
signatures["torch.divide"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "rounding_mode": "string", # Could be an enum but string is closest
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.sparse_bsr_tensor_1"] = {
    "args": {
        "compressed_indices": "tensor",
        "plain_indices": "tensor",
        "values": "tensor",
        "size": "tuple",
        "blocksize": "tuple"
    },
    "kwargs": {
        "dtype": "dtype",
        "requires_grad": "boolean"
    },
    "inner": {},
}
signatures["torch.sparse_bsr_tensor_2"] = {
    "args": {
        "compressed_indices": "tensor",
        "plain_indices": "tensor",
        "values": "tensor",
        "size": "tuple",
        "blocksize": "tuple"
    },
    "kwargs": {
        "dtype": "dtype",
        "requires_grad": "boolean",
        "layout": "string" # Not documented
    },
    "inner": {},
}
signatures["torch.sparse_bsr_tensor_3"] = {
    "args": {
        "compressed_indices": "tensor",
        "plain_indices": "tensor",
        "values": "tensor",
        "size": "list",
        "blocksize": "tuple"
    },
    "kwargs": {
        "dtype": "dtype",
        "requires_grad": "boolean"
    },
    "inner": {},
}
signatures["torch.sparse_bsr_tensor_4"] = {
    "args": {
        "compressed_indices": "tensor",
        "plain_indices": "tensor",
        "values": "tensor",
        "size": "list",
        "blocksize": "tuple"
    },
    "kwargs": {
        "dtype": "dtype",
        "requires_grad": "boolean",
        "layout": "string" # Not documented
    },
    "inner": {},
}
signatures["torch.swapaxes"] = {
    "args": {
        "input": "tensor",
        "axis0": "integer",
        "axis1": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.conj_physical_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.are_deterministic_algorithms_enabled"] = {
    "args": {},
    "kwargs": {},
    "inner": {},
}
signatures["torch.is_anomaly_enabled"] = {
    "args": {},
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.Mish"] = {
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
signatures["torch.nn.BCELoss"] = {
    "args": {},
    "kwargs": {
        "weight": "tensor",
        "size_average": "boolean", # deprecated
        "reduce": "boolean", # deprecated
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
signatures["torch.select"] = {
    "args": {
        "input": "tensor",
        "dim": "integer",
        "index": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.quantile_1"] = {
    "args": {
        "input": "tensor",
        "q": "float"
    },
    "kwargs": {
        "dim": "integer",
        "keepdim": "boolean",
        "interpolation": "string",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.quantile_2"] = {
    "args": {
        "input": "tensor",
        "q": "tensor"
    },
    "kwargs": {
        "dim": "integer",
        "keepdim": "boolean",
        "interpolation": "string",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.vsplit_1"] = {
    "args": {
        "input": "tensor",
        "indices_or_sections": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.vsplit_2"] = {
    "args": {
        "input": "tensor",
        "indices_or_sections": "list"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.vsplit_3"] = {
    "args": {
        "input": "tensor",
        "indices_or_sections": "tuple"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.rsub"] = {
    "args": {
        "input": "tensor",
        "other": "tensor",
    },
    "kwargs": {
        "alpha": "float",
        "out": "tensor",
    },
    "inner": {},
}
signatures["torch.logit_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "eps": "float"
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
signatures["torch.native_dropout"] = {
    "args": {
        "input": "tensor",
        "p": "float",
        "training": "boolean"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.spmm"] = {
    "args": {
        "input": "tensor",
        "mat2": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.jit.is_tracing"] = {
    "args": {},
    "kwargs": {},
    "inner": {},
}
signatures["torch.sigmoid_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.arccosh"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.histogram_1"] = {
    "args": {
        "input": "tensor",
        "bins": "integer"
    },
    "kwargs": {
        "range": "tuple",
        "weight": "tensor",
        "density": "boolean",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.histogram_2"] = {
    "args": {
        "input": "tensor",
        "bins": "tensor"
    },
    "kwargs": {
        "range": "tuple",
        "weight": "tensor",
        "density": "boolean",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.special.ndtri"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.ldexp"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.ones_like"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dtype": "dtype",
        "layout": "string", # could also be a Layout object, but string seems closest
        "requires_grad": "boolean",
        "memory_format": "string" # could also be a MemoryFormat object
    },
    "inner": {},
}
signatures["torch.jit.is_scripting"] = {
    "args": {},
    "kwargs": {},
    "inner": {},
}
signatures["torch.vitals_enabled"] = {
    "args": {},
    "kwargs": {},
    "inner": {}
}
signatures["torch.hann_window_1"] = {
    "args": {
        "window_length": "integer"
    },
    "kwargs": {
        "periodic": "boolean",
        "dtype": "dtype",
        "layout": "string", # Best guess
        "requires_grad": "boolean"
    },
    "inner": {},
}
signatures["torch.hann_window_2"] = {
    "args": {
        "window_length": "integer",
        "periodic": "boolean"
    },
    "kwargs": {
        "dtype": "dtype",
        "layout": "string", # Best guess
        "requires_grad": "boolean"
    },
    "inner": {},
}
signatures["torch.set_autocast_ipu_dtype"] = {
    "args": {
        "dtype": "dtype"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.erfc_"] = {
    "args": {
        "input": "tensor"
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
signatures["torch.special.modified_bessel_k0"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.typename"] = {
    "args": {
        "obj": "tensor"  # Could also be other objects, but tensor is the most common
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.tan_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.jit.isinstance"] = {
    "args": {
        "obj": "list", # Could be a more generic object, but the example uses a list.
        "target_type": "list" # Could be a type, not available as a type.
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.count_nonzero"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dim": "integer"
    },
    "inner": {},
}
signatures["torch.nn.SoftMarginLoss"] = {
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
signatures["torch.set_default_dtype"] = {
    "args": {
        "d": "dtype"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.ParameterList"] = {
    "args": {},
    "kwargs": {
        "values": "list" # Should this be a list of tensors or parameters? Assuming list for now
    },
    "inner": {},
}
signatures["torch.nn.ParameterList.append"] = {
    "args": {
        "value": "tensor" # Assuming a tensor can be converted to a Parameter
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.ParameterList.extend"] = {
    "args": {
        "values": "list" # Assuming a list of tensors that can be converted to parameters
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.autocast_1"] = {
    "args": {
        "device_type": "string",
    },
    "kwargs": {
        "enabled": "boolean"
    },
    "inner": {
        "args": {},
        "kwargs": {}
    }
}
signatures["torch.autocast_2"] = {
    "args": {
        "device_type": "string",
        "dtype": "dtype"
    },
    "kwargs": {
        "enabled": "boolean"
    },
    "inner": {
        "args": {},
        "kwargs": {}
    }
}
signatures["torch.fix"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.nn.Hardsigmoid"] = {
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
signatures["torch.channel_shuffle"] = {
    "args": {
        "input": "tensor",
        "groups": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.CosineEmbeddingLoss"] = {
    "args": {},
    "kwargs": {
        "margin": "float",
        "size_average": "boolean",
        "reduce": "boolean",
        "reduction": "string"
    },
    "inner": {
        "args": {
            "input1": "tensor",
            "input2": "tensor",
            "target": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.acos_"] = {
    "args": {
        "input": "tensor"
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
signatures["torch.bitwise_right_shift"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.sym_fresh_size"] = {
    "args": {},
    "kwargs": {},
    "inner": {}
}
signatures["torch.clip_"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "min": "float", # Could also be tensor or None
        "max": "float" # Could also be tensor or None
    },
    "inner": {},
}
signatures["torch.nn.Identity"] = {
    "args": {
        "args": "list", # Could be any type, but list is a safe bet
    },
    "kwargs": {
        "kwargs": "list", # Could be any type, but list is a safe bet
    },
    "inner": {
        "args": {
            "input": "tensor",
        },
        "kwargs": {}
    },
}
signatures["torch.special.bessel_y1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.trunc_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.bartlett_window"] = {
    "args": {
        "n": "integer"
    },
    "kwargs": {
        "periodic": "boolean",
        "dtype": "dtype",
        "requires_grad": "boolean"
    },
    "inner": {},
}
signatures["torch.index_copy"] = {
    "args": {
        "input": "tensor",
        "dim": "integer",
        "index": "tensor",
        "source": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.clip"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "min": "float", # Could be tensor or float, but choosing float based on examples
        "max": "float", # Could be tensor or float, but choosing float based on examples
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.nn.ZeroPad2d_1"] = {
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
signatures["torch.nn.ZeroPad2d_2"] = {
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
signatures["torch.set_num_threads"] = {
    "args": {
        "threads": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.DataParallel"] = {
    "args": {
        "module": "Module" # Assuming 'Module' is a custom class, representing a nn.Module
    },
    "kwargs": {
        "device_ids": "list",
        "output_device": "integer", # or "torch.device", but choosing integer since the example uses int
        "dim": "integer"
    },
    "inner": {}
}
signatures["torch.any_1"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
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
    "inner": {},
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
    "inner": {},
}
signatures["torch.linalg.vector_norm_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "ord": "integer", # Could also be 'string', but 'integer' is the default case and covers most use cases.
        "dim": "integer", # Could also be 'list' or 'tuple' of integers. Creating separate signatures would be necessary.
        "keepdim": "boolean",
        "dtype": "dtype",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.linalg.vector_norm_2"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "ord": "float", # Could also be 'string', but 'float' is one possibility.
        "dim": "integer", # Could also be 'list' or 'tuple' of integers. Creating separate signatures would be necessary.
        "keepdim": "boolean",
        "dtype": "dtype",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.linalg.vector_norm_3"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "ord": "string", # Could also be 'integer' or 'float', but 'string' is one possibility.
        "dim": "integer", # Could also be 'list' or 'tuple' of integers. Creating separate signatures would be necessary.
        "keepdim": "boolean",
        "dtype": "dtype",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.linalg.vector_norm_4"] = {
    "args": {
        "input": "tensor",
        "ord": "integer",
        "dim": "list",
        "keepdim": "boolean",
        "dtype": "dtype",
        "out": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.linalg.vector_norm_5"] = {
    "args": {
        "input": "tensor",
        "ord": "integer",
        "dim": "tuple",
        "keepdim": "boolean",
        "dtype": "dtype",
        "out": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.linalg.vector_norm_6"] = {
    "args": {
        "input": "tensor",
        "ord": "float",
        "dim": "list",
        "keepdim": "boolean",
        "dtype": "dtype",
        "out": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.linalg.vector_norm_7"] = {
    "args": {
        "input": "tensor",
        "ord": "float",
        "dim": "tuple",
        "keepdim": "boolean",
        "dtype": "dtype",
        "out": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.linalg.vector_norm_8"] = {
    "args": {
        "input": "tensor",
        "ord": "string",
        "dim": "list",
        "keepdim": "boolean",
        "dtype": "dtype",
        "out": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.linalg.vector_norm_9"] = {
    "args": {
        "input": "tensor",
        "ord": "string",
        "dim": "tuple",
        "keepdim": "boolean",
        "dtype": "dtype",
        "out": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.LazyBatchNorm2d"] = {
    "args": {},
    "kwargs": {
        "eps": "float",
        "momentum": "float", # Optional[float] but can be None, so I'm choosing float
        "affine": "boolean",
        "track_running_stats": "boolean",
        "dtype": "dtype"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.clone"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "memory_format": "torch.memory_format" # Could not find this as an explicit type, assuming this
    },
    "inner": {},
}
signatures["torch.ShortStorage_1"] = {
    "args": {
        "size": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.ShortStorage_2"] = {
    "args": {
        "size": "list" # potentially tuple, but list seems closer
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.ShortStorage_3"] = {
    "args": {
        "data": "list" # potentially tuple, but list seems closer
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.ShortStorage_4"] = {
    "args": {
        "storage": "torch.ShortStorage" # It's actually ShortStorage but I am assuming it is not allowed
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.ShortStorage_5"] = {
    "args": {},
    "kwargs": {},
    "inner": {}
}
signatures["torch.not_equal_1"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.not_equal_2"] = {
    "args": {
        "input": "tensor",
        "other": "float"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.nn.quantized.QFunctional"] = {
    "args": {},
    "kwargs": {},
    "inner": {
        "args": {},
        "kwargs": {}
    },
}
signatures["torch.nn.quantized.QFunctional.add_scalar"] = {
    "args": {
        "input": "tensor",
        "other": "float"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.quantized.QFunctional.add"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.quantized.QFunctional.mul_scalar"] = {
    "args": {
        "input": "tensor",
        "other": "float"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.quantized.QFunctional.mul"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.quantized.QFunctional.conv2d"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor",
    },
    "kwargs": {
        "bias": "tensor",
        "stride": "tuple",
        "padding": "tuple",
        "dilation": "tuple",
        "groups": "integer"
    },
    "inner": {},
}
signatures["torch.nn.quantized.QFunctional.linear"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor"
    },
    "kwargs": {
        "bias": "tensor"
    },
    "inner": {},
}
signatures["torch.nn.quantized.QFunctional.relu"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.quantized.QFunctional.hardtanh"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "min_val": "float",
        "max_val": "float"
    },
    "inner": {},
}
signatures["torch.nn.quantized.QFunctional.adaptive_avg_pool2d"] = {
    "args": {
        "input": "tensor",
        "output_size": "tuple" # Could also be integer, create another signature if needed
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.quantized.QFunctional.avg_pool2d"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "kernel_size": "tuple", # can be int
        "stride": "tuple", # can be int
        "padding": "tuple", # can be int
        "ceil_mode": "boolean",
        "count_include_pad": "boolean",
        "divisor_override": "integer"
    },
    "inner": {},
}
signatures["torch.nn.quantized.QFunctional.max_pool2d"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "kernel_size": "tuple", # can be int
        "stride": "tuple", # can be int
        "padding": "tuple", # can be int
        "dilation": "tuple", # can be int
        "ceil_mode": "boolean",
        "return_indices": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.quantized.QFunctional.interpolate"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "size": "tuple", # can be int
        "scale_factor": "float", # can be tuple or list
        "mode": "string",
        "align_corners": "boolean",
        "recompute_scale_factor": "boolean"
    },
    "inner": {},
}
signatures["torch.asin_"] = {
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
        "spacing": "list", # Could also be a scalar or a tensor_list, but list seems most general
        "dim": "list", # Could also be an int, but list seems more general
        "edge_order": "integer"
    },
    "inner": {},
}
signatures["torch.asinh_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.set_warn_always"] = {
    "args": {
        "warn_always": "boolean"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nanmean_1"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "dim": "integer",
        "keepdim": "boolean",
        "dtype": "dtype",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.nanmean_2"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "dim": "tuple",
        "keepdim": "boolean",
        "dtype": "dtype",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.nn.BatchNorm1d"] = {
    "args": {
        "num_features": "integer"
    },
    "kwargs": {
        "eps": "float",
        "momentum": "float",  # Optional[float] but can be None, representing a float
        "affine": "boolean",
        "track_running_stats": "boolean",
        "dtype": "dtype",
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.asarray"] = {
    "args": {
        "obj": "list" # Can be also a tensor, NumPy array, DLPack Capsule, object that implements Python's buffer protocol, scalar, or sequence of scalars
    },
    "kwargs": {
        "dtype": "dtype",
        "copy": "boolean",
        "requires_grad": "boolean"
    },
    "inner": {},
}
signatures["torch.cudnn_affine_grid_generator"] = {
    "args": {
        "theta": "tensor",
        "N": "integer",
        "C": "integer",
        "H": "integer",
        "W": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.as_strided_copy"] = {
    "args": {
        "source": "tensor",
        "size": "tuple",
        "stride": "tuple",
        "storage_offset": "integer"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.alias_copy"] = {
    "args": {
        "self": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.masked_scatter"] = {
    "args": {
        "input": "tensor",
        "mask": "tensor",
        "source": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.manual_seed"] = {
    "args": {
        "seed": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.special.expit"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.fft.rfft"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "n": "integer",
        "dim": "integer",
        "norm": "string",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.multiply"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.isin"] = {
    "args": {
        "elements": "tensor",
        "test_elements": "tensor"
    },
    "kwargs": {
        "assume_unique": "boolean",
        "invert": "boolean"
    },
    "inner": {},
}
signatures["torch.clamp_max"] = {
    "args": {
        "input": "tensor",
        "max": "float"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.isreal"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.is_grad_enabled"] = {
    "args": {},
    "kwargs": {},
    "inner": {},
}
signatures["torch.gcd_"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.set_autocast_xla_enabled"] = {
    "args": {
        "enabled": "boolean"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.arcsin"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.set_anomaly_enabled"] = {
    "args": {
        "mode": "boolean"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.Softplus"] = {
    "args": {},
    "kwargs": {
        "beta": "float",
        "threshold": "float"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.sym_int_1"] = {
    "args": {
        "a": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.sym_int_2"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.arccos_"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.set_autocast_cpu_enabled"] = {
    "args": {
        "enabled": "boolean"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.quantize_per_tensor"] = {
    "args": {
        "input": "tensor",
        "scale": "float",
        "zero_point": "integer",
        "dtype": "dtype"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.is_same_size"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.quantize_per_channel"] = {
    "args": {
        "input": "tensor",
        "scales": "tensor",
        "zero_points": "tensor",
        "axis": "integer",
        "dtype": "dtype"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.blackman_window"] = {
    "args": {
        "window_length": "integer"
    },
    "kwargs": {
        "periodic": "boolean",
        "dtype": "dtype",
        "layout": "string", # Should it be torch.layout?
        "requires_grad": "boolean"
    },
    "inner": {},
}
signatures["torch.kaiser_window_1"] = {
    "args": {
        "length": "integer"
    },
    "kwargs": {
        "periodic": "boolean",
        "beta": "float",
        "dtype": "dtype",
        "layout": "string", # Should this be torch.layout?
        "requires_grad": "boolean"
    },
    "inner": {},
}
signatures["torch.kaiser_window_2"] = {
    "args": {
        "length": "integer",
        "beta": "float"
    },
    "kwargs": {
        "periodic": "boolean",
        "dtype": "dtype",
        "layout": "string", # Should this be torch.layout?
        "requires_grad": "boolean"
    },
    "inner": {},
}
signatures["torch.rand_like"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dtype": "dtype",
        "layout": "string", #string might be too restrictive, but it's the closest
        "requires_grad": "boolean",
        "memory_format": "string" #string might be too restrictive, but it's the closest
    },
    "inner": {},
}
signatures["torch.randint_1"] = {
    "args": {
        "high": "integer",
        "size": "tuple"
    },
    "kwargs": {
        "low": "integer",
        "generator": "torch.Generator", # Should be a Generator object type, but not available
        "out": "tensor",
        "dtype": "dtype",
        "layout": "torch.layout", # Should be a Layout object type, but not available
        "requires_grad": "boolean"
    },
    "inner": {},
}

signatures["torch.randint_2"] = {
    "args": {
        "low": "integer",
        "high": "integer",
        "size": "tuple"
    },
    "kwargs": {
        "generator": "torch.Generator", # Should be a Generator object type, but not available
        "out": "tensor",
        "dtype": "dtype",
        "layout": "torch.layout", # Should be a Layout object type, but not available
        "requires_grad": "boolean"
    },
    "inner": {},
}
signatures["torch.randint_like_1"] = {
    "args": {
        "input": "tensor",
        "low": "integer",
        "high": "integer"
    },
    "kwargs": {
        "dtype": "dtype",
        "layout": "string", # unsure what layout means here, it's torch.layout
        "requires_grad": "boolean",
    },
    "inner": {},
}

signatures["torch.randint_like_2"] = {
    "args": {
        "input": "tensor",
        "high": "integer"
    },
    "kwargs": {
        "dtype": "dtype",
        "layout": "string", # unsure what layout means here, it's torch.layout
        "requires_grad": "boolean",
    },
    "inner": {},
}
signatures["torch.randn_1"] = {
    "args": {
        "size": "integer"  # Should be a sequence of integers, but simplified to integer for single integer input
    },
    "kwargs": {
        "generator": "torch.Generator",  # No type available that represents this; ideally would want custom type
        "out": "tensor",
        "dtype": "dtype",
        "layout": "torch.layout",  # No type available that represents this; ideally would want custom type
        "requires_grad": "boolean",
        "pin_memory": "boolean"
    },
    "inner": {},
}

signatures["torch.randn_2"] = {
    "args": {
        "size": "tuple"
    },
    "kwargs": {
        "generator": "torch.Generator",  # No type available that represents this; ideally would want custom type
        "out": "tensor",
        "dtype": "dtype",
        "layout": "torch.layout",  # No type available that represents this; ideally would want custom type
        "requires_grad": "boolean",
        "pin_memory": "boolean"
    },
    "inner": {},
}

signatures["torch.randn_3"] = {
    "args": {
        "size": "list"
    },
    "kwargs": {
        "generator": "torch.Generator",  # No type available that represents this; ideally would want custom type
        "out": "tensor",
        "dtype": "dtype",
        "layout": "torch.layout",  # No type available that represents this; ideally would want custom type
        "requires_grad": "boolean",
        "pin_memory": "boolean"
    },
    "inner": {},
}
signatures["torch.sparse_csr_tensor_1"] = {
    "args": {
        "crow_indices": "tensor",
        "col_indices": "tensor",
        "values": "tensor",
        "size": "tuple"
    },
    "kwargs": {
        "dtype": "dtype",
        "requires_grad": "boolean"
    },
    "inner": {},
}
signatures["torch.sparse_csr_tensor_2"] = {
    "args": {
        "crow_indices": "tensor",
        "col_indices": "tensor",
        "values": "tensor",
        "size": "list"
    },
    "kwargs": {
        "dtype": "dtype",
        "requires_grad": "boolean"
    },
    "inner": {},
}
signatures["torch.sparse_csr_tensor_3"] = {
    "args": {
        "data": "tensor"
    },
    "kwargs": {
       "dtype": "dtype",
       "requires_grad": "boolean"
    },
    "inner": {},
}
signatures["torch.normal_1"] = {
    "args": {
        "mean": "tensor",
        "std": "tensor"
    },
    "kwargs": {
        "generator": "torch.Generator", # Should be torch.Generator type?
        "out": "tensor"
    },
    "inner": {},
}

signatures["torch.normal_2"] = {
    "args": {
        "mean": "float",
        "std": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}

signatures["torch.normal_3"] = {
    "args": {
        "mean": "tensor",
        "std": "float"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}

signatures["torch.normal_4"] = {
    "args": {
        "mean": "float",
        "std": "float",
        "size": "tuple" # or list?
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.fft.ifft"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "n": "integer",
        "dim": "integer",
        "norm": "string",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.fft.ifft2"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "s": "tuple",
        "dim": "tuple",
        "norm": "string",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.fft.ifftn"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "s": "tuple",
        "dim": "tuple",
        "norm": "string",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.fft.irfft"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "n": "integer",
        "dim": "integer",
        "norm": "string",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.fft.irfft2"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "s": "tuple",
        "dim": "tuple",
        "norm": "string",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.fft.rfftn"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "s": "tuple",
        "dim": "tuple",
        "norm": "string",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.fft.irfftn"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "s": "tuple",
        "dim": "tuple",
        "norm": "string",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.fft.hfft"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "n": "integer",
        "dim": "integer",
        "norm": "string",
        "out": "tensor"
    },
    "inner": {}
}
signatures["torch.fft.ihfft"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "n": "integer",
        "dim": "integer",
        "norm": "string",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.linalg.cholesky"] = {
    "args": {
        "A": "tensor"
    },
    "kwargs": {
        "upper": "boolean",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.linalg.det"] = {
    "args": {
        "A": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.linalg.slogdet"] = {
    "args": {
        "A": "tensor"
    },
    "kwargs": {
        "out": "tuple" # Could also be None, but we only allow specific types
    },
    "inner": {},
}
signatures["torch.linalg.eigvalsh"] = {
    "args": {
        "A": "tensor"
    },
    "kwargs": {
        "UPLO": "string",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.linalg.householder_product"] = {
    "args": {
        "input": "tensor",
        "h": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.linalg.norm_1"] = {
    "args": {
        "A": "tensor"
    },
    "kwargs": {
        "ord": "float",  # Could also be int or string
        "dim": "integer",
        "keepdim": "boolean",
        "out": "tensor",
        "dtype": "dtype"
    },
    "inner": {},
}
signatures["torch.linalg.norm_2"] = {
    "args": {
        "A": "tensor"
    },
    "kwargs": {
        "ord": "float",  # Could also be int or string
        "dim": "tuple",
        "keepdim": "boolean",
        "out": "tensor",
        "dtype": "dtype"
    },
    "inner": {},
}
signatures["torch.linalg.matrix_norm_1"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "ord": "integer",
        "dim": "tuple",
        "keepdim": "boolean",
        "out": "tensor",
        "dtype": "dtype"
    },
    "inner": {},
}

signatures["torch.linalg.matrix_norm_2"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "ord": "string",
        "dim": "tuple",
        "keepdim": "boolean",
        "out": "tensor",
        "dtype": "dtype"
    },
    "inner": {},
}

signatures["torch.linalg.matrix_norm_3"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "ord": "list",
        "dim": "tuple",
        "keepdim": "boolean",
        "out": "tensor",
        "dtype": "dtype"
    },
    "inner": {},
}

signatures["torch.linalg.matrix_norm_4"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "ord": "tensor",
        "dim": "tuple",
        "keepdim": "boolean",
        "out": "tensor",
        "dtype": "dtype"
    },
    "inner": {},
}
signatures["torch.linalg.matrix_power"] = {
    "args": {
        "A": "tensor",
        "n": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.linalg.qr"] = {
    "args": {
        "A": "tensor",
    },
    "kwargs": {
        "mode": "string",
        "out": "tuple",
    },
    "inner": {},
}
signatures["torch.linalg.svd"] = {
    "args": {
        "A": "tensor"
    },
    "kwargs": {
        "full_matrices": "boolean",
        "driver": "string",
        "out": "tuple"
    },
    "inner": {},
}
signatures["torch.linalg.tensorsolve"] = {
    "args": {
        "A": "tensor",
        "B": "tensor"
    },
    "kwargs": {
        "dims": "tuple", # or None, but can't express None
        "out": "tensor" # or None, but can't express None
    },
    "inner": {},
}
signatures["torch.nn.Bilinear"] = {
    "args": {
        "in1_features": "integer",
        "in2_features": "integer",
        "out_features": "integer"
    },
    "kwargs": {
        "bias": "boolean",
        "dtype": "dtype"
    },
    "inner": {
        "args": {
            "input1": "tensor",
            "input2": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.ELU"] = {
    "args": {},
    "kwargs": {
        "alpha": "float",
        "inplace": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.GLU"] = {
    "args": {},
    "kwargs": {
        "dim": "integer"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.GELU"] = {
    "args": {},
    "kwargs": {
        "approximate": "string"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.AdaptiveLogSoftmaxWithLoss"] = {
    "args": {
        "in_features": "integer",
        "n_classes": "integer",
        "cutoffs": "list"  # Sequence should be interpreted as list or tuple, choosing list
    },
    "kwargs": {
        "div_value": "float",
        "head_bias": "boolean",
        "dtype": "dtype"
    },
    "inner": {
        "args": {
            "input": "tensor",
            "target": "tensor"
        },
        "kwargs": {}
    },
}

signatures["torch.nn.AdaptiveLogSoftmaxWithLoss.log_prob"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}

signatures["torch.nn.AdaptiveLogSoftmaxWithLoss.predict"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.BatchNorm3d"] = {
    "args": {
        "num_features": "integer"
    },
    "kwargs": {
        "eps": "float",
        "momentum": "float",  # Optional[float], but float is the best match
        "affine": "boolean",
        "track_running_stats": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.Conv1d"] = {
    "args": {
        "in_channels": "integer",
        "out_channels": "integer",
        "kernel_size": "integer" # or tuple, creating another signature for tuple
    },
    "kwargs": {
        "stride": "integer", # or tuple, creating another signature for tuple
        "padding": "integer", # or tuple or string, creating another signature for tuple and string
        "dilation": "integer", # or tuple, creating another signature for tuple
        "groups": "integer",
        "bias": "boolean",
        "padding_mode": "string",
        "dtype": "dtype"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}

signatures["torch.nn.Conv1d_tuple_kernel"] = {
    "args": {
        "in_channels": "integer",
        "out_channels": "integer",
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "integer", # or tuple, creating another signature for tuple
        "padding": "integer", # or tuple or string, creating another signature for tuple and string
        "dilation": "integer", # or tuple, creating another signature for tuple
        "groups": "integer",
        "bias": "boolean",
        "padding_mode": "string",
        "dtype": "dtype"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}

signatures["torch.nn.Conv1d_tuple_stride"] = {
    "args": {
        "in_channels": "integer",
        "out_channels": "integer",
        "kernel_size": "integer" # or tuple, creating another signature for tuple
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "integer", # or tuple or string, creating another signature for tuple and string
        "dilation": "integer", # or tuple, creating another signature for tuple
        "groups": "integer",
        "bias": "boolean",
        "padding_mode": "string",
        "dtype": "dtype"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}

signatures["torch.nn.Conv1d_tuple_dilation"] = {
    "args": {
        "in_channels": "integer",
        "out_channels": "integer",
        "kernel_size": "integer" # or tuple, creating another signature for tuple
    },
    "kwargs": {
        "stride": "integer", # or tuple, creating another signature for tuple
        "padding": "integer", # or tuple or string, creating another signature for tuple and string
        "dilation": "tuple",
        "groups": "integer",
        "bias": "boolean",
        "padding_mode": "string",
        "dtype": "dtype"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}

signatures["torch.nn.Conv1d_tuple_padding"] = {
    "args": {
        "in_channels": "integer",
        "out_channels": "integer",
        "kernel_size": "integer" # or tuple, creating another signature for tuple
    },
    "kwargs": {
        "stride": "integer", # or tuple, creating another signature for tuple
        "padding": "tuple",
        "dilation": "integer", # or tuple, creating another signature for tuple
        "groups": "integer",
        "bias": "boolean",
        "padding_mode": "string",
        "dtype": "dtype"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}

signatures["torch.nn.Conv1d_string_padding"] = {
    "args": {
        "in_channels": "integer",
        "out_channels": "integer",
        "kernel_size": "integer" # or tuple, creating another signature for tuple
    },
    "kwargs": {
        "stride": "integer", # or tuple, creating another signature for tuple
        "padding": "string",
        "dilation": "integer", # or tuple, creating another signature for tuple
        "groups": "integer",
        "bias": "boolean",
        "padding_mode": "string",
        "dtype": "dtype"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.Conv2d_1"] = {
    "args": {
        "in_channels": "integer",
        "out_channels": "integer",
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "integer",
        "dilation": "integer",
        "groups": "integer",
        "bias": "boolean",
        "padding_mode": "string",
        "dtype": "dtype"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}

signatures["torch.nn.Conv2d_2"] = {
    "args": {
        "in_channels": "integer",
        "out_channels": "integer",
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "tuple",
        "dilation": "tuple",
        "groups": "integer",
        "bias": "boolean",
        "padding_mode": "string",
        "dtype": "dtype"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}

signatures["torch.nn.Conv2d_3"] = {
    "args": {
        "in_channels": "integer",
        "out_channels": "integer",
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "integer",
        "dilation": "integer",
        "groups": "integer",
        "bias": "boolean",
        "padding_mode": "string",
        "dtype": "dtype"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Conv2d_4"] = {
    "args": {
        "in_channels": "integer",
        "out_channels": "integer",
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "tuple",
        "dilation": "tuple",
        "groups": "integer",
        "bias": "boolean",
        "padding_mode": "string",
        "dtype": "dtype"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}

signatures["torch.nn.Conv2d_5"] = {
    "args": {
        "in_channels": "integer",
        "out_channels": "integer",
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "string",
        "dilation": "integer",
        "groups": "integer",
        "bias": "boolean",
        "padding_mode": "string",
        "dtype": "dtype"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}

signatures["torch.nn.Conv2d_6"] = {
    "args": {
        "in_channels": "integer",
        "out_channels": "integer",
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "string",
        "dilation": "tuple",
        "groups": "integer",
        "bias": "boolean",
        "padding_mode": "string",
        "dtype": "dtype"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}

signatures["torch.nn.Conv2d_7"] = {
    "args": {
        "in_channels": "integer",
        "out_channels": "integer",
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "string",
        "dilation": "tuple",
        "groups": "integer",
        "bias": "boolean",
        "padding_mode": "string",
        "dtype": "dtype"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}

signatures["torch.nn.Conv2d_8"] = {
    "args": {
        "in_channels": "integer",
        "out_channels": "integer",
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "string",
        "dilation": "integer",
        "groups": "integer",
        "bias": "boolean",
        "padding_mode": "string",
        "dtype": "dtype"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Conv3d"] = {
    "args": {
        "in_channels": "integer",
        "out_channels": "integer",
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "integer",
        "dilation": "integer",
        "groups": "integer",
        "bias": "boolean",
        "padding_mode": "string",
        "dtype": "dtype"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}

signatures["torch.nn.Conv3d_2"] = {
    "args": {
        "in_channels": "integer",
        "out_channels": "integer",
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "tuple",
        "dilation": "tuple",
        "groups": "integer",
        "bias": "boolean",
        "padding_mode": "string",
        "dtype": "dtype"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}

signatures["torch.nn.Conv3d_3"] = {
    "args": {
        "in_channels": "integer",
        "out_channels": "integer",
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "string",
        "dilation": "integer",
        "groups": "integer",
        "bias": "boolean",
        "padding_mode": "string",
        "dtype": "dtype"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}

signatures["torch.nn.Conv3d_4"] = {
    "args": {
        "in_channels": "integer",
        "out_channels": "integer",
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "string",
        "dilation": "tuple",
        "groups": "integer",
        "bias": "boolean",
        "padding_mode": "string",
        "dtype": "dtype"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.ConvTranspose1d"] = {
    "args": {
        "in_channels": "integer",
        "out_channels": "integer",
        "kernel_size": "tuple", # Can also be integer, creating new signatures
    },
    "kwargs": {
        "stride": "tuple", # Can also be integer, creating new signatures
        "padding": "tuple", # Can also be integer, creating new signatures
        "output_padding": "tuple", # Can also be integer, creating new signatures
        "groups": "integer",
        "bias": "boolean",
        "dilation": "tuple", # Can also be integer, creating new signatures
        "padding_mode": "string",
        "dtype": "dtype",
    },
    "inner": {
        "args": {
            "input": "tensor",
        },
        "kwargs": {}
    }
}

signatures["torch.nn.ConvTranspose1d_1"] = {
    "args": {
        "in_channels": "integer",
        "out_channels": "integer",
        "kernel_size": "integer",
    },
    "kwargs": {
        "stride": "integer",
        "padding": "integer",
        "output_padding": "integer",
        "groups": "integer",
        "bias": "boolean",
        "dilation": "integer",
        "padding_mode": "string",
        "dtype": "dtype",
    },
    "inner": {
        "args": {
            "input": "tensor",
        },
        "kwargs": {}
    }
}
signatures["torch.nn.ConvTranspose2d"] = {
    "args": {
        "in_channels": "integer",
        "out_channels": "integer",
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "integer",
        "output_padding": "integer",
        "groups": "integer",
        "bias": "boolean",
        "dilation": "integer",
        "padding_mode": "string",
        "dtype": "dtype"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {
            "output_size": "tuple"
        }
    }
}

signatures["torch.nn.ConvTranspose2d_1"] = {
    "args": {
        "in_channels": "integer",
        "out_channels": "integer",
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "tuple",
        "output_padding": "tuple",
        "groups": "integer",
        "bias": "boolean",
        "dilation": "tuple",
        "padding_mode": "string",
        "dtype": "dtype"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {
            "output_size": "tuple"
        }
    }
}
signatures["torch.nn.ConvTranspose3d"] = {
    "args": {
        "in_channels": "integer",
        "out_channels": "integer",
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "integer",
        "output_padding": "integer",
        "groups": "integer",
        "bias": "boolean",
        "dilation": "integer",
        "padding_mode": "string",
        "dtype": "dtype"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}

signatures["torch.nn.ConvTranspose3d_1"] = {
    "args": {
        "in_channels": "integer",
        "out_channels": "integer",
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "tuple",
        "output_padding": "tuple",
        "groups": "integer",
        "bias": "boolean",
        "dilation": "tuple",
        "padding_mode": "string",
        "dtype": "dtype"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.Dropout"] = {
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
    },
}
signatures["torch.nn.Dropout2d"] = {
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
signatures["torch.nn.Dropout3d"] = {
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
signatures["torch.nn.AlphaDropout"] = {
    "args": {
        "p": "float",
    },
    "kwargs": {
        "inplace": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
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
    },
}
signatures["torch.nn.Embedding"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean",
        "dtype": "dtype"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}

signatures["torch.nn.Embedding.from_pretrained"] = {
    "args": {
        "embeddings": "tensor"
    },
    "kwargs": {
        "freeze": "boolean",
        "padding_idx": "integer",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "sparse": "boolean"
    },
    "inner": {}
}
signatures["torch.nn.EmbeddingBag"] = {
    "args": {
        "num_embeddings": "integer",
        "embedding_dim": "integer"
    },
    "kwargs": {
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "mode": "string",
        "sparse": "boolean",
        "include_last_offset": "boolean",
        "padding_idx": "integer",
        "dtype": "dtype" # Could also be a string?
    },
    "inner": {
        "args": {
            "input": "tensor",
            "offsets": "tensor",
            "per_sample_weights": "tensor"
        },
        "kwargs": {}
    },
}

signatures["torch.nn.EmbeddingBag.from_pretrained"] = {
    "args": {
        "embeddings": "tensor"
    },
    "kwargs": {
        "freeze": "boolean",
        "max_norm": "float",
        "norm_type": "float",
        "scale_grad_by_freq": "boolean",
        "mode": "string",
        "sparse": "boolean",
        "include_last_offset": "boolean",
        "padding_idx": "integer"
    },
    "inner": {}
}
signatures["torch.nn.Fold_1"] = {
    "args": {
        "output_size": "tuple",
        "kernel_size": "tuple"
    },
    "kwargs": {
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
        "output_size": "tuple",
        "kernel_size": "tuple"
    },
    "kwargs": {
        "dilation": "tuple",
        "padding": "tuple",
        "stride": "tuple"
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
        "output_size": "integer",
        "kernel_size": "integer"
    },
    "kwargs": {
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
signatures["torch.nn.MultiLabelMarginLoss"] = {
    "args": {},
    "kwargs": {
        "size_average": "boolean", # Deprecated, but included for completeness
        "reduce": "boolean", # Deprecated, but included for completeness
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
signatures["torch.nn.TripletMarginWithDistanceLoss"] = {
    "args": {},
    "kwargs": {
        "distance_function": "Callable", # could be a function type?
        "margin": "float",
        "swap": "boolean",
        "reduction": "string"
    },
    "inner": {
        "args": {
            "anchor": "tensor",
            "positive": "tensor",
            "negative": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.CTCLoss"] = {
    "args": {},
    "kwargs": {
        "blank": "integer",
        "reduction": "string",
        "zero_infinity": "boolean"
    },
    "inner": {
        "args": {
            "log_probs": "tensor",
            "targets": "tensor",
            "input_lengths": "tuple", # Can also be a tensor
            "target_lengths": "tuple" # Can also be a tensor
        },
        "kwargs": {}
    }
}
signatures["torch.nn.ReplicationPad2d_1"] = {
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

signatures["torch.nn.ReplicationPad2d_2"] = {
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
signatures["torch.nn.ConstantPad2d_1"] = {
    "args": {
        "padding": "integer",
        "value": "float"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}

signatures["torch.nn.ConstantPad2d_2"] = {
    "args": {
        "padding": "tuple",
        "value": "float"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.ConstantPad3d_1"] = {
    "args": {
        "padding": "integer",
        "value": "float" # Could be any number, assuming float
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.ConstantPad3d_2"] = {
    "args": {
        "padding": "tuple",
        "value": "float" # Could be any number, assuming float
    },
    "kwargs": {},
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
        "divisor_override": "integer" # Optional[int]
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
        "divisor_override": "integer" # Optional[int]
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
        "divisor_override": "integer" # Optional[int]
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
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "tuple",
        "ceil_mode": "boolean",
        "count_include_pad": "boolean",
        "divisor_override": "integer" # Optional[int]
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
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "integer",
        "ceil_mode": "boolean",
        "count_include_pad": "boolean",
        "divisor_override": "integer" # Optional[int]
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
        "stride": "tuple",
        "padding": "integer",
        "ceil_mode": "boolean",
        "count_include_pad": "boolean",
        "divisor_override": "integer" # Optional[int]
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
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "tuple",
        "ceil_mode": "boolean",
        "count_include_pad": "boolean",
        "divisor_override": "integer" # Optional[int]
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
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "tuple",
        "ceil_mode": "boolean",
        "count_include_pad": "boolean",
        "divisor_override": "integer" # Optional[int]
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.AvgPool3d_1"] = {
    "args": {
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "integer",
        "ceil_mode": "boolean",
        "count_include_pad": "boolean",
        "divisor_override": "integer" # Optional[int] is interpreted as integer
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.AvgPool3d_2"] = {
    "args": {
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "tuple",
        "ceil_mode": "boolean",
        "count_include_pad": "boolean",
        "divisor_override": "integer" # Optional[int] is interpreted as integer
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.MaxPool1d_1"] = {
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
signatures["torch.nn.MaxPool1d_2"] = {
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
signatures["torch.nn.AdaptiveMaxPool1d_1"] = {
    "args": {
        "output_size": "integer"
    },
    "kwargs": {
        "return_indices": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.AdaptiveMaxPool1d_2"] = {
    "args": {
        "output_size": "tuple"
    },
    "kwargs": {
        "return_indices": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.AdaptiveMaxPool2d_1"] = {
    "args": {
        "output_size": "integer",
    },
    "kwargs": {
        "return_indices": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.AdaptiveMaxPool2d_2"] = {
    "args": {
        "output_size": "tuple",
    },
    "kwargs": {
        "return_indices": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.AdaptiveMaxPool2d_3"] = {
    "args": {
        "output_size": "None",
    },
    "kwargs": {
        "return_indices": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.AdaptiveMaxPool3d_1"] = {
    "args": {
        "output_size": "integer"
    },
    "kwargs": {
        "return_indices": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}

signatures["torch.nn.AdaptiveMaxPool3d_2"] = {
    "args": {
        "output_size": "tuple"
    },
    "kwargs": {
        "return_indices": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.AdaptiveAvgPool1d_1"] = {
    "args": {
        "output_size": "integer"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.AdaptiveAvgPool1d_2"] = {
    "args": {
        "output_size": "tuple"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.AdaptiveAvgPool2d_1"] = {
    "args": {
        "output_size": "integer"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.AdaptiveAvgPool2d_2"] = {
    "args": {
        "output_size": "tuple"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.RNN"] = {
    "args": {
        "input_size": "integer",
        "hidden_size": "integer",
    },
    "kwargs": {
        "num_layers": "integer",
        "nonlinearity": "string",
        "bias": "boolean",
        "batch_first": "boolean",
        "dropout": "float",
        "bidirectional": "boolean",
        "dtype": "dtype", # Could also be None?
    },
    "inner": {
        "args": {
            "input": "tensor",
            "hx": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.LSTM"] = {
    "args": {
        "input_size": "integer",
        "hidden_size": "integer"
    },
    "kwargs": {
        "num_layers": "integer",
        "bias": "boolean",
        "batch_first": "boolean",
        "dropout": "float",
        "bidirectional": "boolean",
        "proj_size": "integer",
        "dtype": "dtype"
    },
    "inner": {
        "args": {
            "input": "tensor",
            "h_0": "tensor",
            "c_0": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.GRU"] = {
    "args": {
        "input_size": "integer",
        "hidden_size": "integer",
    },
    "kwargs": {
        "num_layers": "integer",
        "bias": "boolean",
        "batch_first": "boolean",
        "dropout": "float",
        "bidirectional": "boolean",
        "dtype": "dtype",
    },
    "inner": {
        "args": {
            "input": "tensor",
            "h_0": "tensor",
        },
        "kwargs": {}
    },
}
signatures["torch.nn.RNNCell"] = {
    "args": {
        "input_size": "integer",
        "hidden_size": "integer"
    },
    "kwargs": {
        "bias": "boolean",
        "nonlinearity": "string",
        "dtype": "dtype"
    },
    "inner": {
        "args": {
            "input": "tensor",
            "hidden": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.Transformer"] = {
    "args": {},
    "kwargs": {
        "d_model": "integer",
        "nhead": "integer",
        "num_encoder_layers": "integer",
        "num_decoder_layers": "integer",
        "dim_feedforward": "integer",
        "dropout": "float",
        "activation": "string", # Could be a string or a Callable, but sticking to string for simplicity
        "custom_encoder": "list", # Should be Any
        "custom_decoder": "list", # Should be Any
        "layer_norm_eps": "float",
        "batch_first": "boolean",
        "norm_first": "boolean",
        "bias": "boolean",
        # device skipped
        "dtype": "dtype"
    },
    "inner": {
        "args": {
            "src": "tensor",
            "tgt": "tensor"
        },
        "kwargs": {
            "src_mask": "tensor",
            "tgt_mask": "tensor",
            "memory_mask": "tensor",
            "src_key_padding_mask": "tensor",
            "tgt_key_padding_mask": "tensor",
            "memory_key_padding_mask": "tensor",
            "src_is_causal": "boolean",
            "tgt_is_causal": "boolean",
            "memory_is_causal": "boolean"
        }
    }
}

signatures["torch.nn.Transformer.generate_square_subsequent_mask"] = {
    "args": {
        "sz": "integer"
    },
    "kwargs": {
        # device skipped
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["torch.nn.TransformerEncoderLayer"] = {
    "args": {
        "d_model": "integer",
        "nhead": "integer"
    },
    "kwargs": {
        "dim_feedforward": "integer",
        "dropout": "float",
        "activation": "string",  # Could also be a callable, but string is a closer match.
        "layer_norm_eps": "float",
        "batch_first": "boolean",
        "norm_first": "boolean",
        "bias": "boolean",
        "dtype": "dtype",
    },
    "inner": {
        "args": {
            "src": "tensor"
        },
        "kwargs": {
            "src_mask": "tensor",
            "src_key_padding_mask": "tensor",
            "is_causal": "boolean"
        }
    }
}
signatures["torch.nn.TransformerDecoderLayer"] = {
    "args": {
        "d_model": "integer",
        "nhead": "integer"
    },
    "kwargs": {
        "dim_feedforward": "integer",
        "dropout": "float",
        "activation": "string", # Could also be a callable, but sticking to string as per instructions
        "layer_norm_eps": "float",
        "batch_first": "boolean",
        "norm_first": "boolean",
        "bias": "boolean"
    },
    "inner": {
        "args": {
            "tgt": "tensor",
            "memory": "tensor"
        },
        "kwargs": {
            "tgt_mask": "tensor", #Optional[Tensor]
            "memory_mask": "tensor", #Optional[Tensor]
            "tgt_key_padding_mask": "tensor", #Optional[Tensor]
            "memory_key_padding_mask": "tensor", #Optional[Tensor]
            "tgt_is_causal": "boolean",
            "memory_is_causal": "boolean"
        }
    }
}
signatures["torch.nn.Upsample_1"] = {
    "args": {},
    "kwargs": {
        "size": "integer",
        "scale_factor": "float",
        "mode": "string",
        "align_corners": "boolean",
        "recompute_scale_factor": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}

signatures["torch.nn.Upsample_2"] = {
    "args": {},
    "kwargs": {
        "size": "tuple",
        "scale_factor": "float",
        "mode": "string",
        "align_corners": "boolean",
        "recompute_scale_factor": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}

signatures["torch.nn.Upsample_3"] = {
    "args": {},
    "kwargs": {
        "size": "tuple",
        "scale_factor": "tuple",
        "mode": "string",
        "align_corners": "boolean",
        "recompute_scale_factor": "boolean"
    },
    "inner": {
        "args": {
            "input": "tensor"
        },
        "kwargs": {}
    },
}
signatures["torch.nn.functional.conv1d"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor"
    },
    "kwargs": {
        "bias": "tensor",
        "stride": "integer",
        "padding": "string", # Could be an integer or tuple as well, creating more signatures
        "dilation": "integer",
        "groups": "integer"
    },
    "inner": {},
}
signatures["torch.nn.functional.conv2d_1"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor"
    },
    "kwargs": {
        "bias": "tensor",
        "stride": "integer",
        "padding": "integer",
        "dilation": "integer",
        "groups": "integer"
    },
    "inner": {},
}
signatures["torch.nn.functional.conv2d_2"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor"
    },
    "kwargs": {
        "bias": "tensor",
        "stride": "tuple",
        "padding": "integer",
        "dilation": "integer",
        "groups": "integer"
    },
    "inner": {},
}
signatures["torch.nn.functional.conv2d_3"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor"
    },
    "kwargs": {
        "bias": "tensor",
        "stride": "integer",
        "padding": "tuple",
        "dilation": "integer",
        "groups": "integer"
    },
    "inner": {},
}
signatures["torch.nn.functional.conv2d_4"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor"
    },
    "kwargs": {
        "bias": "tensor",
        "stride": "tuple",
        "padding": "tuple",
        "dilation": "integer",
        "groups": "integer"
    },
    "inner": {},
}
signatures["torch.nn.functional.conv2d_5"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor"
    },
    "kwargs": {
        "bias": "tensor",
        "stride": "integer",
        "padding": "string", # Should be a string ("valid", "same")
        "dilation": "integer",
        "groups": "integer"
    },
    "inner": {},
}
signatures["torch.nn.functional.conv2d_6"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor"
    },
    "kwargs": {
        "bias": "tensor",
        "stride": "tuple",
        "padding": "string", # Should be a string ("valid", "same")
        "dilation": "integer",
        "groups": "integer"
    },
    "inner": {},
}
signatures["torch.nn.functional.conv2d_7"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor"
    },
    "kwargs": {
        "bias": "tensor",
        "stride": "integer",
        "padding": "integer",
        "dilation": "tuple",
        "groups": "integer"
    },
    "inner": {},
}
signatures["torch.nn.functional.conv2d_8"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor"
    },
    "kwargs": {
        "bias": "tensor",
        "stride": "tuple",
        "padding": "integer",
        "dilation": "tuple",
        "groups": "integer"
    },
    "inner": {},
}
signatures["torch.nn.functional.conv2d_9"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor"
    },
    "kwargs": {
        "bias": "tensor",
        "stride": "integer",
        "padding": "tuple",
        "dilation": "tuple",
        "groups": "integer"
    },
    "inner": {},
}
signatures["torch.nn.functional.conv2d_10"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor"
    },
    "kwargs": {
        "bias": "tensor",
        "stride": "tuple",
        "padding": "tuple",
        "dilation": "tuple",
        "groups": "integer"
    },
    "inner": {},
}
signatures["torch.nn.functional.conv2d_11"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor"
    },
    "kwargs": {
        "bias": "tensor",
        "stride": "integer",
        "padding": "string", # Should be a string ("valid", "same")
        "dilation": "tuple",
        "groups": "integer"
    },
    "inner": {},
}
signatures["torch.nn.functional.conv2d_12"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor"
    },
    "kwargs": {
        "bias": "tensor",
        "stride": "tuple",
        "padding": "string", # Should be a string ("valid", "same")
        "dilation": "tuple",
        "groups": "integer"
    },
    "inner": {},
}
signatures["torch.nn.functional.conv3d"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor"
    },
    "kwargs": {
        "bias": "tensor",
        "stride": "tuple",
        "padding": "string", # or could be integer or tuple, need more signatures
        "dilation": "tuple", # or could be integer, need more signatures
        "groups": "integer"
    },
    "inner": {},
}
signatures["torch.nn.functional.conv3d_stride_int"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor"
    },
    "kwargs": {
        "bias": "tensor",
        "stride": "integer",
        "padding": "string", # or could be integer or tuple, need more signatures
        "dilation": "tuple", # or could be integer, need more signatures
        "groups": "integer"
    },
    "inner": {},
}
signatures["torch.nn.functional.conv3d_padding_int"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor"
    },
    "kwargs": {
        "bias": "tensor",
        "stride": "tuple",
        "padding": "integer",
        "dilation": "tuple", # or could be integer, need more signatures
        "groups": "integer"
    },
    "inner": {},
}
signatures["torch.nn.functional.conv3d_padding_tuple"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor"
    },
    "kwargs": {
        "bias": "tensor",
        "stride": "tuple",
        "padding": "tuple",
        "dilation": "tuple", # or could be integer, need more signatures
        "groups": "integer"
    },
    "inner": {},
}
signatures["torch.nn.functional.conv3d_dilation_int"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor"
    },
    "kwargs": {
        "bias": "tensor",
        "stride": "tuple",
        "padding": "string", # or could be integer or tuple, need more signatures
        "dilation": "integer",
        "groups": "integer"
    },
    "inner": {},
}
signatures["torch.nn.functional.conv3d_dilation_int_stride_int"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor"
    },
    "kwargs": {
        "bias": "tensor",
        "stride": "integer",
        "padding": "string", # or could be integer or tuple, need more signatures
        "dilation": "integer",
        "groups": "integer"
    },
    "inner": {},
}
signatures["torch.nn.functional.conv3d_dilation_int_padding_int"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor"
    },
    "kwargs": {
        "bias": "tensor",
        "stride": "tuple",
        "padding": "integer",
        "dilation": "integer",
        "groups": "integer"
    },
    "inner": {},
}
signatures["torch.nn.functional.conv3d_dilation_int_padding_tuple"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor"
    },
    "kwargs": {
        "bias": "tensor",
        "stride": "tuple",
        "padding": "tuple",
        "dilation": "integer",
        "groups": "integer"
    },
    "inner": {},
}
signatures["torch.nn.functional.conv_transpose1d"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor",
    },
    "kwargs": {
        "bias": "tensor",
        "stride": "integer", # Could also be a tuple. Creating another signature.
        "padding": "integer", # Could also be a tuple. Creating another signature.
        "output_padding": "integer", # Could also be a tuple. Creating another signature.
        "groups": "integer",
        "dilation": "integer" # Could also be a tuple. Creating another signature.
    },
    "inner": {},
}

signatures["torch.nn.functional.conv_transpose1d_stride_tuple"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor",
    },
    "kwargs": {
        "bias": "tensor",
        "stride": "tuple",
        "padding": "integer", # Could also be a tuple. Creating another signature.
        "output_padding": "integer", # Could also be a tuple. Creating another signature.
        "groups": "integer",
        "dilation": "integer" # Could also be a tuple. Creating another signature.
    },
    "inner": {},
}

signatures["torch.nn.functional.conv_transpose1d_padding_tuple"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor",
    },
    "kwargs": {
        "bias": "tensor",
        "stride": "integer", # Could also be a tuple. Creating another signature.
        "padding": "tuple",
        "output_padding": "integer", # Could also be a tuple. Creating another signature.
        "groups": "integer",
        "dilation": "integer" # Could also be a tuple. Creating another signature.
    },
    "inner": {},
}

signatures["torch.nn.functional.conv_transpose1d_output_padding_tuple"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor",
    },
    "kwargs": {
        "bias": "tensor",
        "stride": "integer", # Could also be a tuple. Creating another signature.
        "padding": "integer", # Could also be a tuple. Creating another signature.
        "output_padding": "tuple",
        "groups": "integer",
        "dilation": "integer" # Could also be a tuple. Creating another signature.
    },
    "inner": {},
}

signatures["torch.nn.functional.conv_transpose1d_dilation_tuple"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor",
    },
    "kwargs": {
        "bias": "tensor",
        "stride": "integer", # Could also be a tuple. Creating another signature.
        "padding": "integer", # Could also be a tuple. Creating another signature.
        "output_padding": "integer", # Could also be a tuple. Creating another signature.
        "groups": "integer",
        "dilation": "tuple"
    },
    "inner": {},
}

signatures["torch.nn.functional.conv_transpose1d_stride_padding_tuple"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor",
    },
    "kwargs": {
        "bias": "tensor",
        "stride": "tuple",
        "padding": "tuple",
        "output_padding": "integer", # Could also be a tuple. Creating another signature.
        "groups": "integer",
        "dilation": "integer" # Could also be a tuple. Creating another signature.
    },
    "inner": {},
}

signatures["torch.nn.functional.conv_transpose1d_stride_output_padding_tuple"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor",
    },
    "kwargs": {
        "bias": "tensor",
        "stride": "tuple",
        "padding": "integer", # Could also be a tuple. Creating another signature.
        "output_padding": "tuple",
        "groups": "integer",
        "dilation": "integer" # Could also be a tuple. Creating another signature.
    },
    "inner": {},
}

signatures["torch.nn.functional.conv_transpose1d_stride_dilation_tuple"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor",
    },
    "kwargs": {
        "bias": "tensor",
        "stride": "tuple",
        "padding": "integer", # Could also be a tuple. Creating another signature.
        "output_padding": "integer", # Could also be a tuple. Creating another signature.
        "groups": "integer",
        "dilation": "tuple"
    },
    "inner": {},
}

signatures["torch.nn.functional.conv_transpose1d_padding_output_padding_tuple"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor",
    },
    "kwargs": {
        "bias": "tensor",
        "stride": "integer", # Could also be a tuple. Creating another signature.
        "padding": "tuple",
        "output_padding": "tuple",
        "groups": "integer",
        "dilation": "integer" # Could also be a tuple. Creating another signature.
    },
    "inner": {},
}

signatures["torch.nn.functional.conv_transpose1d_padding_dilation_tuple"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor",
    },
    "kwargs": {
        "bias": "tensor",
        "stride": "integer", # Could also be a tuple. Creating another signature.
        "padding": "tuple",
        "output_padding": "integer", # Could also be a tuple. Creating another signature.
        "groups": "integer",
        "dilation": "tuple"
    },
    "inner": {},
}

signatures["torch.nn.functional.conv_transpose1d_output_padding_dilation_tuple"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor",
    },
    "kwargs": {
        "bias": "tensor",
        "stride": "integer", # Could also be a tuple. Creating another signature.
        "padding": "integer", # Could also be a tuple. Creating another signature.
        "output_padding": "tuple",
        "groups": "integer",
        "dilation": "tuple"
    },
    "inner": {},
}

signatures["torch.nn.functional.conv_transpose1d_stride_padding_output_padding_tuple"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor",
    },
    "kwargs": {
        "bias": "tensor",
        "stride": "tuple",
        "padding": "tuple",
        "output_padding": "tuple",
        "groups": "integer",
        "dilation": "integer" # Could also be a tuple. Creating another signature.
    },
    "inner": {},
}

signatures["torch.nn.functional.conv_transpose1d_stride_padding_dilation_tuple"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor",
    },
    "kwargs": {
        "bias": "tensor",
        "stride": "tuple",
        "padding": "tuple",
        "output_padding": "integer", # Could also be a tuple. Creating another signature.
        "groups": "integer",
        "dilation": "tuple"
    },
    "inner": {},
}

signatures["torch.nn.functional.conv_transpose1d_stride_output_padding_dilation_tuple"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor",
    },
    "kwargs": {
        "bias": "tensor",
        "stride": "tuple",
        "padding": "integer", # Could also be a tuple. Creating another signature.
        "output_padding": "tuple",
        "groups": "integer",
        "dilation": "tuple"
    },
    "inner": {},
}

signatures["torch.nn.functional.conv_transpose1d_padding_output_padding_dilation_tuple"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor",
    },
    "kwargs": {
        "bias": "tensor",
        "stride": "integer", # Could also be a tuple. Creating another signature.
        "padding": "tuple",
        "output_padding": "tuple",
        "groups": "integer",
        "dilation": "tuple"
    },
    "inner": {},
}

signatures["torch.nn.functional.conv_transpose1d_stride_padding_output_padding_dilation_tuple"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor",
    },
    "kwargs": {
        "bias": "tensor",
        "stride": "tuple",
        "padding": "tuple",
        "output_padding": "tuple",
        "groups": "integer",
        "dilation": "tuple"
    },
    "inner": {},
}
signatures["torch.nn.functional.conv_transpose3d"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor",
    },
    "kwargs": {
        "bias": "tensor",
        "stride": "integer",  # Could be tuple
        "padding": "integer", # Could be tuple
        "output_padding": "integer",  # Could be tuple
        "groups": "integer",
        "dilation": "integer" # Could be tuple
    },
    "inner": {},
}
signatures["torch.nn.functional.conv_transpose3d_1"] = {
    "args": {
        "input": "tensor",
        "weight": "tensor",
    },
    "kwargs": {
        "bias": "tensor",
        "stride": "tuple",
        "padding": "tuple",
        "output_padding": "tuple",
        "groups": "integer",
        "dilation": "tuple"
    },
    "inner": {},
}
signatures["torch.nn.functional.pdist"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "p": "float" # Could also be integer, but float is more general
    },
    "inner": {},
}
signatures["torch.nn.functional.dropout"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "p": "float",
        "training": "boolean",
        "inplace": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.dropout2d"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "p": "float",
        "training": "boolean",
        "inplace": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.dropout3d"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "p": "float",
        "training": "boolean",
        "inplace": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.feature_alpha_dropout"] = {
    "args": {
        "input": "tensor",
        "p": "float"
    },
    "kwargs": {
        "training": "boolean",
        "inplace": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.one_hot"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "num_classes": "integer"
    },
    "inner": {},
}
signatures["torch.nn.functional.fold"] = {
    "args": {
        "input": "tensor",
        "output_size": "tuple",
        "kernel_size": "tuple"
    },
    "kwargs": {
        "dilation": "integer",
        "padding": "integer",
        "stride": "integer"
    },
    "inner": {},
}
signatures["torch.nn.functional.unfold"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "tuple",
    },
    "kwargs": {
        "dilation": "integer",
        "padding": "integer",
        "stride": "integer",
    },
    "inner": {},
}
signatures["torch.nn.functional.instance_norm"] = {
    "args": {
        "input": "tensor",
        "running_mean": "tensor",
        "running_var": "tensor",
        "weight": "tensor",
        "bias": "tensor",
    },
    "kwargs": {
        "use_input_stats": "boolean",
        "momentum": "float",
        "eps": "float",
    },
    "inner": {},
}
signatures["torch.nn.functional.bilinear"] = {
    "args": {
        "input1": "tensor",
        "input2": "tensor",
        "weight": "tensor"
    },
    "kwargs": {
        "bias": "tensor"
    },
    "inner": {},
}
signatures["torch.nn.functional.elu"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "alpha": "float",
        "inplace": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.hardtanh"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "min_val": "float",
        "max_val": "float",
        "inplace": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.gumbel_softmax_1"] = {
    "args": {
        "logits": "tensor",
    },
    "kwargs": {
        "tau": "float",
        "hard": "boolean",
        "dim": "integer"
    },
    "inner": {},
}

signatures["torch.nn.functional.gumbel_softmax_2"] = {
    "args": {
        "logits": "tensor",
        "tau": "float",
    },
    "kwargs": {
        "hard": "boolean",
        "dim": "integer"
    },
    "inner": {},
}
signatures["torch.nn.functional.glu"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.functional.mish"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "inplace": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.tanhshrink"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.functional.threshold"] = {
    "args": {
        "input": "tensor",
        "threshold": "float",
        "value": "float"
    },
    "kwargs": {
        "inplace": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.kl_div"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "reduction": "string",
        "log_target": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.binary_cross_entropy"] = {
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
    "inner": {},
}
signatures["torch.nn.functional.hinge_embedding_loss"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "margin": "float",
        "reduction": "string"
    },
    "inner": {},
}
signatures["torch.nn.functional.multi_margin_loss"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "p": "integer",
        "margin": "float",
        "weight": "tensor",
        "size_average": "boolean", # deprecated
        "reduce": "boolean", # deprecated
        "reduction": "string"
    },
    "inner": {},
}
signatures["torch.nn.functional.cosine_embedding_loss"] = {
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
    "inner": {},
}
signatures["torch.nn.functional.smooth_l1_loss"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "reduction": "string",
        "delta": "float"
    },
    "inner": {},
}
signatures["torch.nn.functional.huber_loss"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "delta": "float",
        "reduction": "string"
    },
    "inner": {},
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
    "inner": {},
}
signatures["torch.nn.functional.soft_margin_loss"] = {
    "args": {
        "input": "tensor",
        "target": "tensor"
    },
    "kwargs": {
        "size_average": "boolean", # deprecated
        "reduce": "boolean", # deprecated
        "reduction": "string"
    },
    "inner": {},
}
signatures["torch.nn.functional.triplet_margin_loss"] = {
    "args": {
        "anchor": "tensor",
        "positive": "tensor",
        "negative": "tensor",
    },
    "kwargs": {
        "margin": "float",
        "p": "float",
        "eps": "float",
        "swap": "boolean",
        "reduction": "string"
    },
    "inner": {},
}
signatures["torch.nn.functional.triplet_margin_with_distance_loss"] = {
    "args": {
        "anchor": "tensor",
        "positive": "tensor",
        "negative": "tensor"
    },
    "kwargs": {
        "distance_function": "callable", # This should ideally be a callable but there isn't a callable option
        "margin": "float",
        "swap": "boolean",
        "reduction": "string"
    },
    "inner": {},
}
signatures["torch.nn.functional.ctc_loss"] = {
    "args": {
        "log_probs": "tensor",
        "targets": "tensor",
        "input_lengths": "tensor",
        "target_lengths": "tensor"
    },
    "kwargs": {
        "blank": "integer",
        "reduction": "string",
        "zero_infinity": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.avg_pool3d_1"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "integer",
        "ceil_mode": "boolean",
        "count_include_pad": "boolean",
        "divisor_override": "integer"
    },
    "inner": {},
}

signatures["torch.nn.functional.avg_pool3d_2"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "tuple",
        "ceil_mode": "boolean",
        "count_include_pad": "boolean",
        "divisor_override": "integer"
    },
    "inner": {},
}

signatures["torch.nn.functional.avg_pool3d_3"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "list" # Could also be a tuple of integers
    },
    "kwargs": {
        "stride": "list", # Could also be a tuple of integers
        "padding": "list", # Could also be a tuple of integers
        "ceil_mode": "boolean",
        "count_include_pad": "boolean",
        "divisor_override": "integer"
    },
    "inner": {},
}

signatures["torch.nn.functional.avg_pool3d_4"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "tuple",
        "ceil_mode": "boolean",
        "count_include_pad": "boolean",
        "divisor_override": "integer"
    },
    "inner": {},
}

signatures["torch.nn.functional.avg_pool3d_5"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "integer",
        "ceil_mode": "boolean",
        "count_include_pad": "boolean",
        "divisor_override": "integer"
    },
    "inner": {},
}

signatures["torch.nn.functional.avg_pool3d_6"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "list",
        "padding": "list",
        "ceil_mode": "boolean",
        "count_include_pad": "boolean",
        "divisor_override": "integer"
    },
    "inner": {},
}

signatures["torch.nn.functional.avg_pool3d_7"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "tuple"
    },
    "kwargs": {
        "stride": "list",
        "padding": "list",
        "ceil_mode": "boolean",
        "count_include_pad": "boolean",
        "divisor_override": "integer"
    },
    "inner": {},
}

signatures["torch.nn.functional.avg_pool3d_8"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "list"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "integer",
        "ceil_mode": "boolean",
        "count_include_pad": "boolean",
        "divisor_override": "integer"
    },
    "inner": {},
}

signatures["torch.nn.functional.avg_pool3d_9"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "list"
    },
    "kwargs": {
        "stride": "tuple",
        "padding": "tuple",
        "ceil_mode": "boolean",
        "count_include_pad": "boolean",
        "divisor_override": "integer"
    },
    "inner": {},
}
signatures["torch.nn.functional.adaptive_max_pool3d_1"] = {
    "args": {
        "input": "tensor",
        "output_size": "tuple"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.functional.adaptive_max_pool3d_2"] = {
    "args": {
        "input": "tensor",
        "output_size": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.functional.adaptive_avg_pool3d_1"] = {
    "args": {
        "input": "tensor",
        "output_size": "tuple"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.functional.adaptive_avg_pool3d_2"] = {
    "args": {
        "input": "tensor",
        "output_size": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.functional.max_unpool1d"] = {
    "args": {
        "input": "tensor",
        "indices": "tensor",
        "kernel_size": "integer"
    },
    "kwargs": {
        "stride": "integer",
        "padding": "integer",
        "output_size": "tuple"
    },
    "inner": {},
}


signatures["torch.nn.functional.max_unpool3d"] = {
    "args": {
        "input": "tensor",
        "indices": "tensor",
        "output_size": "tuple"
    },
    "kwargs": {
        "stride": "integer", #Could also be tuple
        "padding": "integer" #Could also be tuple
    },
    "inner": {},
}
signatures["torch.nn.functional.fractional_max_pool2d"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "tuple",
        "output_size": "tuple"
    },
    "kwargs": {
        "output_ratio": "tuple",
        "return_indices": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.fractional_max_pool3d_1"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "tuple",
        "output_size": "tuple"
    },
    "kwargs": {
        "output_ratio": "None", # Assuming None is not a type, creating separate signatures for output_ratio instead.
        "return_indices": "boolean"
    },
    "inner": {},
}

signatures["torch.nn.functional.fractional_max_pool3d_2"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "tuple",
        "output_size": "None"
    },
    "kwargs": {
        "output_ratio": "tuple",
        "return_indices": "boolean"
    },
    "inner": {},
}

signatures["torch.nn.functional.fractional_max_pool3d_3"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "integer",
        "output_size": "tuple"
    },
    "kwargs": {
        "output_ratio": "None",
        "return_indices": "boolean"
    },
    "inner": {},
}

signatures["torch.nn.functional.fractional_max_pool3d_4"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "integer",
        "output_size": "None"
    },
    "kwargs": {
        "output_ratio": "tuple",
        "return_indices": "boolean"
    },
    "inner": {},
}

signatures["torch.nn.functional.fractional_max_pool3d_5"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "tuple",
        "output_size": "integer" # Considering that the output size can be passed as int too.
    },
    "kwargs": {
        "output_ratio": "None",
        "return_indices": "boolean"
    },
    "inner": {},
}

signatures["torch.nn.functional.fractional_max_pool3d_6"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "tuple",
        "output_size": "None"
    },
    "kwargs": {
        "output_ratio": "tuple",
        "return_indices": "boolean"
    },
    "inner": {},
}

signatures["torch.nn.functional.fractional_max_pool3d_7"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "integer",
        "output_size": "integer"
    },
    "kwargs": {
        "output_ratio": "None",
        "return_indices": "boolean"
    },
    "inner": {},
}

signatures["torch.nn.functional.fractional_max_pool3d_8"] = {
    "args": {
        "input": "tensor",
        "kernel_size": "integer",
        "output_size": "None"
    },
    "kwargs": {
        "output_ratio": "tuple",
        "return_indices": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.local_response_norm"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "size": "integer",
        "alpha": "float",
        "beta": "float",
        "k": "float"
    },
    "inner": {},
}
signatures["torch.nn.functional.group_norm"] = {
    "args": {
        "input": "tensor",
        "num_groups": "integer"
    },
    "kwargs": {
        "weight": "tensor",
        "bias": "tensor",
        "eps": "float",
    },
    "inner": {},
}
signatures["torch.nn.functional.pixel_unshuffle"] = {
    "args": {
        "input": "tensor",
        "downscale_factor": "integer"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.functional.affine_grid"] = {
    "args": {
        "theta": "tensor",
        "size": "tuple"
    },
    "kwargs": {
        "align_corners": "boolean"
    },
    "inner": {},
}
signatures["torch.nn.functional.grid_sample"] = {
    "args": {
        "input": "tensor",
        "grid": "tensor"
    },
    "kwargs": {
        "mode": "string",
        "padding_mode": "string",
        "align_corners": "boolean"
    },
    "inner": {}
}

signatures["torch.nn.init.calculate_gain"] = {
    "args": {
        "nonlinearity": "string"
    },
    "kwargs": {},
    "inner": {},
}

signatures["torch.nn.init.constant_"] = {
    "args": {
        "tensor": "tensor",
        "val": "float" # Could also be integer or boolean, assuming float is most general
    },
    "kwargs": {},
    "inner": {},
}

signatures["torch.nn.init.dirac_"] = {
    "args": {
        "tensor": "tensor",
        "offset": "integer"
    },
    "kwargs": {},
    "inner": {},
}

signatures["torch.nn.init.eye_"] = {
    "args": {
        "tensor": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.init.normal_"] = {
    "args": {
        "tensor": "tensor",
        "mean": "float",
        "std": "float"
    },
    "kwargs": {},
    "inner": {},
}

signatures["torch.nn.init.ones_"] = {
    "args": {
        "tensor": "tensor"
    },
    "kwargs": {},
    "inner": {},
}

signatures["torch.nn.init.orthogonal_"] = {
    "args": {
        "tensor": "tensor"
    },
    "kwargs": {
        "gain": "float"
    },
    "inner": {},
}
signatures["torch.nn.init.sparse_"] = {
    "args": {
        "tensor": "tensor",
        "sparsity": "float"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.init.uniform_"] = {
    "args": {
        "tensor": "tensor",
        "a": "float",
        "b": "float"
    },
    "kwargs": {},
    "inner": {},
}


signatures["torch.nn.init.kaiming_normal_"] = {
    "args": {
        "tensor": "tensor",
        "a": "float",
        "mode": "string",
        "nonlinearity": "string"
    },
    "kwargs": {},
    "inner": {}
}
signatures["torch.nn.init.kaiming_uniform_"] = {
    "args": {
        "tensor": "tensor",
        "a": "float",
        "mode": "string",
        "nonlinearity": "string"
    },
    "kwargs": {},
    "inner": {},
}

signatures["torch.nn.init.xavier_normal_"] = {
    "args": {
        "tensor": "tensor",
        "gain": "float"
    },
    "kwargs": {},
    "inner": {},
}


signatures["torch.nn.init.xavier_uniform_"] = {
    "args": {
        "tensor": "tensor",
        "gain": "float"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.init.zeros_"] = {
    "args": {
        "tensor": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.utils.clip_grad_value__1"] = {
    "args": {
        "parameters": "tensor_list",
        "clip_value": "float"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.nn.utils.rnn.pack_padded_sequence"] = {
    "args": {
        "input": "tensor",
        "lengths": "tensor",
        "batch_first": "boolean",
        "enforce_sorted": "boolean"
    },
    "kwargs": {},
    "inner": {},
}


signatures["torch.nn.utils.rnn.pack_sequence"] = {
    "args": {
        "sequences": "tensor_list",
        "enforce_sorted": "boolean"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.special.polygamma"] = {
    "args": {
        "n": "integer",
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.special.digamma"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}

signatures["torch.special.erf"] = {
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
signatures["torch.special.logsumexp_1"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "dim": "integer",
        "keepdim": "boolean",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.special.logsumexp_2"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "dim": "list", # Could be a tuple of ints too
        "keepdim": "boolean",
        "out": "tensor"
    },
    "inner": {},
}


signatures["torch.special.logit_1"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "eps": "float"
    },
    "inner": {}
}

signatures["torch.special.logit_2"] = {
    "args": {
        "input": "tensor",
    },
    "kwargs": {
        "eps": "tensor"
    },
    "inner": {}
}
signatures["torch.special.exp2"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.special.expm1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.special.xlogy"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.special.zeta_1"] = {
    "args": {
        "x": "tensor",
        "q": "tensor"
    },
    "kwargs": {},
    "inner": {},
}

signatures["torch.special.zeta_2"] = {
    "args": {
        "x": "tensor",
        "q": "float"
    },
    "kwargs": {},
    "inner": {},
}

signatures["torch.special.zeta_3"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.special.i0"] = {
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
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.special.round"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.special.log1p"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.special.log_softmax_1"] = {
    "args": {
        "input": "tensor",
        "dim": "integer"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {},
}
signatures["torch.special.log_softmax_2"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dim": "integer",
        "dtype": "dtype"
    },
    "inner": {},
}

signatures = {}
signatures["torch.BoolStorage"] = {
    "size": "integer" # Could also accept a tuple, but integer is more common.
}
signatures["torch.ByteStorage"] = {
    "size": "integer"
}
signatures["torch.CharStorage"] = {
    "size": "integer" #Could also accept a tuple
}
signatures["torch.ComplexDoubleStorage"] = {
    "size": "integer"
}
signatures["torch.ComplexFloatStorage"] = {
    "size": "integer" # or "list"? It appears to accept a size that represents the length of the storage.
}
signatures["torch.DeserializationStorageContext"] = {
    "storage": "tensor", # Could also be Storage, but tensor seems more appropriate
    "offset": "integer",
    "size": "integer"
}
signatures["torch.DoubleStorage"] = {
    "size": "integer" # Could also be "list" or "tuple" of integers
}
signatures["torch.FloatStorage"] = {
    "size": "integer" # Could also be a tuple, but integer seems more common for initial size
}
signatures["torch.HalfStorage"] = {
    "size": "integer" # could also be a tuple, but integer is more common.
}
signatures["torch.IntStorage"] = {
    "size": "integer"
}
signatures["torch.LongStorage"] = {
    "size": "integer" # Could also be a tuple or list representing the size. Choosing integer as most common single value.
}
signatures["torch.SerializationStorageContext"] = {
    "tensor": "tensor",
}
signatures["torch.ShortStorage"] = {
    "size": "integer" # Could also accept a tuple for multi-dimensional storage, but single integer is more common
}
signatures["torch.TypedStorage"] = {
    "dtype": "dtype" # Could also be a string? Not sure.
}
signatures["torch.UntypedStorage"] = {
    "size": "integer"
}
signatures["torch.abs_"] = {
    "input": "tensor"
}
signatures["torch.absolute"] = {
    "input": "tensor"
}
signatures["torch.accelerator.cuda.is_available"] = {} # This API takes no arguments
signatures["torch.accelerator.cuda.current_device"] = {} # This API takes no arguments
signatures["torch.accelerator.cuda.device_count"] = {} # This API takes no arguments
signatures["torch.accelerator.cuda.get_device_name"] = {
    "device": "integer"
}
signatures["torch.accelerator.cuda.set_device"] = {
    "device": "integer"
}
signatures["torch.accelerator.cuda.empty_cache"] = {} # This API takes no arguments
signatures["torch.acos_"] = {
    "input": "tensor"
}
signatures["torch.acosh_"] = {
    "input": "tensor"
}
signatures["torch.addmv_"] = {
    "input": "tensor",
    "mat": "tensor",
    "vec": "tensor",
    "beta": "float",
    "alpha": "float"
}
signatures["torch.adjoint"] = {
    "input": "tensor"
}
signatures["torch.affine_grid_generator"] = {
    "theta": "tensor",
    "size": "list", # Could also be tuple, but list seems more common.
    "align_corners": "boolean"
}
signatures["torch.alias_copy"] = {
    "input": "tensor"
}
signatures["torch.align_tensors"] = {
    "tensors": "tensor_list" # Could also be tuple, but tensor_list seems more appropriate
}
signatures["torch.all"] = {
    "input": "tensor",
    "dim": "integer",
    "keepdim": "boolean"
}
signatures["torch.alpha_dropout"] = {
    "input": "tensor",
    "p": "float",
    "train": "boolean"
}
signatures["torch.alpha_dropout_"] = {
    "input": "tensor",
    "p": "float",
    "train": "boolean"
}
signatures["torch.aminmax"] = {
    "input": "tensor",
    "dim": "integer", # could be None, but integer seems more common
    "keepdim": "boolean"
}
signatures["torch.amp.autocast"] = {
    "enabled": "boolean",
    "dtype": "dtype"
}
signatures["torch.amp.GradScaler"] = {
    "init_scale": "float",
    "growth_factor": "float",
    "backoff_factor": "float",
    "growth_interval": "integer",
    "enabled": "boolean"
}
signatures["torch.amp.float16_conversion"] = {
    "tensor": "tensor"
}
signatures["torch.any"] = {
    "input": "tensor",
    "dim": "integer",
    "keepdim": "boolean"
}
signatures["torch.ao.quantize"] = {
    "model": "tensor", # Could be a nn.Module object, but internally it's a collection of tensors.  Best approximation.
    "qconfig_mapping": "list", # Usually a list of QConfigMapping objects
    "inplace": "boolean"
}
signatures["torch.ao.quantize_dynamic"] = {
    "model": "tensor", # Could be a nn.Module object
    "qconfig_mapping": "list", # Usually a list of QConfigMapping objects
    "dtype": "dtype",
    "inplace": "boolean"
}
signatures["torch.ao.quantization.get_default_qconfig"] = {
    "backend": "string"
}
signatures["torch.ao.quantization.get_default_qconfig_mapping"] = {
    "backend": "string"
}
signatures["torch.ao.quantization.prepare"] = {
    "model": "tensor", # Could be a nn.Module object
    "qconfig_mapping": "list",
    "example_input": "tensor",
    "inplace": "boolean",
    "remove_weight_observers": "boolean"
}
signatures["torch.ao.quantization.convert"] = {
    "model": "tensor", # Could be a nn.Module object
    "inplace": "boolean"
}
signatures["torch.ao.quantization.prepare_qat"] = {
    "model": "tensor", # Could be a nn.Module object
    "qconfig_mapping": "list",
    "example_input": "tensor",
    "inplace": "boolean"
}
signatures["torch.ao.quantization.QuantStub"] = {} # No arguments
signatures["torch.ao.quantization.DeQuantStub"] = {} # No arguments
signatures["torch.ao.quantization.ObserverBase"] = {
    "dtype": "dtype",
    "qscheme": "string",
    "reduce_range": "boolean"
}
signatures["torch.ao.quantization.MinMaxObserver"] = {
    "dtype": "dtype",
    "qscheme": "string",
    "reduce_range": "boolean",
    "quant_min": "integer",
    "quant_max": "integer"
}
signatures["torch.ao.quantization.MovingAvgMinMaxObserver"] = {
    "dtype": "dtype",
    "qscheme": "string",
    "reduce_range": "boolean",
    "quant_min": "integer",
    "quant_max": "integer",
    "averaging_constant": "float"
}
signatures["torch.ao.quantization.HistogramObserver"] = {
    "dtype": "dtype",
    "qscheme": "string",
    "reduce_range": "boolean",
    "quant_min": "integer",
    "quant_max": "integer",
    "nbins": "integer"
}
signatures["torch.ao.quantization.default_qconfig"] = {
    "backend": "string" # Or None, assuming string is more common
}
signatures["torch.arccos"] = {
    "input": "tensor"
}
signatures["torch.arccos_"] = {
    "input": "tensor"
}
signatures["torch.arccosh"] = {
    "input": "tensor"
}
signatures["torch.arccosh_"] = {
    "input": "tensor"
}
signatures["torch.arcsin"] = {
    "input": "tensor"
}
signatures["torch.arcsin_"] = {
    "input": "tensor"
}
signatures["torch.arcsinh"] = {
    "input": "tensor"
}
signatures["torch.arcsinh_"] = {
    "input": "tensor"
}
signatures["torch.arctan"] = {
    "input": "tensor"
}
signatures["torch.arctan_"] = {
    "input": "tensor"
}
signatures["torch.arctanh"] = {
    "input": "tensor"
}
signatures["torch.arctanh_"] = {
    "input": "tensor"
}
signatures["torch.are_deterministic_algorithms_enabled"] = {}
signatures["torch.argwhere"] = {
    "input": "tensor"
}
signatures["torch.as_strided_"] = {
    "input": "tensor",
    "size": "tuple",
    "stride": "tuple",
    "storage_offset": "integer"
}
signatures["torch.as_strided_copy"] = {
    "input": "tensor",
    "size": "tuple",
    "stride": "tuple"
}
signatures["torch.as_strided_scatter"] = {
    "input": "tensor",
    "source": "tensor",
    "size": "tuple",
    "stride": "tuple",
    "storage_offset": "integer" # Could be "integer" or "tuple" depending on the version or common usage. I chose "integer"
}
signatures["torch.as_tensor"] = {
    "data": "list", # Could also be tensor, but most common usage is with list, tuple or numbers
    "dtype": "dtype",
    "device": "string" #This argument is skipped because you asked for it
}
signatures["torch.asarray"] = {
    "obj": "list", # Could also be "tensor" or other types. Choosing "list" because it's a general container.
    "dtype": "dtype",
    "copy": "boolean",
    "requires_grad": "boolean"
}
signatures["torch.asin_"] = {
    "input": "tensor"
}
signatures["torch.asinh_"] = {
    "input": "tensor"
}
signatures["torch.atan_"] = {
    "input": "tensor"
}
signatures["torch.atanh_"] = {
    "input": "tensor"
}
signatures["torch.autocast"] = {
    "device_type": "string", # Could be an enum but string seems most appropriate
    "enabled": "boolean"
}
signatures["torch.autocast_decrement_nesting"] = {}
signatures["torch.autocast_increment_nesting"] = {}
signatures["torch.autograd.backward"] = {
    "tensors": "tensor_list",
    "grad_tensors": "tensor_list",
    "retain_graph": "boolean",
    "create_graph": "boolean",
    "grad_variables": "tensor_list" # Deprecated argument, keep for compatibility
}
signatures["torch.autograd.grad"] = {
    "outputs": "tensor_list",
    "inputs": "tensor_list",
    "grad_outputs": "tensor_list",
    "retain_graph": "boolean",
    "create_graph": "boolean",
    "only_inputs": "boolean",
    "allow_unused": "boolean"
}
signatures["torch.autograd.functional.jacobian"] = {
    "func": "list", # Function, approximating it with list
    "inputs": "tensor"
}
signatures["torch.autograd.functional.hessian"] = {
    "func": "list", # Function, approximating it with list
    "inputs": "tensor"
}
signatures["torch.backends.cudnn.benchmark"] = {
    "benchmark": "boolean"
}
signatures["torch.backends.cudnn.deterministic"] = {
    "deterministic": "boolean"
}
signatures["torch.backends.cudnn.enabled"] = {
    "enabled": "boolean"
}
signatures["torch.backends.cudnn.version"] = {}
signatures["torch.backends.cuda.is_built"] = {}
signatures["torch.backends.mkldnn.enabled"] = {
    "enabled": "boolean"
}
signatures["torch.backends.mkl.is_available"] = {}
signatures["torch.backends.openmp.is_available"] = {}
signatures["torch.backends.openmp.max_threads"] = {
    "threads": "integer" # Could potentially be None as well.
}
signatures["torch.bartlett_window"] = {
    "window_length": "integer",
    "periodic": "boolean",
    "dtype": "dtype"
}
signatures["torch.batch_norm"] = {
    "input": "tensor",
    "running_mean": "tensor",
    "running_var": "tensor",
    "weight": "tensor",
    "bias": "tensor",
    "training": "boolean",
    "momentum": "float",
    "eps": "float",
    "cudnn_enabled": "boolean" # could be None (treated as boolean), or bool
}
signatures["torch.batch_norm_backward_elemt"] = {
    "grad_out": "tensor",
    "input": "tensor",
    "mean": "tensor",
    "invstd": "tensor",
    "weight": "tensor",
    "mean_dy": "tensor",
    "mean_dy_xmu": "tensor",
    "norm_except_dim": "integer"
}
signatures["torch.batch_norm_backward_reduce"] = {
    "grad_out": "tensor",
    "input": "tensor",
    "mean": "tensor",
    "ivar": "tensor",
    "weight": "tensor",
    "grad_input_mask": "integer", # Could be a list of integers but it's treated as a bitmask.
}
signatures["torch.batch_norm_elemt"] = {
    "input": "tensor",
    "weight": "tensor",
    "bias": "tensor",
    "mean": "tensor",
    "invstd": "tensor",
    "eps": "float"
}
signatures["torch.batch_norm_gather_stats"] = {
    "input": "tensor",
    "mean": "tensor",
    "invstd": "tensor",
    "running_mean": "tensor",
    "running_var": "tensor",
    "momentum": "float",
    "eps": "float",
    "count": "integer" # Could be a tensor of one element
}
signatures["torch.batch_norm_gather_stats_with_counts"] = {
    "mean": "tensor",
    "ivar": "tensor",
    "input": "tensor",
    "batch_size": "list", # Could also be tuple
    "running_mean": "tensor",
    "running_ivar": "tensor",
    "momentum": "float",
    "eps": "float",
    "count": "integer"
}
signatures["torch.batch_norm_stats"] = {
    "input": "tensor",
    "weight": "tensor",
    "bias": "tensor",
    "running_mean": "tensor",
    "running_var": "tensor",
    "eps": "float"
}
signatures["torch.batch_norm_update_stats"] = {
    "input": "tensor",
    "running_mean": "tensor",
    "running_var": "tensor",
    "momentum": "float"
}
signatures["torch.bernoulli"] = {
    "input": "tensor",
    "generator": "torch.Generator" # Should be a Generator object, but that's not in the list
}
signatures["torch.bilinear"] = {
    "input1": "tensor",
    "input2": "tensor",
    "weight": "tensor",
    "bias": "tensor"
}
signatures["torch.binary_cross_entropy_with_logits"] = {
    "input": "tensor",
    "target": "tensor",
    "weight": "tensor",
    "size_average": "boolean",
    "reduce": "boolean",
    "reduction": "string",
    "pos_weight": "tensor" # Could be a list of floats but tensor seems more common
}
signatures["torch.binomial"] = {
    "count": "tensor",
    "prob": "tensor"
}
signatures["torch.bitwise_left_shift"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.bitwise_right_shift"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.blackman_window"] = {
    "n": "integer",
    "periodic": "boolean"
}
signatures["abs"] = {
    "input": "tensor"
}
signatures["acos"] = {
    "input": "tensor"
}
signatures["acosh"] = {
    "input": "tensor"
}
signatures["add"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["addcdiv"] = {
    "input": "tensor",
    "value": "float",
    "tensor1": "tensor",
    "tensor2": "tensor"
}
signatures["addcmul"] = {
    "input": "tensor",
    "value": "float",
    "tensor1": "tensor",
    "tensor2": "tensor"
}
signatures["angle"] = {
    "input": "tensor"
}
signatures["argmax"] = {
    "input": "tensor",
    "dim": "integer",
    "keepdim": "boolean"
}
signatures["argmin"] = {
    "input": "tensor",
    "dim": "integer",
    "keepdim": "boolean"
}
signatures["asin"] = {
    "input": "tensor"
}
signatures["asinh"] = {
    "input": "tensor"
}
signatures["atan"] = {
    "input": "tensor"
}
signatures["atan2"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["atanh"] = {
    "input": "tensor"
}
signatures["baddbmm"] = {
    "input": "tensor",
    "batch1": "tensor",
    "batch2": "tensor"
}
signatures["bitwise_and"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["bitwise_not"] = {
    "input": "tensor"
}
signatures["bitwise_or"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["bitwise_xor"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["broadcast_to"] = {
    "input": "tensor",
    "size": "tuple" # should it be tuple or list?
}
signatures["ceil"] = {
    "input": "tensor"
}
signatures["clamp"] = {
    "input": "tensor",
    "min": "float",
    "max": "float"
}
signatures["clip"] = {
    "input": "tensor",
    "min": "float",
    "max": "float"
}
signatures["conj"] = {
    "input": "tensor"
}
signatures["copysign"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["cos"] = {
    "input": "tensor"
}
signatures["cosh"] = {
    "input": "tensor"
}
signatures["cross"] = {
    "input": "tensor",
    "other": "tensor",
    "dim": "integer"
}
signatures["cumprod"] = {
    "input": "tensor",
    "dim": "integer"
}
signatures["cumsum"] = {
    "input": "tensor",
    "dim": "integer"
}
signatures["deg2rad"] = {
    "input": "tensor"
}
signatures["diff"] = {
    "input": "tensor",
    "n": "integer",
    "dim": "integer"
}
signatures["digamma"] = {
    "input": "tensor"
}
signatures["div"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["dot"] = {
    "input": "tensor",
    "tensor": "tensor"
}
signatures["eq"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["equal"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["exp"] = {
    "input": "tensor"
}
signatures["expm1"] = {
    "input": "tensor"
}
signatures["fake_quantize_per_channel_affine"] = {
    "input": "tensor",
    "scale": "tensor",
    "zero_point": "tensor",
    "channel_axis": "integer",
    "quant_min": "integer",
    "quant_max": "integer"
}
signatures["fake_quantize_per_tensor_affine"] = {
    "input": "tensor",
    "scale": "float",
    "zero_point": "integer",
    "quant_min": "integer",
    "quant_max": "integer"
}
signatures["fft"] = {
    "input": "tensor",
    "signal_ndim": "integer",
    "forward": "boolean"
}
signatures["floor"] = {
    "input": "tensor"
}
signatures["floor_divide"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["fmod"] = {
    "input": "tensor",
    "divisor": "tensor"
}
signatures["frac"] = {
    "input": "tensor"
}
signatures["frexp"] = {
    "input": "tensor"
}
signatures["gcd"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["ge"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["greater"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["greater_equal"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["igamma"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["igammac"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["imag"] = {
    "input": "tensor"
}
signatures["index_select"] = {
    "input": "tensor",
    "dim": "integer",
    "index": "tensor"
}
signatures["int_repr"] = {
    "input": "tensor"
}
signatures["inverse"] = {
    "input": "tensor"
}
signatures["isfinite"] = {
    "input": "tensor"
}
signatures["isinf"] = {
    "input": "tensor"
}
signatures["isnan"] = {
    "input": "tensor"
}
signatures["isneginf"] = {
    "input": "tensor"
}
signatures["isposinf"] = {
    "input": "tensor"
}
signatures["lcm"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["ldexp"] = {
    "input": "tensor",
    "exponent": "tensor"
}
signatures["le"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["less"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["less_equal"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["lgamma"] = {
    "input": "tensor"
}
signatures["log"] = {
    "input": "tensor"
}
signatures["log10"] = {
    "input": "tensor"
}
signatures["log1p"] = {
    "input": "tensor"
}
signatures["log2"] = {
    "input": "tensor"
}
signatures["log_softmax"] = {
    "input": "tensor",
    "dim": "integer"
}
signatures["logical_and"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["logical_not"] = {
    "input": "tensor"
}
signatures["logical_or"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["logical_xor"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["matmul"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["max"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["maximum"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["mean"] = {
    "input": "tensor",
    "dim": "list" #can also be integer/tuple
}
signatures["min"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["minimum"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["mm"] = {
    "input": "tensor",
    "mat2": "tensor"
}
signatures["mod"] = {
    "input": "tensor",
    "divisor": "tensor"
}
signatures["mul"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["multiply"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["mvlgamma"] = {
    "input": "tensor",
    "p": "integer"
}
signatures["ne"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["negative"] = {
    "input": "tensor"
}
signatures["nextafter"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["normal"] = {
    "mean": "float",
    "std": "float",
    "size": "list" #should be a list or tuple?
}
signatures["not_equal"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["ones_like"] = {
    "input": "tensor"
}
signatures["permute"] = {
    "input": "tensor",
    "dims": "tuple"
}
signatures["polygamma"] = {
    "n": "integer",
    "input": "tensor"
}
signatures["positive"] = {
    "input": "tensor"
}
signatures["pow"] = {
    "input": "tensor",
    "exponent": "tensor"
}
signatures["prod"] = {
    "input": "tensor",
    "dim": "integer"
}
signatures["rad2deg"] = {
    "input": "tensor"
}
signatures["rand_like"] = {
    "input": "tensor"
}
signatures["randint_like"] = {
    "input": "tensor",
    "low": "integer",
    "high": "integer"
}
signatures["randn_like"] = {
    "input": "tensor"
}
signatures["real"] = {
    "input": "tensor"
}
signatures["reciprocal"] = {
    "input": "tensor"
}
signatures["remainder"] = {
    "input": "tensor",
    "divisor": "tensor"
}
signatures["round"] = {
    "input": "tensor"
}
signatures["rsqrt"] = {
    "input": "tensor"
}
signatures["scatter"] = {
    "input": "tensor",
    "dim": "integer",
    "index": "tensor",
    "src": "tensor"
}
signatures["sgn"] = {
    "input": "tensor"
}
signatures["sigmoid"] = {
    "input": "tensor"
}
signatures["signbit"] = {
    "input": "tensor"
}
signatures["sin"] = {
    "input": "tensor"
}
signatures["sinc"] = {
    "input": "tensor"
}
signatures["sinh"] = {
    "input": "tensor"
}
signatures["softmax"] = {
    "input": "tensor",
    "dim": "integer"
}
signatures["sqrt"] = {
    "input": "tensor"
}
signatures["square"] = {
    "input": "tensor"
}
signatures["sub"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["sum"] = {
    "input": "tensor",
    "dim": "list" #can also be integer/tuple
}
signatures["tan"] = {
    "input": "tensor"
}
signatures["tanh"] = {
    "input": "tensor"
}
signatures["trunc"] = {
    "input": "tensor"
}
signatures["true_divide"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["zeros_like"] = {
    "input": "tensor"
}
signatures["fft_fftfreq"] = {
    "n": "integer",
    "d": "float"
}
signatures["fft_rfftfreq"] = {
    "n": "integer",
    "d": "float"
}
signatures["broadcast_tensors"] = {
    "tensors": "tensor_list"
}
signatures["cdist"] = {
    "x1": "tensor",
    "x2": "tensor",
    "p": "float"
}
signatures["chain_matmul"] = {
    "matrices": "tensor_list"
}
signatures["combinations"] = {
    "input": "tensor",
    "r": "integer"
}
signatures["corrcoef"] = {
    "input": "tensor"
}
signatures["cov"] = {
    "input": "tensor",
    "correction": "integer",
    "fweights": "tensor",
    "aweights": "tensor"
}
signatures["cross_entropy"] = {
    "input": "tensor",
    "target": "tensor"
}
signatures["det"] = {
    "input": "tensor"
}
signatures["diag"] = {
    "input": "tensor",
    "diagonal": "integer"
}
signatures["diagflat"] = {
    "input": "tensor",
    "offset": "integer"
}
signatures["diagonal"] = {
    "input": "tensor",
    "offset": "integer",
    "dim1": "integer",
    "dim2": "integer"
}
signatures["einsum"] = {
    "equation": "string",
    "tensors": "tensor_list"
}
signatures["flatten"] = {
    "input": "tensor",
    "start_dim": "integer",
    "end_dim": "integer"
}
signatures["flip"] = {
    "input": "tensor",
    "dims": "list" #can be integer/tuple
}
signatures["fliplr"] = {
    "input": "tensor"
}
signatures["flipud"] = {
    "input": "tensor"
}
signatures["float_power"] = {
    "input": "tensor",
    "exponent": "tensor"
}
signatures["histc"] = {
    "input": "tensor",
    "bins": "integer",
    "min": "float",
    "max": "float"
}
signatures["i0"] = {
    "input": "tensor"
}
signatures["fft_ifft"] = {
    "input": "tensor",
    "signal_ndim": "integer",
    "normalized": "boolean"
}
signatures["fft_ihfft"] = {
    "input": "tensor",
    "signal_ndim": "integer",
    "normalized": "boolean"
}
signatures["fft_irfft"] = {
    "input": "tensor",
    "signal_ndim": "integer",
    "normalized": "boolean"
}
signatures["fft_rfft"] = {
    "input": "tensor",
    "signal_ndim": "integer",
    "normalized": "boolean"
}
signatures["isclose"] = {
    "input": "tensor",
    "other": "tensor",
    "rtol": "float",
    "atol": "float",
    "equal_nan": "boolean"
}
signatures["kron"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["kthvalue"] = {
    "input": "tensor",
    "k": "integer",
    "dim": "integer"
}
signatures["lstsq"] = {
    "input": "tensor",
    "b": "tensor"
}
signatures["lu"] = {
    "input": "tensor"
}
signatures["masked_fill"] = {
    "input": "tensor",
    "mask": "tensor",
    "value": "float"
}
signatures["masked_scatter"] = {
    "input": "tensor",
    "mask": "tensor",
    "source": "tensor"
}
signatures["matrix_power"] = {
    "input": "tensor",
    "n": "integer"
}
signatures["moveaxis"] = {
    "input": "tensor",
    "source": "integer",
    "destination": "integer"
}
signatures["movedim"] = {
    "input": "tensor",
    "source": "integer",
    "destination": "integer"
}
signatures["msort"] = {
    "input": "tensor"
}
signatures["narrow"] = {
    "input": "tensor",
    "dim": "integer",
    "start": "integer",
    "length": "integer"
}
signatures["ndimension"] = {
    "input": "tensor"
}
signatures["numel"] = {
    "input": "tensor"
}
signatures["pca_lowrank"] = {
    "input": "tensor",
    "q": "integer"
}
signatures["pinverse"] = {
    "input": "tensor",
    "rcond": "float"
}
signatures["polar"] = {
    "abs": "tensor",
    "angle": "tensor"
}
signatures["quantile"] = {
    "input": "tensor",
    "q": "float"
}
signatures["ravel"] = {
    "input": "tensor"
}
signatures["repeat_interleave"] = {
    "input": "tensor",
    "repeats": "integer",
    "dim": "integer"
}
signatures["rot90"] = {
    "input": "tensor",
    "k": "integer",
    "dims": "tuple"
}
signatures["row_indices"] = {
    "input": "tensor"
}
signatures["segment_reduce"] = {
    "data": "tensor",
    "segment_ids": "tensor",
    "reduce": "string",
    "init": "float"
}
signatures["select"] = {
    "input": "tensor",
    "dim": "integer",
    "index": "integer"
}
signatures["select_scatter"] = {
    "input": "tensor",
    "src": "tensor",
    "dim": "integer",
    "index": "integer"
}
signatures["set_printoptions"] = {
    "precision": "integer",
    "threshold": "float",
    "edgeitems": "integer",
    "linewidth": "integer",
    "profile": "string",
    "sci_mode": "boolean"
}
signatures["slogdet"] = {
    "input": "tensor"
}
signatures["sort"] = {
    "input": "tensor",
    "dim": "integer",
    "descending": "boolean"
}
signatures["space_split"] = {
    "input": "tensor",
    "num_blocks": "list"
}
signatures["svd"] = {
    "input": "tensor"
}
signatures["take"] = {
    "input": "tensor",
    "index": "tensor"
}
signatures["tensordot"] = {
    "input": "tensor",
    "other": "tensor",
    "dims": "list" #can be integer/tuple
}
signatures["tile"] = {
    "input": "tensor",
    "reps": "tuple" #should be tuple or list?
}
signatures["trace"] = {
    "input": "tensor"
}
signatures["transpose"] = {
    "input": "tensor",
    "dim0": "integer",
    "dim1": "integer"
}
signatures["trapz"] = {
    "y": "tensor",
    "x": "tensor",
    "dim": "integer"
}
signatures["triangular_solve"] = {
    "input": "tensor",
    "A": "tensor",
    "upper": "boolean",
    "transpose": "boolean",
    "unitriangular": "boolean"
}
signatures["unfold"] = {
    "input": "tensor",
    "dimension": "integer",
    "size": "integer",
    "step": "integer"
}
signatures["unique"] = {
    "input": "tensor"
}
signatures["unique_consecutive"] = {
    "input": "tensor"
}
signatures["values"] = {
    "input": "tensor"
}
signatures["vdot"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["view_as_real"] = {
    "input": "tensor"
}
signatures["view_as_complex"] = {
    "input": "tensor"
}
signatures["where"] = {
    "condition": "tensor",
    "input": "tensor",
    "other": "tensor"
}
signatures["fft_hfft"] = {
    "input": "tensor",
    "signal_ndim": "integer",
    "normalized": "boolean"
}
signatures["bartlett_window"] = {
    "n": "integer"
}
signatures["blackman_window"] = {
    "n": "integer"
}
signatures["hamming_window"] = {
    "n": "integer"
}
signatures["hann_window"] = {
    "n": "integer"
}
signatures["torch.can_cast"] = {
    "from": "dtype",
    "to": "dtype"
}
signatures["torch.ccol_indices_copy"] = {
    "input": "tensor",
    "indices": "tensor" # Could also be integer but tensor seems more common for indexing
}
signatures["torch.cdist"] = {
    "x1": "tensor",
    "x2": "tensor",
    "p": "float",
    "compute_mode": "string"
}
signatures["torch.ceil_"] = {
    "input": "tensor"
}
signatures["torch.celu"] = {
    "input": "tensor",
    "alpha": "float" # Could potentially be "tensor" if there's a variant accepting a tensor for alpha. But float is more common.
}
signatures["torch.celu_"] = {
    "input": "tensor",
    "alpha": "float"
}
signatures["torch.channel_shuffle"] = {
    "input": "tensor",
    "groups": "integer"
}
signatures["torch.cholesky"] = {
    "input": "tensor",
    "upper": "boolean"
}
signatures["torch.cholesky_solve"] = {
    "input": "tensor",
    "L": "tensor"
}
signatures["torch.choose_qparams_optimized"] = {
    "input": "tensor",
    "numel": "integer", # Could be a more specific integer type if available
    "n_quantized_bins": "integer",
    "qparams": "tensor",
    "dtype": "dtype",
    "is_signed": "boolean"
}
signatures["torch.clamp_"] = {
    "input": "tensor",
    "min": "float", # or "tensor", but float is more common
    "max": "float"  # or "tensor", but float is more common
}
signatures["torch.clamp_max"] = {
    "input": "tensor",
    "max": "float" # Could be integer or tensor too but float seems most common
}
signatures["torch.clamp_max_"] = {
    "input": "tensor",
    "max": "float" # Could potentially be an integer as well, but float is more common
}
signatures["torch.clamp_min"] = {
    "input": "tensor",
    "min": "float" # could also be a tensor
}
signatures["torch.clamp_min_"] = {
    "input": "tensor",
    "min": "float" # Could also accept a tensor, but float seems more common
}
signatures["torch.classes.load_library"] = {
    "path": "string"
}
signatures["torch.classes.register_module"] = {
    "module_name": "string",
    "module_class": "string" # Ideally this would be a type or class object but closest type is "string"
}
signatures["torch.classes.register_class"] = {
    "class_name": "string",
    "class_type": "string" # Ideally this would be a type or class object but closest type is "string"
}
signatures["torch.classes.register_python_class"] = {
    "module_name": "string",
    "class_name": "string",
    "python_class": "string" # Ideally this would be a type or class object but closest type is "string"
}
signatures["torch.classes.list_modules"] = {}
signatures["torch.classes.list_classes"] = {}
signatures["torch.classproperty"] = { # This decorator doesn't accept arguments
}
signatures["torch.clear_autocast_cache"] = {}
signatures["torch.clip"] = {
    "input": "tensor",
    "min": "float", # Could also be tensor or None
    "max": "float" # Could also be tensor or None
}
signatures["torch.clip_"] = {
    "input": "tensor",
    "min": "float", # Can also be a tensor. Choosing float as it's more common for scalar bounds.
    "max": "float" # Can also be a tensor. Choosing float as it's more common for scalar bounds.
}
signatures["torch.clone"] = {
    "input": "tensor",
    "memory_format": "string" # Actually torch.memory_format, but string is the closest
}
signatures["torch.col_indices_copy"] = {
    "self": "tensor",
    "indices": "tensor",
    "source": "tensor"
}
signatures["torch.column_stack"] = {
    "tensors": "tensor_list"
}
signatures["torch.compile"] = {
    "model": "Callable", # Could also be NoneType, but Callable seems more appropriate given the description
    "fullgraph": "boolean",
    "dynamic": "boolean", # Could be Optional[bool], but boolean is the best representation here
    "backend": "string", # Could be Callable, but string is the most common default
    "mode": "string",
    "options": "dict",
    "disable": "boolean"
}
signatures["torch.compile"] = {
    "model": "tensor", # Should it be callable instead?
    "options": "dict", # Should it be a dictionary?
    "dynamic": "boolean",
    "backend": "string",
    "fullgraph": "boolean",
    "mode": "string",
    "config": "string",
    "debug": "boolean"
}
signatures["torch.concat"] = {
    "tensors": "tensor_list",
    "dim": "integer"
}
signatures["torch.concatenate"] = {
    "tensors": "tensor_list",
    "axis": "integer"
}
signatures["torch.cond"] = {
    "pred": "boolean", # Could also be a tensor, but boolean is more common in simple if statements
    "true_fn": "Callable", #There is no callable type, but this is what it takes.
    "false_fn": "Callable", #There is no callable type, but this is what it takes.
    "operands": "tuple"
}
signatures["torch.conj"] = {
    "input": "tensor"
}
signatures["torch.conj_physical"] = {
    "input": "tensor"
}
signatures["torch.conj_physical_"] = {
    "input": "tensor"
}
signatures["torch.constant_pad_nd"] = {
    "input": "tensor",
    "pad": "list", # Or maybe tuple, list is more common for pad
    "value": "float" # Or integer, but float is more general
}
signatures["torch.conv_tbc"] = {
    "input": "tensor",
    "weight": "tensor",
    "bias": "tensor",
    "pad": "integer",
    "stride": "integer",
    "dilation": "integer"
}
signatures["torch.convolution"] = {
    "input": "tensor",
    "weight": "tensor",
    "bias": "tensor",
    "stride": "tuple", # or integer
    "padding": "tuple", # or string, or integer
    "dilation": "tuple", # or integer
    "transposed": "boolean",
    "output_padding": "tuple", # or integer
    "groups": "integer"
}
signatures["torch.corrcoef"] = {
    "input": "tensor"
}
signatures["torch.cos_"] = {
    "input": "tensor"
}
signatures["torch.cosh_"] = {
    "input": "tensor"
}
signatures["torch.cosine_embedding_loss"] = {
    "input1": "tensor",
    "input2": "tensor",
    "target": "tensor",
    "margin": "float",
    "size_average": "boolean",
    "reduce": "boolean",
    "reduction": "string"
}
signatures["torch.cosine_similarity"] = {
    "x1": "tensor",
    "x2": "tensor",
    "dim": "integer",
    "eps": "float"
}
signatures["torch.cov"] = {
    "input": "tensor",
    "correction": "integer",
    "fweights": "tensor",
    "aweights": "tensor"
}

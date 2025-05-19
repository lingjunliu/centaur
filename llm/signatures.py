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
signatures["torch.abs"] = {
    "input": "tensor"
}
signatures["torch.acos"] = {
    "input": "tensor"
}
signatures["torch.acosh"] = {
    "input": "tensor"
}
signatures["torch.add"] = {
    "input": "tensor",
    "other": "tensor",
    "alpha": "float"
}
signatures["torch.addmm"] = {
    "input": "tensor",
    "mat1": "tensor",
    "mat2": "tensor",
    "beta": "float",
    "alpha": "float"
}
signatures["torch.all"] = {
    "input": "tensor",
    "dim": "integer",
    "keepdim": "boolean"
}
signatures["torch.any"] = {
    "input": "tensor",
    "dim": "integer",
    "keepdim": "boolean"
}
signatures["torch.arange"] = {
    "start": "float",
    "end": "float",
    "step": "float",
    "dtype": "dtype"
}
signatures["torch.argmax"] = {
    "input": "tensor",
    "dim": "integer",
    "keepdim": "boolean"
}
signatures["torch.argmin"] = {
    "input": "tensor",
    "dim": "integer",
    "keepdim": "boolean"
}
signatures["torch.asin"] = {
    "input": "tensor"
}
signatures["torch.asinh"] = {
    "input": "tensor"
}
signatures["torch.atan"] = {
    "input": "tensor"
}
signatures["torch.atanh"] = {
    "input": "tensor"
}
signatures["torch.atan2"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.baddbmm"] = {
    "input": "tensor",
    "batch1": "tensor",
    "batch2": "tensor",
    "beta": "float",
    "alpha": "float"
}
signatures["torch.bernoulli"] = {
    "input": "tensor"
}
signatures["torch.binary_cross_entropy"] = {
    "input": "tensor",
    "target": "tensor",
    "weight": "tensor",
    "reduction": "string"
}
signatures["torch.bitwise_and"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.bitwise_not"] = {
    "input": "tensor"
}
signatures["torch.bitwise_or"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.bitwise_xor"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.bmm"] = {
    "input": "tensor",
    "mat2": "tensor"
}
signatures["torch.broadcast_to"] = {
    "input": "tensor",
    "size": "tuple" # This could be a list as well, but tuple is more common for shapes
}
signatures["torch.cat"] = {
    "tensors": "tensor_list",
    "dim": "integer"
}
signatures["torch.ceil"] = {
    "input": "tensor"
}
signatures["torch.clamp"] = {
    "input": "tensor",
    "min": "float",
    "max": "float"
}
signatures["torch.clip"] = {
    "input": "tensor",
    "min": "float",
    "max": "float"
}
signatures["torch.clone"] = {
    "input": "tensor"
}
signatures["torch.combinations"] = {
    "input": "tensor",
    "r": "integer",
    "with_replacement": "boolean"
}
signatures["torch.complex"] = {
    "real": "tensor",
    "imag": "tensor"
}
signatures["torch.conj"] = {
    "input": "tensor"
}
signatures["torch.conj_physical"] = {
    "input": "tensor"
}
signatures["torch.cos"] = {
    "input": "tensor"
}
signatures["torch.cosh"] = {
    "input": "tensor"
}
signatures["torch.count_nonzero"] = {
    "input": "tensor",
    "dim": "integer" # Optional argument, can be None
}
signatures["torch.cross"] = {
    "input": "tensor",
    "other": "tensor",
    "dim": "integer"
}
signatures["torch.cumprod"] = {
    "input": "tensor",
    "dim": "integer",
    "dtype": "dtype"
}
signatures["torch.cumsum"] = {
    "input": "tensor",
    "dim": "integer",
    "dtype": "dtype"
}
signatures["torch.deg2rad"] = {
    "input": "tensor"
}
signatures["torch.diff"] = {
    "input": "tensor",
    "n": "integer",
    "dim": "integer",
    "prepend": "tensor",
    "append": "tensor"
}
signatures["torch.digamma"] = {
    "input": "tensor"
}
signatures["torch.div"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.dot"] = {
    "input": "tensor",
    "tensor": "tensor"
}
signatures["torch.dropout"] = {
    "input": "tensor",
    "p": "float",
    "train": "boolean"
}
signatures["torch.empty"] = {
    "size": "tuple", # or list of integers, but tuple is more common
    "dtype": "dtype"
}
signatures["torch.eq"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.equal"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.exp"] = {
    "input": "tensor"
}
signatures["torch.expm1"] = {
    "input": "tensor"
}
signatures["torch.fft.fft"] = {
    "input": "tensor",
    "n": "integer",
    "dim": "tuple", # or list
    "norm": "string"
}
signatures["torch.fft.ifft"] = {
    "input": "tensor",
    "n": "integer",
    "dim": "tuple", # or list
    "norm": "string"
}
signatures["torch.fft.fftn"] = {
    "input": "tensor",
    "s": "tuple",
    "dim": "tuple",
    "norm": "string"
}
signatures["torch.fft.ifftn"] = {
    "input": "tensor",
    "s": "tuple",
    "dim": "tuple",
    "norm": "string"
}
signatures["torch.fft.rfft"] = {
    "input": "tensor",
    "n": "integer",
    "dim": "tuple",
    "norm": "string"
}
signatures["torch.fft.irfft"] = {
    "input": "tensor",
    "n": "integer",
    "dim": "tuple",
    "norm": "string"
}
signatures["torch.fft.rfftn"] = {
    "input": "tensor",
    "s": "tuple",
    "dim": "tuple",
    "norm": "string"
}
signatures["torch.fft.irfftn"] = {
    "input": "tensor",
    "s": "tuple",
    "dim": "tuple",
    "norm": "string"
}
signatures["torch.fft.hfft"] = {
    "input": "tensor",
    "n": "integer",
    "dim": "tuple",
    "norm": "string"
}
signatures["torch.fft.ihfft"] = {
    "input": "tensor",
    "n": "integer",
    "dim": "tuple",
    "norm": "string"
}
signatures["torch.fill"] = {
    "input": "tensor",
    "value": "float" # could be int or tensor too
}
signatures["torch.flatten"] = {
    "input": "tensor",
    "start_dim": "integer",
    "end_dim": "integer"
}
signatures["torch.flip"] = {
    "input": "tensor",
    "dims": "tuple" #or list
}
signatures["torch.floor"] = {
    "input": "tensor"
}
signatures["torch.floor_divide"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.fmod"] = {
    "input": "tensor",
    "divisor": "tensor"
}
signatures["torch.frac"] = {
    "input": "tensor"
}
signatures["torch.gather"] = {
    "input": "tensor",
    "dim": "integer",
    "index": "tensor"
}
signatures["torch.gcd"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.ge"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.greater_equal"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.gt"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.greater"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.hardshrink"] = {
    "input": "tensor",
    "lambd": "float"
}
signatures["torch.heaviside"] = {
    "input": "tensor",
    "values": "tensor"
}
signatures["torch.histogram"] = {
    "input": "tensor",
    "bins": "integer",
    "range": "tuple", # This is a tuple of (min, max)
    "weight": "tensor",
    "density": "boolean"
}
signatures["torch.hypot"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.i0"] = {
    "input": "tensor"
}
signatures["torch.imag"] = {
    "input": "tensor"
}
signatures["torch.index_select"] = {
    "input": "tensor",
    "dim": "integer",
    "index": "tensor"
}
signatures["torch.inverse"] = {
    "input": "tensor"
}
signatures["torch.isclose"] = {
    "input": "tensor",
    "other": "tensor",
    "rtol": "float",
    "atol": "float",
    "equal_nan": "boolean"
}
signatures["torch.isfinite"] = {
    "input": "tensor"
}
signatures["torch.isinf"] = {
    "input": "tensor"
}
signatures["torch.isnan"] = {
    "input": "tensor"
}
signatures["torch.isreal"] = {
    "input": "tensor"
}
signatures["torch.lcm"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.ldexp"] = {
    "input": "tensor",
    "values": "tensor"
}
signatures["torch.le"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.less_equal"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.lerp"] = {
    "input": "tensor",
    "end": "tensor",
    "weight": "float" # can also be a tensor
}
signatures["torch.lgamma"] = {
    "input": "tensor"
}
signatures["torch.log"] = {
    "input": "tensor"
}
signatures["torch.log10"] = {
    "input": "tensor"
}
signatures["torch.log1p"] = {
    "input": "tensor"
}
signatures["torch.log2"] = {
    "input": "tensor"
}
signatures["torch.logaddexp"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.logaddexp2"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.logcumsumexp"] = {
    "input": "tensor",
    "dim": "integer"
}
signatures["torch.logdet"] = {
    "input": "tensor"
}
signatures["torch.logical_and"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.logical_not"] = {
    "input": "tensor"
}
signatures["torch.logical_or"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.logical_xor"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.logsumexp"] = {
    "input": "tensor",
    "dim": "tuple", # or list
    "keepdim": "boolean"
}
signatures["torch.lt"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.less"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.masked_fill"] = {
    "input": "tensor",
    "mask": "tensor",
    "value": "float" # Can also be tensor, int
}
signatures["torch.matmul"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.max"] = {
    "input": "tensor",
    "dim": "integer",
    "keepdim": "boolean"
}
signatures["torch.maximum"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.mean"] = {
    "input": "tensor",
    "dim": "tuple", #or integer, or list, but tuple is more common when multiple dimensions are provided
    "keepdim": "boolean",
    "dtype": "dtype"
}
signatures["torch.median"] = {
    "input": "tensor",
    "dim": "integer",
    "keepdim": "boolean"
}
signatures["torch.min"] = {
    "input": "tensor",
    "dim": "integer",
    "keepdim": "boolean"
}
signatures["torch.minimum"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.mm"] = {
    "input": "tensor",
    "mat2": "tensor"
}
signatures["torch.movedim"] = {
    "input": "tensor",
    "source": "tuple", #or integer, or list
    "destination": "tuple" #or integer, or list
}
signatures["torch.moveaxis"] = {
    "input": "tensor",
    "source": "integer",
    "destination": "integer"
}
signatures["torch.mul"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.multinomial"] = {
    "input": "tensor",
    "num_samples": "integer",
    "replacement": "boolean"
}
signatures["torch.multiply"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.nan_to_num"] = {
    "input": "tensor",
    "nan": "float", #can also be int
    "posinf": "float", #can also be int
    "neginf": "float" #can also be int
}
signatures["torch.narrow"] = {
    "input": "tensor",
    "dim": "integer",
    "start": "integer",
    "length": "integer"
}
signatures["torch.ne"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.negative"] = {
    "input": "tensor"
}
signatures["torch.nextafter"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.nonzero"] = {
    "input": "tensor",
    "as_tuple": "boolean"
}
signatures["torch.normal"] = {
    "mean": "float",
    "std": "float",
    "size": "tuple"
}
signatures["torch.not_equal"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.ones"] = {
    "size": "tuple", # or list of integers, but tuple is more common
    "dtype": "dtype"
}
signatures["torch.ones_like"] = {
    "input": "tensor",
    "dtype": "dtype"
}
signatures["torch.permute"] = {
    "input": "tensor",
    "dims": "tuple" # or list of integers
}
signatures["torch.polygamma"] = {
    "n": "integer",
    "input": "tensor"
}
signatures["torch.positive"] = {
    "input": "tensor"
}
signatures["torch.pow"] = {
    "input": "tensor",
    "exponent": "tensor" # can also be float
}
signatures["torch.prod"] = {
    "input": "tensor",
    "dim": "integer",
    "keepdim": "boolean",
    "dtype": "dtype"
}
signatures["torch.rad2deg"] = {
    "input": "tensor"
}
signatures["torch.rand"] = {
    "size": "tuple", # or list of integers, but tuple is more common
    "dtype": "dtype"
}
signatures["torch.rand_like"] = {
    "input": "tensor",
    "dtype": "dtype"
}
signatures["torch.randint"] = {
    "low": "integer",
    "high": "integer",
    "size": "tuple", # or list of integers, but tuple is more common
    "dtype": "dtype"
}
signatures["torch.randint_like"] = {
    "input": "tensor",
    "low": "integer",
    "high": "integer",
    "dtype": "dtype"
}
signatures["torch.randn"] = {
    "size": "tuple", # or list of integers, but tuple is more common
    "dtype": "dtype"
}
signatures["torch.randn_like"] = {
    "input": "tensor",
    "dtype": "dtype"
}
signatures["torch.ravel"] = {
    "input": "tensor"
}
signatures["torch.real"] = {
    "input": "tensor"
}
signatures["torch.reciprocal"] = {
    "input": "tensor"
}
signatures["torch.remainder"] = {
    "input": "tensor",
    "divisor": "tensor"
}
signatures["torch.renorm"] = {
    "input": "tensor",
    "p": "float",
    "dim": "integer",
    "maxnorm": "float"
}
signatures["torch.reshape"] = {
    "input": "tensor",
    "shape": "tuple" # or list of integers
}
signatures["torch.round"] = {
    "input": "tensor"
}
signatures["torch.rsqrt"] = {
    "input": "tensor"
}
signatures["torch.scatter"] = {
    "input": "tensor",
    "dim": "integer",
    "index": "tensor",
    "src": "tensor" # can be a tensor or a float
}
signatures["torch.scatter_add"] = {
    "input": "tensor",
    "dim": "integer",
    "index": "tensor",
    "src": "tensor"
}
signatures["torch.searchsorted"] = {
    "sorted_sequence": "tensor",
    "values": "tensor",
    "out_int32": "boolean",
    "right": "boolean"
}
signatures["torch.select"] = {
    "input": "tensor",
    "dim": "integer",
    "index": "integer"
}
signatures["torch.sgn"] = {
    "input": "tensor"
}
signatures["torch.sigmoid"] = {
    "input": "tensor"
}
signatures["torch.sign"] = {
    "input": "tensor"
}
signatures["torch.sin"] = {
    "input": "tensor"
}
signatures["torch.sinc"] = {
    "input": "tensor"
}
signatures["torch.sinh"] = {
    "input": "tensor"
}
signatures["torch.sort"] = {
    "input": "tensor",
    "dim": "integer",
    "descending": "boolean"
}
signatures["torch.sqrt"] = {
    "input": "tensor"
}
signatures["torch.square"] = {
    "input": "tensor"
}
signatures["torch.squeeze"] = {
    "input": "tensor",
    "dim": "integer"
}
signatures["torch.stack"] = {
    "tensors": "tensor_list",
    "dim": "integer"
}
signatures["torch.std"] = {
    "input": "tensor",
    "dim": "tuple",  # or integer, or list
    "unbiased": "boolean",
    "keepdim": "boolean"
}
signatures["torch.std_mean"] = {
    "input": "tensor",
    "dim": "tuple", # or integer, or list
    "unbiased": "boolean",
    "keepdim": "boolean"
}
signatures["torch.sub"] = {
    "input": "tensor",
    "other": "tensor",
    "alpha": "float"
}
signatures["torch.sum"] = {
    "input": "tensor",
    "dim": "tuple", # or integer, or list
    "keepdim": "boolean",
    "dtype": "dtype"
}
signatures["torch.swapaxes"] = {
    "input": "tensor",
    "dim0": "integer",
    "dim1": "integer"
}
signatures["torch.swapdims"] = {
    "input": "tensor",
    "dim0": "integer",
    "dim1": "integer"
}
signatures["torch.take"] = {
    "input": "tensor",
    "index": "tensor"
}
signatures["torch.tan"] = {
    "input": "tensor"
}
signatures["torch.tanh"] = {
    "input": "tensor"
}
signatures["torch.tensor_split"] = {
    "input": "tensor",
    "indices_or_sections": "integer",  # Can also be a list or tuple
    "dim": "integer"
}
signatures["torch.tile"] = {
    "input": "tensor",
    "reps": "tuple" # or list
}
signatures["torch.topk"] = {
    "input": "tensor",
    "k": "integer",
    "dim": "integer",
    "largest": "boolean",
    "sorted": "boolean"
}
signatures["torch.trace"] = {
    "input": "tensor"
}
signatures["torch.transpose"] = {
    "input": "tensor",
    "dim0": "integer",
    "dim1": "integer"
}
signatures["torch.trapz"] = {
    "y": "tensor",
    "x": "tensor",
    "dim": "integer"
}
signatures["torch.trunc"] = {
    "input": "tensor"
}
signatures["torch.true_divide"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.unique"] = {
    "input": "tensor",
    "sorted": "boolean",
    "return_inverse": "boolean",
    "return_counts": "boolean",
    "dim": "integer"
}
signatures["torch.unsqueeze"] = {
    "input": "tensor",
    "dim": "integer"
}
signatures["torch.var"] = {
    "input": "tensor",
    "dim": "tuple", #or integer, or list
    "unbiased": "boolean",
    "keepdim": "boolean"
}
signatures["torch.var_mean"] = {
    "input": "tensor",
    "dim": "tuple", #or integer, or list
    "unbiased": "boolean",
    "keepdim": "boolean"
}
signatures["torch.vdot"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.where"] = {
    "condition": "tensor",
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.zeros"] = {
    "size": "tuple", # or list of integers, but tuple is more common
    "dtype": "dtype"
}
signatures["torch.zeros_like"] = {
    "input": "tensor",
    "dtype": "dtype"
}
signatures["torch.cpu"] = {}
signatures["torch.cross"] = {
    "input": "tensor",
    "other": "tensor",
    "dim": "integer"
}
signatures["torch.crow_indices_copy"] = {
    "input": "tensor"
}
signatures["torch.ctc_loss"] = {
    "log_probs": "tensor",
    "targets": "tensor",
    "input_lengths": "tensor",
    "target_lengths": "tensor",
    "blank": "integer",
    "reduction": "string",
    "zero_infinity": "boolean"
}
signatures["torch.ctypes.as_array"] = {
    "obj": "integer", # Should it be "integer"? This is actually a ctypes data object.
    "size": "tuple"
}
signatures["torch.cuda.is_available"] = {}
signatures["torch.cuda.device_count"] = {}
signatures["torch.cuda.get_device_name"] = {
    "device": "integer"
}
signatures["torch.cuda.set_device"] = {
    "device": "integer"
}
signatures["torch.cuda.current_device"] = {}
signatures["torch.cuda.empty_cache"] = {}
signatures["torch.cuda.memory_allocated"] = {
    "device": "integer"
}
signatures["torch.cuda.memory_reserved"] = {
    "device": "integer"
}
signatures["torch.cuda.max_memory_allocated"] = {
    "device": "integer"
}
signatures["torch.cuda.max_memory_reserved"] = {
    "device": "integer"
}
signatures["torch.cuda.reset_peak_memory_stats"] = {
    "device": "integer"
}
signatures["torch.cuda.synchronize"] = {
    "device": "integer"
}
signatures["torch.cuda.ipc_collect"] = {}
signatures["torch.cuda.stream"] = {}
signatures["torch.cuda.Stream"] = {
    "device": "integer", # It might be better if this was a device object instead of an integer
    "priority": "integer",
    "ordering": "string" # unsure about the ordering argument, assuming it is string based on the documentation
}
signatures["torch.cuda.Event"] = {
    "enable_timing": "boolean",
    "interprocess": "boolean"
}
signatures["torch.cuda.is_initialized"] = {}
signatures["torch.cuda.manual_seed"] = {
    "seed": "integer"
}
signatures["torch.cuda.manual_seed_all"] = {
    "seed": "integer"
}
signatures["torch.cuda.seed"] = {}
signatures["torch.cuda.seed_all"] = {}
signatures["torch.cuda.cudnn_enabled"] = {}
signatures["torch.cuda.set_enabled_lgraph"] = {
    "enabled": "boolean"
}
signatures["torch.cudnn_affine_grid_generator"] = {
    "theta": "tensor",
    "N": "integer",
    "C": "integer",
    "H": "integer",
    "W": "integer"
}
signatures["torch.cudnn_batch_norm"] = {
    "input": "tensor",
    "weight": "tensor",
    "bias": "tensor",
    "running_mean": "tensor",
    "running_var": "tensor",
    "eps": "float",
    "exponential_average_factor": "float",
    "training": "boolean"
}
signatures["torch.cudnn_convolution"] = {
    "input": "tensor",
    "weight": "tensor",
    "bias": "tensor",
    "padding": "list", #could be tuple?
    "stride": "list", #could be tuple?
    "dilation": "list", #could be tuple?
    "groups": "integer",
    "benchmark": "boolean",
    "deterministic": "boolean",
    "allow_tf32": "boolean"
}
signatures["torch.cudnn_convolution_add_relu"] = {
    "input": "tensor",
    "weight": "tensor",
    "bias": "tensor",
    "padding": "list", # Could be tuple or int as well, choosing list as it's more general
    "stride": "list", # Could be tuple or int as well, choosing list as it's more general
    "dilation": "list", # Could be tuple or int as well, choosing list as it's more general
    "groups": "integer"
}
signatures["torch.cudnn_convolution_relu"] = {
    "input": "tensor",
    "weight": "tensor",
    "bias": "tensor",
    "padding": "list", # Could also be a tuple or integer, but list seems more common
    "stride": "list", # Could also be a tuple or integer, but list seems more common
    "dilation": "list", # Could also be a tuple or integer, but list seems more common
    "groups": "integer"
}
signatures["torch.cudnn_convolution_transpose"] = {
    "input": "tensor",
    "weight": "tensor",
    "bias": "tensor", # could also be None
    "padding": "tuple",
    "output_padding": "tuple",
    "stride": "tuple",
    "dilation": "tuple",
    "groups": "integer"
}
signatures["torch.cudnn_grid_sampler"] = {
    "input": "tensor",
    "grid": "tensor"
}
signatures["torch.cudnn_is_acceptable"] = {
    "input": "tensor"
}
signatures["torch.cummin"] = {
    "input": "tensor",
    "dim": "integer"
}
signatures["torch.cumulative_trapezoid"] = {
    "y": "tensor",
    "x": "tensor",
    "dim": "integer"
}
signatures["torch.dequantize"] = {
    "tensor": "tensor"
}
signatures["torch.detach"] = {
    "input": "tensor"
}
signatures["torch.detach_"] = {
    "self": "tensor"
}
signatures["torch.detach_copy"] = {
    "input": "tensor"
}
signatures["torch.device"] = {
    "type": "string",
    "index": "integer" # Could be None, but integer is more common
}
signatures["torch.diag"] = {
    "input": "tensor",
    "diagonal": "integer"
}
signatures["torch.diag_embed"] = {
    "input": "tensor",
    "offset": "integer",
    "dim1": "integer",
    "dim2": "integer"
}
signatures["torch.diagflat"] = {
    "input": "tensor",
    "offset": "integer"
}
signatures["torch.diagonal"] = {
    "input": "tensor",
    "offset": "integer",
    "dim1": "integer",
    "dim2": "integer"
}
signatures["torch.diagonal_copy"] = {
    "input": "tensor",
    "diagonal": "tensor",
    "offset": "integer",
    "dim1": "integer",
    "dim2": "integer"
}
signatures["torch.diagonal_scatter"] = {
    "input": "tensor",
    "src": "tensor",
    "offset": "integer",
    "dim1": "integer",
    "dim2": "integer"
}
signatures["torch.diff"] = {
    "input": "tensor",
    "n": "integer",
    "dim": "integer",
    "prepend": "tensor",
    "append": "tensor"
}
signatures["torch.digamma"] = {
    "input": "tensor"
}
signatures["torch.distributed.init_process_group"] = {
    "backend": "string",
    "init_method": "string",
    "timeout": "float",
    "world_size": "integer",
    "rank": "integer",
    "group_name": "string"
}
signatures["torch.distributed.get_rank"] = {}
signatures["torch.distributed.get_world_size"] = {}
signatures["torch.distributed.is_initialized"] = {}
signatures["torch.distributed.destroy_process_group"] = {
    "group": "string" # Could be a ProcessGroup object. Assuming it's named via string
}
signatures["torch.distributed.all_gather"] = {
    "tensor_list": "tensor_list",
    "tensor": "tensor",
    "group": "string" # Could be a ProcessGroup object. Assuming it's named via string
}
signatures["torch.distributed.all_gather_into_tensor"] = {
    "output_tensor": "tensor",
    "input_tensor": "tensor",
    "group": "string" # Could be a ProcessGroup object. Assuming it's named via string
}
signatures["torch.distributed.all_reduce"] = {
    "tensor": "tensor",
    "op": "string",
    "group": "string", # Could be a ProcessGroup object. Assuming it's named via string
    "async_op": "boolean"
}
signatures["torch.distributed.broadcast"] = {
    "tensor": "tensor",
    "src": "integer",
    "group": "string", # Could be a ProcessGroup object. Assuming it's named via string
    "async_op": "boolean"
}
signatures["torch.distributed.reduce"] = {
    "tensor": "tensor",
    "dst": "integer",
    "op": "string",
    "group": "string", # Could be a ProcessGroup object. Assuming it's named via string
    "async_op": "boolean"
}
signatures["torch.distributed.scatter"] = {
    "tensor": "tensor",
    "scatter_list": "tensor_list",
    "src": "integer",
    "group": "string", # Could be a ProcessGroup object. Assuming it's named via string
    "async_op": "boolean"
}
signatures["torch.distributed.gather"] = {
    "tensor": "tensor",
    "gather_list": "tensor_list",
    "dst": "integer",
    "group": "string", # Could be a ProcessGroup object. Assuming it's named via string
    "async_op": "boolean"
}
signatures["torch.distributed.barrier"] = {
    "group": "string", # Could be a ProcessGroup object. Assuming it's named via string
    "async_op": "boolean",
    "timeout": "float"
}
signatures["torch.distributed.send"] = {
    "tensor": "tensor",
    "dst": "integer",
    "tag": "integer"
}
signatures["torch.distributed.recv"] = {
    "tensor": "tensor",
    "src": "integer",
    "tag": "integer"
}
signatures["torch.distributed.isend"] = {
    "tensor": "tensor",
    "dst": "integer",
    "tag": "integer"
}
signatures["torch.distributed.irecv"] = {
    "tensor": "tensor",
    "src": "integer",
    "tag": "integer"
}
signatures["torch.distributed.new_group"] = {
    "ranks": "list" # Expect a list of integers
}
signatures["torch.distributed.get_backend"] = {
    "group": "string" # Could be a ProcessGroup object. Assuming it's named via string
}
signatures["torch.distributed.send_recv"] = {
    "send_tensor": "tensor",
    "recv_tensor": "tensor",
    "dst": "integer",
    "send_tag": "integer",
    "recv_tag": "integer"
}
signatures["torch.distributed.is_available"] = {}
signatures["torch.distributions.bernoulli.Bernoulli"] = {
    "probs": "tensor",
    "logits": "tensor",
    "validate_args": "boolean"
}
signatures["torch.distributions.categorical.Categorical"] = {
    "probs": "tensor",
    "logits": "tensor",
    "validate_args": "boolean"
}
signatures["torch.distributions.normal.Normal"] = {
    "loc": "tensor",
    "scale": "tensor",
    "validate_args": "boolean"
}
signatures["torch.distributions.uniform.Uniform"] = {
    "low": "tensor",
    "high": "tensor",
    "validate_args": "boolean"
}
signatures["torch.distributions.beta.Beta"] = {
    "concentration1": "tensor",
    "concentration0": "tensor",
    "validate_args": "boolean"
}
signatures["torch.distributions.gamma.Gamma"] = {
    "concentration": "tensor",
    "rate": "tensor",
    "validate_args": "boolean"
}
signatures["torch.distributions.exponential.Exponential"] = {
    "rate": "tensor",
    "validate_args": "boolean"
}
signatures["torch.distributions.laplace.Laplace"] = {
    "loc": "tensor",
    "scale": "tensor",
    "validate_args": "boolean"
}
signatures["torch.distributions.dirichlet.Dirichlet"] = {
    "concentration": "tensor",
    "validate_args": "boolean"
}
signatures["torch.distributions.multivariate_normal.MultivariateNormal"] = {
    "loc": "tensor",
    "covariance_matrix": "tensor",
    "precision_matrix": "tensor",
    "scale_tril": "tensor",
    "validate_args": "boolean"
}
signatures["torch.distributions.studentT.StudentT"] = {
    "df": "tensor",
    "loc": "tensor",
    "scale": "tensor",
    "validate_args": "boolean"
}
signatures["torch.distributions.poisson.Poisson"] = {
    "rate": "tensor",
    "validate_args": "boolean"
}
signatures["torch.distributions.binomial.Binomial"] = {
    "total_count": "tensor",
    "probs": "tensor",
    "logits": "tensor",
    "validate_args": "boolean"
}
signatures["torch.divide"] = {
    "input": "tensor",
    "other": "tensor",
    "rounding_mode": "string" # Could also be None, but "string" seems more appropriate for the possible values
}
signatures["torch.dropout"] = {
    "input": "tensor",
    "p": "float",
    "train": "boolean"
}
signatures["torch.dropout_"] = {
    "input": "tensor",
    "p": "float",
    "train": "boolean"
}
signatures["torch.dsmm"] = {
    "mat1": "tensor",
    "mat2": "tensor"
}
signatures["torch.dsplit"] = {
    "input": "tensor",
    "indices_or_sections": "list" # Could be an integer as well, but list/tuple is more general as hinted in docstring
}
signatures["torch.dtype"] = {
}
signatures["torch.embedding"] = {
    "weight": "tensor",
    "input": "tensor",
    "padding_idx": "integer",
    "max_norm": "float",
    "norm_type": "float",
    "scale_grad_by_freq": "boolean",
    "sparse": "boolean"
}
signatures["torch.embedding_bag"] = {
    "weight": "tensor",
    "input": "tensor",
    "offsets": "tensor",
    "max_norm": "float",
    "norm_type": "float",
    "scale_grad_by_freq": "boolean",
    "mode": "string",
    "sparse": "boolean",
    "per_sample_weights": "tensor",
    "include_last_offset": "boolean",
    "_weight": "tensor" # I'm not entirely sure about this one, but it appears to be a tensor related to the weights.
}
signatures["torch.embedding_renorm_"] = {
    "weight": "tensor",
    "max_norm": "float",
    "norm_type": "float" # Could also be integer, but float is more common for representing norm types
}
signatures["torch.empty"] = {
    "size": "tuple", # Could also be a variable number of integers, but tuple seems more common
    "dtype": "dtype",
    "layout": "layout", # Not a standard type, but specified in documentation
    "requires_grad": "boolean",
    "pin_memory": "boolean",
    "memory_format": "memory_format" # Not a standard type, but specified in documentation
}
signatures["torch.empty_like"] = {
    "input": "tensor",
    "dtype": "dtype",
    "layout": "string", # Should maybe be torch.layout
    "requires_grad": "boolean",
    "memory_format": "string" # Should maybe be torch.memory_format
}
signatures["torch.empty_permuted"] = {
    "size": "tuple",
    "permutation": "list", # Could also be tuple, but list is more common
    "dtype": "dtype",
    "layout": "string", # there is a torch.layout, but it is specified as a string in documentation
    "pin_memory": "boolean",
    "requires_grad": "boolean"
}
signatures["torch.empty_quantized"] = {
    "size": "list", # could also be tuple
    "qtensor": "tensor",
    "dtype": "dtype",
    "layout": "integer",
    "pin_memory": "boolean",
    "requires_grad": "boolean"
}
signatures["torch.empty_strided"] = {
    "size": "tuple",
    "stride": "tuple",
    "dtype": "dtype",
    "layout": "string", # Could also be an enum but string is more general
    "requires_grad": "boolean",
    "pin_memory": "boolean"
}
signatures["torch.enable_grad"] = {
    "orig_func": "None" # Could also be a function, but None is the default
}
signatures["torch.equal"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.erf_"] = {
    "input": "tensor"
}
signatures["torch.erfc"] = {
    "input": "tensor"
}
signatures["torch.erfc_"] = {
    "input": "tensor"
}
signatures["torch.erfinv"] = {
    "input": "tensor"
}
signatures["torch.exp"] = {
    "input": "tensor"
}
signatures["torch.exp_"] = {
    "input": "tensor"
}
signatures["torch.expand_copy"] = {
    "input": "tensor",
    "size": "tuple" # Could also be list, but tuple is more common
}
signatures["torch.expm1"] = {
    "input": "tensor"
}
signatures["torch.export"] = {
    "model": "tensor", # can also be nn.Module, but tensor is more general
    "args": "tuple",
    "kwargs": "dictionary", # Assuming dictionary exists as a valid option - should ideally be a more specific dictionary type.
    "constraints": "list" # List of constraints
}
signatures["torch.eye"] = {
    "n": "integer",
    "m": "integer", # Could also be None, but integer is the more common use case
    "dtype": "dtype",
    "layout": "string", # Or torch.layout, but string is a closer match based on the options
    "requires_grad": "boolean"
}
signatures["torch.fake_quantize_per_channel_affine"] = {
    "input": "tensor",
    "scale": "tensor",
    "zero_point": "tensor",
    "quant_min": "integer",
    "quant_max": "integer"
}
signatures["torch.fake_quantize_per_tensor_affine"] = {
    "input": "tensor",
    "scale": "float",
    "zero_point": "integer",
    "quant_min": "integer",
    "quant_max": "integer"
}
signatures["torch.fbgemm_linear_quantize_weight"] = {
    "weight": "tensor"
}
signatures["torch.fbgemm_pack_quantized_matrix"] = {
    "input": "tensor",
    "q_scale": "float",
    "q_zero_point": "integer" # Could be a tensor, but usually an integer
}
signatures["torch.feature_alpha_dropout"] = {
    "input": "tensor",
    "p": "float",
    "training": "boolean",
    "inplace": "boolean" # Could potentially be boolean. But it defaults to false.
}
signatures["torch.feature_alpha_dropout_"] = {
    "input": "tensor",
    "p": "float",
    "training": "boolean"
}
signatures["torch.feature_dropout"] = {
    "input": "tensor",
    "p": "float",
    "train": "boolean" # Could be None, but boolean is more common
}
signatures["torch.feature_dropout_"] = {
    "input": "tensor",
    "p": "float",
    "train": "boolean" # Might also accept None
}
signatures["torch.fft.fft"] = {
    "input": "tensor",
    "n": "integer",
    "dim": "integer",
    "norm": "string"
}
signatures["torch.fft.ifft"] = {
    "input": "tensor",
    "n": "integer",
    "dim": "integer",
    "norm": "string"
}
signatures["torch.fft.fft2"] = {
    "input": "tensor",
    "s": "tuple", # Could be integer or tuple of integers
    "dim": "tuple",
    "norm": "string"
}
signatures["torch.fft.ifft2"] = {
    "input": "tensor",
    "s": "tuple", # Could be integer or tuple of integers
    "dim": "tuple",
    "norm": "string"
}
signatures["torch.fft.fftn"] = {
    "input": "tensor",
    "s": "tuple", # Could be integer or tuple of integers
    "dim": "tuple",
    "norm": "string"
}
signatures["torch.fft.ifftn"] = {
    "input": "tensor",
    "s": "tuple", # Could be integer or tuple of integers
    "dim": "tuple",
    "norm": "string"
}
signatures["torch.fft.rfft"] = {
    "input": "tensor",
    "n": "integer",
    "dim": "integer",
    "norm": "string"
}
signatures["torch.fft.irfft"] = {
    "input": "tensor",
    "n": "integer",
    "dim": "integer",
    "norm": "string"
}
signatures["torch.fft.rfft2"] = {
    "input": "tensor",
    "s": "tuple", # Could be integer or tuple of integers
    "dim": "tuple",
    "norm": "string"
}
signatures["torch.fft.irfft2"] = {
    "input": "tensor",
    "s": "tuple", # Could be integer or tuple of integers
    "dim": "tuple",
    "norm": "string"
}
signatures["torch.fft.rfftn"] = {
    "input": "tensor",
    "s": "tuple", # Could be integer or tuple of integers
    "dim": "tuple",
    "norm": "string"
}
signatures["torch.fft.irfftn"] = {
    "input": "tensor",
    "s": "tuple", # Could be integer or tuple of integers
    "dim": "tuple",
    "norm": "string"
}
signatures["torch.fft.hfft"] = {
    "input": "tensor",
    "n": "integer",
    "dim": "integer",
    "norm": "string"
}
signatures["torch.fft.ihfft"] = {
    "input": "tensor",
    "n": "integer",
    "dim": "integer",
    "norm": "string"
}
signatures["torch.fft.fftfreq"] = {
    "n": "integer",
    "d": "float", # spacing
    "device": "string",
    "dtype": "dtype"
}
signatures["torch.fft.rfftfreq"] = {
    "n": "integer",
    "d": "float",
    "device": "string",
    "dtype": "dtype"
}
signatures["torch.fft.fftshift"] = {
    "input": "tensor",
    "dim": "tuple" # can be integer or tuple of integers, choosing most commonly used
}
signatures["torch.fft.ifftshift"] = {
    "input": "tensor",
    "dim": "tuple" # can be integer or tuple of integers, choosing most commonly used
}
signatures["torch.fft.fft"] = {
    "input": "tensor",
    "n": "integer",
    "dim": "integer",
    "norm": "string"
}
signatures["torch.fft.ifft"] = {
    "input": "tensor",
    "n": "integer",
    "dim": "integer",
    "norm": "string"
}
signatures["torch.fft.fft2"] = {
    "input": "tensor",
    "s": "tuple",
    "dim": "tuple",
    "norm": "string"
}
signatures["torch.fft.ifft2"] = {
    "input": "tensor",
    "s": "tuple",
    "dim": "tuple",
    "norm": "string"
}
signatures["torch.fft.fftn"] = {
    "input": "tensor",
    "s": "tuple",
    "dim": "tuple",
    "norm": "string"
}
signatures["torch.fft.ifftn"] = {
    "input": "tensor",
    "s": "tuple",
    "dim": "tuple",
    "norm": "string"
}
signatures["torch.fft.rfft"] = {
    "input": "tensor",
    "n": "integer",
    "dim": "integer",
    "norm": "string"
}
signatures["torch.fft.irfft"] = {
    "input": "tensor",
    "n": "integer",
    "dim": "integer",
    "norm": "string"
}
signatures["torch.fft.rfft2"] = {
    "input": "tensor",
    "s": "tuple",
    "dim": "tuple",
    "norm": "string"
}
signatures["torch.fft.irfft2"] = {
    "input": "tensor",
    "s": "tuple",
    "dim": "tuple",
    "norm": "string"
}
signatures["torch.fft.rfftn"] = {
    "input": "tensor",
    "s": "tuple",
    "dim": "tuple",
    "norm": "string"
}
signatures["torch.fft.irfftn"] = {
    "input": "tensor",
    "s": "tuple",
    "dim": "tuple",
    "norm": "string"
}
signatures["torch.fft.hfft"] = {
    "input": "tensor",
    "n": "integer",
    "dim": "integer",
    "norm": "string"
}
signatures["torch.fft.ihfft"] = {
    "input": "tensor",
    "n": "integer",
    "dim": "integer",
    "norm": "string"
}
signatures["torch.fft.fftshift"] = {
    "input": "tensor",
    "dim": "tuple" # Could also be an integer, but tuple is more common
}
signatures["torch.fft.ifftshift"] = {
    "input": "tensor",
    "dim": "tuple" # Could also be an integer, but tuple is more common
}
signatures["torch.fft.fftfreq"] = {
    "n": "integer",
    "d": "float"
}
signatures["torch.fft.rfftfreq"] = {
    "n": "integer",
    "d": "float"
}
signatures["torch.fft.fft"] = {
    "input": "tensor",
    "n": "integer",
    "dim": "integer",
    "norm": "string"
}
signatures["torch.fft.fft2"] = {
    "input": "tensor",
    "s": "tuple",
    "dim": "tuple",
    "norm": "string"
}
signatures["torch.fft.fftfreq"] = {
    "n": "integer",
    "d": "float",
    "dtype": "dtype",
    "layout": "string", # could also be a torch.layout object
    "requires_grad": "boolean"
}
signatures["torch.fft.fftn"] = {
    "input": "tensor",
    "s": "tuple",
    "dim": "tuple",
    "norm": "string"
}
signatures["torch.fft.fftshift"] = {
    "input": "tensor",
    "dim": "tuple" # Could also be integer, but tuple is more general
}
signatures["torch.fft.hfft"] = {
    "input": "tensor",
    "n": "integer",
    "dim": "integer",
    "norm": "string"
}
signatures["torch.fft.hfft2"] = {
    "input": "tensor",
    "s": "tuple",
    "dim": "tuple",
    "norm": "string"
}
signatures["torch.fft.hfftn"] = {
    "input": "tensor",
    "s": "tuple",
    "dim": "tuple",
    "norm": "string"
}
signatures["torch.fft.ifft"] = {
    "input": "tensor",
    "n": "integer",
    "dim": "integer",
    "norm": "string"
}
signatures["torch.fft.ifft2"] = {
    "input": "tensor",
    "s": "tuple",
    "dim": "tuple",
    "norm": "string"
}
signatures["torch.fft.ifftn"] = {
    "input": "tensor",
    "s": "tuple", # Could also be None, but tuple seems more common when specified
    "dim": "tuple", # Could also be None, but tuple seems more common when specified
    "norm": "string"
}
signatures["torch.fft.ifftshift"] = {
    "input": "tensor",
    "dim": "tuple" # Could also be an integer, but tuple is more general and covers both cases
}
signatures["torch.fft.ihfft"] = {
    "input": "tensor",
    "n": "integer",
    "dim": "integer",
    "norm": "string"
}
signatures["torch.fft.ihfft2"] = {
    "input": "tensor",
    "s": "tuple",
    "dim": "tuple",
    "norm": "string"
}
signatures["torch.fft.ihfftn"] = {
    "input": "tensor",
    "s": "tuple",
    "dim": "tuple",
    "norm": "string"
}
signatures["torch.fft.irfft"] = {
    "input": "tensor",
    "n": "integer",
    "dim": "integer",
    "norm": "string"
}
signatures["torch.fft.irfft2"] = {
    "input": "tensor",
    "s": "tuple",
    "dim": "tuple",
    "norm": "string"
}
signatures["torch.fft.irfftn"] = {
    "input": "tensor",
    "s": "tuple",
    "dim": "tuple",
    "norm": "string"
}
signatures["torch.fft.rfft"] = {
    "input": "tensor",
    "n": "integer",
    "dim": "integer",
    "norm": "string"
}
signatures["torch.fft.rfft2"] = {
    "input": "tensor",
    "s": "tuple", # Could also be a list of integers, but tuple is more common for sizes
    "dim": "tuple",
    "norm": "string"
}
signatures["torch.fft.rfftfreq"] = {
    "n": "integer",
    "d": "float",
    "dtype": "dtype",
    "layout": "torch.layout", # This should probably be string but there isn't an option that matches exactly
    "requires_grad": "boolean"
}
signatures["torch.fft.rfftn"] = {
    "input": "tensor",
    "s": "tuple",
    "dim": "tuple",
    "norm": "string"
}
signatures["torch.fft.fftfreq"] = {
    "n": "integer",
    "d": "float"
}
signatures["torch.fft.rfftfreq"] = {
    "n": "integer",
    "d": "float"
}
signatures["torch.fft.fft"] = {
    "input": "tensor",
    "n": "integer",
    "dim": "tuple",
    "norm": "string"
}
signatures["torch.fft.ifft"] = {
    "input": "tensor",
    "n": "integer",
    "dim": "tuple",
    "norm": "string"
}
signatures["torch.fft.rfft"] = {
    "input": "tensor",
    "n": "integer",
    "dim": "tuple",
    "norm": "string"
}
signatures["torch.fft.irfft"] = {
    "input": "tensor",
    "n": "integer",
    "dim": "tuple",
    "norm": "string"
}
signatures["torch.fft.hfft"] = {
    "input": "tensor",
    "n": "integer",
    "dim": "tuple",
    "norm": "string"
}
signatures["torch.fft.ihfft"] = {
    "input": "tensor",
    "n": "integer",
    "dim": "tuple",
    "norm": "string"
}
signatures["torch.fft.fft2"] = {
    "input": "tensor",
    "s": "tuple",
    "dim": "tuple",
    "norm": "string"
}
signatures["torch.fft.ifft2"] = {
    "input": "tensor",
    "s": "tuple",
    "dim": "tuple",
    "norm": "string"
}
signatures["torch.fft.rfft2"] = {
    "input": "tensor",
    "s": "tuple",
    "dim": "tuple",
    "norm": "string"
}
signatures["torch.fft.irfft2"] = {
    "input": "tensor",
    "s": "tuple",
    "dim": "tuple",
    "norm": "string"
}
signatures["torch.fft.fftn"] = {
    "input": "tensor",
    "s": "tuple",
    "dim": "tuple",
    "norm": "string"
}
signatures["torch.fft.ifftn"] = {
    "input": "tensor",
    "s": "tuple",
    "dim": "tuple",
    "norm": "string"
}
signatures["torch.fft.rfftn"] = {
    "input": "tensor",
    "s": "tuple",
    "dim": "tuple",
    "norm": "string"
}
signatures["torch.fft.irfftn"] = {
    "input": "tensor",
    "s": "tuple",
    "dim": "tuple",
    "norm": "string"
}
signatures["torch.fft.fftshift"] = {
    "input": "tensor",
    "dim": "tuple"
}
signatures["torch.fft.ifftshift"] = {
    "input": "tensor",
    "dim": "tuple"
}
signatures["torch.fft.fft"] = {
    "input": "tensor",
    "n": "integer", # Can also be None
    "dim": "integer",
    "norm": "string"
}
signatures["torch.fft.ifft"] = {
    "input": "tensor",
    "n": "integer", # Can also be None
    "dim": "integer",
    "norm": "string"
}
signatures["torch.fft.rfft"] = {
    "input": "tensor",
    "n": "integer", # Can also be None
    "dim": "integer",
    "norm": "string"
}
signatures["torch.fft.irfft"] = {
    "input": "tensor",
    "n": "integer", # Can also be None
    "dim": "integer",
    "norm": "string"
}
signatures["torch.fft.hfft"] = {
    "input": "tensor",
    "n": "integer", # Can also be None
    "dim": "integer",
    "norm": "string"
}
signatures["torch.fft.ihfft"] = {
    "input": "tensor",
    "n": "integer", # Can also be None
    "dim": "integer",
    "norm": "string"
}
signatures["torch.fft.fftn"] = {
    "input": "tensor",
    "s": "tuple", # Can also be None
    "dim": "tuple", # Can also be a list
    "norm": "string"
}
signatures["torch.fft.ifftn"] = {
    "input": "tensor",
    "s": "tuple", # Can also be None
    "dim": "tuple", # Can also be a list
    "norm": "string"
}
signatures["torch.fft.rfftn"] = {
    "input": "tensor",
    "s": "tuple", # Can also be None
    "dim": "tuple", # Can also be a list
    "norm": "string"
}
signatures["torch.fft.irfftn"] = {
    "input": "tensor",
    "s": "tuple", # Can also be None
    "dim": "tuple", # Can also be a list
    "norm": "string"
}
signatures["torch.fft.fft2"] = {
    "input": "tensor",
    "s": "tuple", # Can also be None
    "dim": "tuple",
    "norm": "string"
}
signatures["torch.fft.ifft2"] = {
    "input": "tensor",
    "s": "tuple", # Can also be None
    "dim": "tuple",
    "norm": "string"
}
signatures["torch.fft.rfft2"] = {
    "input": "tensor",
    "s": "tuple", # Can also be None
    "dim": "tuple",
    "norm": "string"
}
signatures["torch.fft.irfft2"] = {
    "input": "tensor",
    "s": "tuple", # Can also be None
    "dim": "tuple",
    "norm": "string"
}
signatures["torch.fft.fftfreq"] = {
    "n": "integer",
    "d": "float",
    "device": "string" #actually a device but skipping
}
signatures["torch.fft.rfftfreq"] = {
    "n": "integer",
    "d": "float",
    "device": "string" #actually a device but skipping
}
signatures["torch.fft.fftshift"] = {
    "input": "tensor",
    "dim": "tuple" # Can also be None or integer
}
signatures["torch.fft.ifftshift"] = {
    "input": "tensor",
    "dim": "tuple" # Can also be None or integer
}
signatures["torch.fft.fftangle"] = {
    "input": "tensor"
}
signatures["torch.fft.fftpack.fft"] = {
    "input": "tensor",
    "n": "integer",
    "axis": "integer",
    "norm": "string"
}
signatures["torch.fft.fftpack.ifft"] = {
    "input": "tensor",
    "n": "integer",
    "axis": "integer",
    "norm": "string"
}
signatures["torch.fft.fftpack.fft2"] = {
    "input": "tensor",
    "s": "tuple",
    "axes": "tuple",
    "norm": "string"
}
signatures["torch.fft.fftpack.ifft2"] = {
    "input": "tensor",
    "s": "tuple",
    "axes": "tuple",
    "norm": "string"
}
signatures["torch.fft.fftpack.fftn"] = {
    "input": "tensor",
    "s": "tuple",
    "axes": "tuple",
    "norm": "string"
}
signatures["torch.fft.fftpack.ifftn"] = {
    "input": "tensor",
    "s": "tuple",
    "axes": "tuple",
    "norm": "string"
}
signatures["torch.fft.fftpack.rfft"] = {
    "input": "tensor",
    "n": "integer",
    "axis": "integer",
    "norm": "string"
}
signatures["torch.fft.fftpack.irfft"] = {
    "input": "tensor",
    "n": "integer",
    "axis": "integer",
    "norm": "string"
}
signatures["torch.fft.fftpack.rfft2"] = {
    "input": "tensor",
    "s": "tuple",
    "axes": "tuple",
    "norm": "string"
}
signatures["torch.fft.fftpack.irfft2"] = {
    "input": "tensor",
    "s": "tuple",
    "axes": "tuple",
    "norm": "string"
}
signatures["torch.fft.fftpack.rfftn"] = {
    "input": "tensor",
    "s": "tuple",
    "axes": "tuple",
    "norm": "string"
}
signatures["torch.fft.fftpack.irfftn"] = {
    "input": "tensor",
    "s": "tuple",
    "axes": "tuple",
    "norm": "string"
}
signatures["torch.fill"] = {
    "input": "tensor",
    "value": "float" # Can also be integer or boolean, but float is most common for general values
}
signatures["torch.fill_"] = {
    "input": "tensor",
    "value": "float" # Could potentially be "integer" as well, but "float" allows for more general use.
}
signatures["torch.finfo"] = {
    "dtype": "dtype"
}
signatures["torch.fix"] = {
    "input": "tensor"
}
signatures["torch.fix_"] = {
    "input": "tensor"
}
signatures["torch.flatten"] = {
    "input": "tensor",
    "start_dim": "integer",
    "end_dim": "integer"
}
signatures["torch.floor_"] = {
    "input": "tensor"
}
signatures["torch.floor_divide"] = {
    "input": "tensor",
    "other": "tensor" # other could be a float or an integer, but tensor covers both cases the best
}
signatures["torch.fmax"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.fmod"] = {
    "input": "tensor",
    "other": "tensor" # Could be scalar, but tensor is more general
}
signatures["torch.fork"] = {
    "fn": "string",  # Or perhaps a function pointer type if there was one
    "args": "tuple",
    "kwargs": "tuple" # Should probably be dictionary but sticking to guideline of most common type
}
signatures["torch.frac_"] = {
    "input": "tensor"
}
signatures["torch.frexp"] = {
    "input": "tensor"
}
signatures["torch.frobenius_norm"] = {
    "input": "tensor",
    "dim": "list", # Could also be integer, but list is more common when reducing multiple dimensions
    "keepdim": "boolean"
}
signatures["torch.from_dlpack"] = {
    "dlpack": "tensor"
}
signatures["torch.from_file"] = {
    "filename": "string",
    "size": "integer",
    "dtype": "dtype",
    "offset": "integer"
}
signatures["torch.from_numpy"] = {
    "a": "list" # Should be "tensor", since it takes a numpy array
}
signatures["torch.frombuffer"] = {
    "buffer": "string", # Could also be object, but string is the closest available type
    "dtype": "dtype",
    "count": "integer",
    "offset": "integer",
    "requires_grad": "boolean"
}
signatures["torch.full"] = {
    "size": "tuple", # Could also be a list or torch.Size
    "fill_value": "float", # Could also be an integer, but float is more general
    "dtype": "dtype",
    "layout": "string", #torch.layout
    "requires_grad": "boolean"
}
signatures["torch.func.grad"] = {
    "func": "callable", # Should ideally be "function" or "callable" type
    "argnums": "integer",
    "has_aux": "boolean",
    "holomorphic": "boolean",
    "allow_int": "boolean"
}
signatures["torch.func.vmap"] = {
    "func": "callable", # Should ideally be "function" or "callable" type
    "in_dims": "integer", # can be integer, tuple or list
    "out_dims": "integer",
    "randomness": "string",
    "chunk_size": "integer"
}
signatures["torch.func.jacrev"] = {
    "func": "callable", # Should ideally be "function" or "callable" type
    "argnums": "integer",
    "holomorphic": "boolean",
    "allow_int": "boolean"
}
signatures["torch.func.jacfwd"] = {
    "func": "callable", # Should ideally be "function" or "callable" type
    "argnums": "integer",
    "holomorphic": "boolean",
    "allow_int": "boolean"
}
signatures["torch.func.hessian"] = {
    "func": "callable", # Should ideally be "function" or "callable" type
    "argnums": "integer",
    "holomorphic": "boolean",
    "allow_int": "boolean"
}
signatures["torch.func.jvp"] = {
    "func": "callable", # Should ideally be "function" or "callable" type
    "primals": "tensor", # can also be a tuple of tensors
    "tangents": "tensor" # can also be a tuple of tensors
}
signatures["torch.func.vjp"] = {
    "func": "callable", # Should ideally be "function" or "callable" type
    "primals": "tensor", # can also be a tuple of tensors
    "cotangents": "tensor" # can also be a tuple of tensors
}
signatures["torch.func.hvp"] = {
    "func": "callable", # Should ideally be "function" or "callable" type
    "primals": "tensor",
    "tangents": "tensor"
}
signatures["torch.func.transforms.filter"] = {
    "in_tree": "list", # could be any tree-like structure but using list
    "leaves": "list"
}
signatures["torch.func.transforms.tree_map"] = {
    "func": "callable", # Should ideally be "function" or "callable" type
    "tree": "list", # could be any tree-like structure but using list
}
signatures["torch.func.functionalize"] = {
    "module": "module", # there is no module type to select.
    "disable": "boolean"
}
signatures["torch.functional.conv1d"] = {
    "input": "tensor",
    "weight": "tensor",
    "bias": "tensor",
    "stride": "integer", # Could be tuple as well
    "padding": "integer", # Could be tuple as well
    "dilation": "integer",
    "groups": "integer"
}
signatures["torch.functional.conv2d"] = {
    "input": "tensor",
    "weight": "tensor",
    "bias": "tensor",
    "stride": "integer", # Could be tuple as well
    "padding": "integer", # Could be tuple as well
    "dilation": "integer", # Could be tuple as well
    "groups": "integer"
}
signatures["torch.functional.conv3d"] = {
    "input": "tensor",
    "weight": "tensor",
    "bias": "tensor",
    "stride": "integer", # Could be tuple as well
    "padding": "integer", # Could be tuple as well
    "dilation": "integer", # Could be tuple as well
    "groups": "integer"
}
signatures["torch.functional.relu"] = {
    "input": "tensor",
    "inplace": "boolean"
}
signatures["torch.functional.leaky_relu"] = {
    "input": "tensor",
    "negative_slope": "float",
    "inplace": "boolean"
}
signatures["torch.functional.max_pool1d"] = {
    "input": "tensor",
    "kernel_size": "integer", # can also be a tuple
    "stride": "integer", # can also be a tuple
    "padding": "integer", # can also be a tuple
    "dilation": "integer", # can also be a tuple
    "ceil_mode": "boolean"
}
signatures["torch.functional.max_pool2d"] = {
    "input": "tensor",
    "kernel_size": "integer", # can also be a tuple
    "stride": "integer", # can also be a tuple
    "padding": "integer", # can also be a tuple
    "dilation": "integer", # can also be a tuple
    "ceil_mode": "boolean"
}
signatures["torch.functional.max_pool3d"] = {
    "input": "tensor",
    "kernel_size": "integer", # can also be a tuple
    "stride": "integer", # can also be a tuple
    "padding": "integer", # can also be a tuple
    "dilation": "integer", # can also be a tuple
    "ceil_mode": "boolean"
}
signatures["torch.functional.adaptive_avg_pool1d"] = {
    "input": "tensor",
    "output_size": "integer" # can also be a tuple
}
signatures["torch.functional.adaptive_avg_pool2d"] = {
    "input": "tensor",
    "output_size": "integer" # can also be a tuple
}
signatures["torch.functional.adaptive_avg_pool3d"] = {
    "input": "tensor",
    "output_size": "integer" # can also be a tuple
}
signatures["torch.functional.avg_pool1d"] = {
    "input": "tensor",
    "kernel_size": "integer", # can also be a tuple
    "stride": "integer", # can also be a tuple
    "padding": "integer", # can also be a tuple
    "ceil_mode": "boolean",
    "count_include_pad": "boolean"
}
signatures["torch.functional.avg_pool2d"] = {
    "input": "tensor",
    "kernel_size": "integer", # can also be a tuple
    "stride": "integer", # can also be a tuple
    "padding": "integer", # can also be a tuple
    "ceil_mode": "boolean",
    "count_include_pad": "boolean",
    "divisor_override": "integer"
}
signatures["torch.functional.avg_pool3d"] = {
    "input": "tensor",
    "kernel_size": "integer", # can also be a tuple
    "stride": "integer", # can also be a tuple
    "padding": "integer", # can also be a tuple
    "ceil_mode": "boolean",
    "count_include_pad": "boolean"
}
signatures["torch.functional.dropout"] = {
    "input": "tensor",
    "p": "float",
    "training": "boolean",
    "inplace": "boolean"
}
signatures["torch.functional.dropout2d"] = {
    "input": "tensor",
    "p": "float",
    "training": "boolean",
    "inplace": "boolean"
}
signatures["torch.functional.dropout3d"] = {
    "input": "tensor",
    "p": "float",
    "training": "boolean",
    "inplace": "boolean"
}
signatures["torch.functional.linear"] = {
    "input": "tensor",
    "weight": "tensor",
    "bias": "tensor"
}
signatures["torch.functional.batch_norm"] = {
    "input": "tensor",
    "running_mean": "tensor",
    "running_var": "tensor",
    "weight": "tensor",
    "bias": "tensor",
    "training": "boolean",
    "momentum": "float",
    "eps": "float"
}
signatures["torch.functional.layer_norm"] = {
    "input": "tensor",
    "normalized_shape": "list", # Can also be single integer
    "weight": "tensor",
    "bias": "tensor",
    "eps": "float"
}
signatures["torch.functional.group_norm"] = {
    "input": "tensor",
    "num_groups": "integer",
    "weight": "tensor",
    "bias": "tensor",
    "eps": "float"
}
signatures["torch.functional.embedding"] = {
    "input": "tensor",
    "weight": "tensor",
    "padding_idx": "integer",
    "max_norm": "float",
    "norm_type": "float",
    "scale_grad_by_freq": "boolean",
    "sparse": "boolean"
}
signatures["torch.functional.embedding_bag"] = {
    "weight": "tensor",
    "indices": "tensor",
    "offsets": "tensor",
    "scale_grad_by_freq": "boolean",
    "mode": "string",
    "sparse": "boolean",
    "per_sample_weights": "tensor"
}
signatures["torch.fused_moving_avg_obs_fake_quant"] = {
    "input": "tensor",
    "running_min": "tensor",
    "running_max": "tensor",
    "scale": "tensor",
    "zero_point": "tensor",
    "averaging_constant": "float",
    "quant_min": "integer",
    "quant_max": "integer"
}
signatures["torch.futures.Future"] = {} # Empty signature because it's a class constructor
signatures["torch.futures.Future.wait"] = {} # Method with no arguments.
signatures["torch.futures.Future.result"] = {} # Method with no arguments.
signatures["torch.futures.Future.cancel"] = {
    "msg": "string"
}
signatures["torch.futures.Future.done"] = {} # Method with no arguments.
signatures["torch.futures.Future.then"] = {
    "callback": "list" # Assuming a list of callbacks is the most common usage, even if other callables might be acceptable.
}
signatures["torch.futures.Future.set_result"] = {
    "result": "tensor" # Assuming tensors are the most common result. Could be 'list' or 'tuple' for multiple returns, but single tensor is the base case.
}
signatures["torch.futures.Future.set_exception"] = {
    "exception": "string" # Exceptions are normally string messages or class instances, but we use "string" for simplicity.
}
signatures["torch.fx.symbolic_trace"] = {
    "root": "object", # unsure, likely an object representing the module to trace
    "concrete_args": "tuple" # unsure, likely a tuple of concrete arguments
}
signatures["torch.fx.GraphModule"] = {
    "graph": "object", # unsure, likely a Graph object
    "module": "object" # unsure, likely a nn.Module object
}
signatures["torch.fx.Graph.forward"] = {
} # Seems like a standard module forward and doesn't really have a useful signature beyond self, *args, **kwargs
signatures["torch.fx.Node.format_node"] = {
    "prefix": "string"
}
signatures["torch.fx.Interpreter.run"] = {
    "module": "object", # or GraphModule?
    "args": "tuple",
    "kwargs": "object" # unsure, likely dict
}
signatures["torch.gather"] = {
    "input": "tensor",
    "dim": "integer",
    "index": "tensor",
    "sparse_grad": "boolean" # Optional argument, but including it for completeness
}
signatures["torch.gcd"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.gcd_"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.geqrf"] = {
    "input": "tensor"
}
signatures["torch.ger"] = {
    "input": "tensor",
    "vec2": "tensor"
}
signatures["torch.get_autocast_cpu_dtype"] = {} # This api has no arguments.
signatures["torch.get_autocast_dtype"] = {
}
signatures["torch.get_autocast_gpu_dtype"] = {}
signatures["torch.get_autocast_ipu_dtype"] = {
    "device": "string" # Could also be a device object, but string is simpler
}
signatures["torch.get_autocast_xla_dtype"] = {
    "dtype": "dtype"
}
signatures["torch.get_default_device"] = {}
signatures["torch.get_default_dtype"] = {}
signatures["torch.get_deterministic_debug_mode"] = {}
signatures["torch.get_device"] = {
    "input": "tensor"
}
signatures["torch.get_device_module"] = {
    "device": "string", # Could also be a torch.device object, but "string" seems more common for device representation
    "module": "string"
}
signatures["torch.get_file_path"] = {
    "url": "string",
    "model_dir": "string",
    "progress": "boolean"
}
signatures["torch.get_num_interop_threads"] = {}
signatures["torch.get_num_threads"] = {}
signatures["torch.get_rng_state"] = {}
signatures["torch.glob"] = {
    "pathname": "string",
    "recursive": "boolean"
}
signatures["torch.gradient"] = {
    "input": "tensor",
    "spacing": "list", # Could also be scalar or tensor_list, choosing list for simplicity
    "dim": "list", # Could also be integer, choosing list for consistency
    "edge_order": "integer"
}
signatures["torch.greater"] = {
    "input": "tensor",
    "other": "tensor" # could also be float/integer but tensor is more general
}
signatures["torch.greater_equal"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.grid_sampler"] = {
    "input": "tensor",
    "grid": "tensor",
    "interpolation_mode": "string",
    "padding_mode": "string",
    "align_corners": "boolean"
}
signatures["torch.group_norm"] = {
    "input": "tensor",
    "num_groups": "integer",
    "weight": "tensor",
    "bias": "tensor",
    "eps": "float",
    "affine": "boolean"
}
signatures["torch.gru"] = {
    "input": "tensor",
    "h_0": "tensor",
    "weight": "tensor",
    "bias": "tensor",
    "num_layers": "integer",
    "dropout": "float",
    "train": "boolean",
    "bidirectional": "boolean",
    "batch_first": "boolean"
}
signatures["torch.gru_cell"] = {
    "input": "tensor",
    "hx": "tensor",
    "weight_ih": "tensor",
    "weight_hh": "tensor",
    "bias_ih": "tensor",
    "bias_hh": "tensor"
}
signatures["torch.hamming_window"] = {
    "window_length": "integer",
    "periodic": "boolean",
    "alpha": "float",
    "beta": "float",
    "dtype": "dtype"
}
signatures["torch.hann_window"] = {
    "window_length": "integer",
    "periodic": "boolean",
    "dtype": "dtype",
    "layout": "string", # should be torch.layout but string is closest
    "requires_grad": "boolean"
}
signatures["torch.hardshrink"] = {
    "input": "tensor",
    "lambd": "float" # Could be argued that lambd can be an integer as well, but float is more common
}
signatures["torch.hinge_embedding_loss"] = {
    "input": "tensor",
    "target": "tensor",
    "margin": "float",
    "reduction": "string"
}
signatures["torch.histogram"] = {
    "input": "tensor",
    "bins": "integer", # Could also be a tensor, but integer is more common according to docs
    "range": "tuple",
    "weight": "tensor",
    "density": "boolean"
}
signatures["torch.histogramdd"] = {
    "input": "tensor",
    "bins": "list", # Could also be a tensor_list or integer, but list seems most general
    "range": "list", # Should be a tuple, but list is the closest available option
    "weight": "tensor",
    "density": "boolean"
}
signatures["torch.hsmm"] = {
    "log_probs": "tensor",
    "transition_matrix": "tensor",
    "emission_matrix": "tensor",
    "lengths": "tensor", # Could be integer_list if only integer values are allowed in the tensor
    "init_dist": "tensor",
    "return_log_likelihood": "boolean"
}
signatures["torch.hsplit"] = {
    "input": "tensor",
    "indices_or_sections": "list" #Could also be integer or tuple of ints, but list seems most general
}
signatures["torch.hspmm"] = {
    "mat1": "tensor",
    "mat2": "tensor"
}
signatures["torch.hub.load"] = {
    "repo_or_dir": "string",
    "model": "string",
    "source": "string",
    "force_reload": "boolean",
    "verbose": "boolean",
    "skip_validation": "boolean",
    "pretrained": "boolean",
    "trust_repo": "boolean"
}
signatures["torch.hub.help"] = {
    "repo_or_dir": "string",
    "model": "string",
    "force_reload": "boolean",
    "trust_repo": "boolean"
}
signatures["torch.hub.list"] = {
    "repo_or_dir": "string",
    "force_reload": "boolean",
    "trust_repo": "boolean"
}
signatures["torch.hub.get_dir"] = {}
signatures["torch.igammac"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.iinfo"] = {
    "dtype": "dtype"
}
signatures["torch.imag"] = {
    "input": "tensor"
}
signatures["torch.import_ir_module"] = {
    "module_string": "string",
    "extra_library_paths": "list", # could be a list of strings, but list seems more general
    "extra_operation_namespaces": "list", #could be list of strings
    "import_jit_module": "boolean"
}
signatures["torch.import_ir_module_from_buffer"] = {
    "buffer": "string", # Could potentially be a byte array but "string" seems more appropriate for a buffer
    "extra_files": "list",
    "module_name": "string",
    "loaded_module": "string",
}
signatures["torch.importlib"] = {
    "name": "string",
    "package": "string" # Could potentially be None, should maybe be more broad e.g. "object"
}
signatures["torch.index_add"] = {
    "input": "tensor",
    "dim": "integer",
    "index": "tensor",
    "source": "tensor"
}
signatures["torch.index_copy"] = {
    "input": "tensor",
    "dim": "integer",
    "index": "tensor",
    "source": "tensor"
}
signatures["torch.index_fill"] = {
    "input": "tensor",
    "dim": "integer",
    "index": "tensor",
    "value": "float" # Could be float or tensor, choosing the most common usage which I believe is float.
}
signatures["torch.index_put"] = {
    "input": "tensor",
    "indices": "tuple", # could be list as well
    "values": "tensor",
    "accumulate": "boolean"
}
signatures["torch.index_put_"] = {
    "input": "tensor",
    "indices": "tuple", # Could also be a list of tensors
    "values": "tensor",
    "accumulate": "boolean"
}
signatures["torch.index_reduce"] = {
    "input": "tensor",
    "dim": "integer",
    "index": "tensor",
    "source": "tensor",
    "reduce": "string",
    "include_self": "boolean"
}
signatures["torch.indices_copy"] = {
    "self": "tensor",
    "source": "tensor"
}
signatures["torch.inference_mode"] = {
    "mode": "boolean" # I think this should be an optional boolean that defaults to True if no argument is provided.
}
signatures["torch.init_num_threads"] = {
    "threads": "integer"
}
signatures["torch.initial_seed"] = {} # No arguments
signatures["torch.inspect"] = {
    "input": "tensor"
}
signatures["torch.instance_norm"] = {
    "input": "tensor",
    "running_mean": "tensor",
    "running_var": "tensor",
    "weight": "tensor",
    "bias": "tensor",
    "use_input_stats": "boolean",
    "momentum": "float",
    "eps": "float"
}
signatures["torch.int_repr"] = {
    "input": "tensor"
}
signatures["torch.inverse"] = {
    "input": "tensor"
}
signatures["torch.is_anomaly_check_nan_enabled"] = {}
signatures["torch.is_anomaly_enabled"] = {}
signatures["torch.is_autocast_cache_enabled"] = {}
signatures["torch.is_autocast_cpu_enabled"] = {}
signatures["torch.is_autocast_enabled"] = {}
signatures["torch.is_autocast_ipu_enabled"] = {}
signatures["torch.is_autocast_xla_enabled"] = {}
signatures["torch.is_complex"] = {
    "input": "tensor"
}
signatures["torch.is_conj"] = {
    "input": "tensor"
}
signatures["torch.is_deterministic_algorithms_warn_only_enabled"] = {}
signatures["torch.is_distributed"] = {}
signatures["torch.is_floating_point"] = {
    "input": "tensor"
}
signatures["torch.is_grad_enabled"] = {}
signatures["torch.is_inference"] = {
}
signatures["torch.is_inference_mode_enabled"] = {}
signatures["torch.is_neg"] = {
    "input": "tensor"
}
signatures["torch.is_nonzero"] = {
    "input": "tensor"
}
signatures["torch.is_same_size"] = {
    "tensor1": "tensor",
    "tensor2": "tensor"
}
signatures["torch.is_signed"] = {
    "input": "tensor"
}
signatures["torch.is_storage"] = {
    "obj": "object" # I am not sure what the exact object type should be, but "object" seems the most general.
}
signatures["torch.is_vulkan_available"] = {}
signatures["torch.is_warn_always_enabled"] = {}
signatures["torch.isin"] = {
    "elements": "tensor", # Could also be scalar, but tensor is more general
    "test_elements": "tensor", # Could also be scalar, but tensor is more general
    "assume_unique": "boolean",
    "invert": "boolean"
}
signatures["torch.istft"] = {
    "input": "tensor",
    "n_fft": "integer",
    "hop_length": "integer",  # Could be None, but integer is more common
    "win_length": "integer",  # Could be None, but integer is more common
    "window": "tensor",  # Could be None, but tensor is more common
    "center": "boolean",
    "normalized": "boolean",
    "onesided": "boolean",  # Could be None, but boolean is more common
    "length": "integer",  # Could be None, but integer is more common
    "return_complex": "boolean"
}
signatures["torch.jit.script"] = {
    "obj": "object" # Could be a module or a function
}
signatures["torch.jit.trace"] = {
    "func": "object", # Callable function
    "example_inputs": "tuple", # List or tuple of tensors, but tuple seems more common
    "check_trace": "boolean",
    "check_inputs": "tuple",
    "check_tolerance": "float",
    "strict": "boolean",
    "_force_outplace": "boolean"
}
signatures["torch.jit.save"] = {
    "m": "object", # ScriptModule
    "filename": "string",
    "extra_files": "dictionary",
    "_extra_files_abs_path": "string",
    "with_backward_pass": "boolean",
    "format": "string",
    "deduplicate": "boolean"
}
signatures["torch.jit.load"] = {
    "filename": "string",
    "map_location": "string", # Can also be a function, but string is probably the most common
    "_extra_files": "dictionary",
    "_validate_shapes": "boolean"
}
signatures["torch.jit.freeze"] = {
    "mod": "object", # ScriptModule
    "preserve_mobile": "boolean",
    "inline_everything": "boolean"
}
signatures["torch.jit.Attribute"] = {
    "value": "tensor", # Could be other types as well, but tensor seems like a reasonable default given the context of TorchScript
    "type": "dtype" # This is a Python type, but "dtype" seems like the closest match
}
signatures["torch.jit.CompilationUnit"] = {} # Empty signature, as CompilationUnit is a class, not a function with arguments
signatures["torch.jit.Error"] = {
    "msg": "string"
}
signatures["torch.jit.Final"] = {
    "fn": "list" # could also be a function itself, but list seems to be the more common case
}
signatures["torch.jit.Future"] = {}
signatures["torch.jit.Iterator"] = {
}
signatures["torch.jit.ONNXTracedModule"] = {
    "graph": "string", # I am assuming graph is a string representation of the ONNX graph
    "input_names": "list",
    "output_names": "list"
}
signatures["torch.jit.RecursiveScriptClass"] = {
}
signatures["torch.jit.RecursiveScriptModule"] = {
}
signatures["torch.jit.ScriptFunction.save"] = {
    "filename": "string",
    "_extra_files": "dict" # Could also be "list" or "tuple" depending on the actual use case, but "dict" seems more appropriate based on the description
}
signatures["torch.jit.ScriptFunction.save_to_buffer"] = {
    "_extra_files": "dict" # Could also be "list" or "tuple" depending on the actual use case, but "dict" seems more appropriate based on the description
}
signatures["torch.jit.ScriptFunction.get_debug_state"] = {}
signatures["torch.jit.ScriptModule.add_module"] = {
    "name": "string",
    "module": "Module" # Should be torch.nn.Module
}
signatures["torch.jit.ScriptModule.apply"] = {
    "fn": "Module" # Should be Callable[[nn.Module], None]
}
signatures["torch.jit.ScriptModule.bfloat16"] = {}
signatures["torch.jit.ScriptModule.buffers"] = {
    "recurse": "boolean"
}
signatures["torch.jit.ScriptModule.children"] = {}
signatures["torch.jit.ScriptModule.compile"] = {
    "args": "list",
    "kwargs": "list"
}
signatures["torch.jit.ScriptModule.cpu"] = {}
signatures["torch.jit.ScriptModule.cuda"] = {
    "device": "integer"
}
signatures["torch.jit.ScriptModule.double"] = {}
signatures["torch.jit.ScriptModule.eval"] = {}
signatures["torch.jit.ScriptModule.extra_repr"] = {}
signatures["torch.jit.ScriptModule.float"] = {}
signatures["torch.jit.ScriptModule.get_buffer"] = {
    "target": "string"
}
signatures["torch.jit.ScriptModule.get_extra_state"] = {}
signatures["torch.jit.ScriptModule.get_parameter"] = {
    "target": "string"
}
signatures["torch.jit.ScriptModule.get_submodule"] = {
    "target": "string"
}
signatures["torch.jit.ScriptModule.half"] = {}
signatures["torch.jit.ScriptModule.ipu"] = {
    "device": "integer"
}
signatures["torch.jit.ScriptModule.load_state_dict"] = {
    "state_dict": "list", #Should be dict
    "strict": "boolean",
    "assign": "boolean"
}
signatures["torch.jit.ScriptModule.modules"] = {}
signatures["torch.jit.ScriptModule.mtia"] = {
    "device": "integer"
}
signatures["torch.jit.ScriptModule.named_buffers"] = {
    "prefix": "string",
    "recurse": "boolean",
    "remove_duplicate": "boolean"
}
signatures["torch.jit.ScriptModule.named_children"] = {}
signatures["torch.jit.ScriptModule.named_modules"] = {
    "memo": "list", #Optional[set['Module']]
    "prefix": "string",
    "remove_duplicate": "boolean"
}
signatures["torch.jit.ScriptModule.named_parameters"] = {
    "prefix": "string",
    "recurse": "boolean",
    "remove_duplicate": "boolean"
}
signatures["torch.jit.ScriptModule.parameters"] = {
    "recurse": "boolean"
}
signatures["torch.jit.ScriptModule.register_backward_hook"] = {
    "hook": "Module" #Should be Callable
}
signatures["torch.jit.ScriptModule.register_buffer"] = {
    "name": "string",
    "tensor": "tensor",
    "persistent": "boolean"
}
signatures["torch.jit.ScriptModule.register_forward_hook"] = {
    "hook": "Module", #Should be Callable
    "prepend": "boolean",
    "with_kwargs": "boolean",
    "always_call": "boolean"
}
signatures["torch.jit.ScriptModule.register_forward_pre_hook"] = {
    "hook": "Module", #Should be Callable
    "prepend": "boolean",
    "with_kwargs": "boolean"
}
signatures["torch.jit.ScriptModule.register_full_backward_hook"] = {
    "hook": "Module", #Should be Callable
    "prepend": "boolean"
}
signatures["torch.jit.ScriptModule.register_full_backward_pre_hook"] = {
    "hook": "Module", #Should be Callable
    "prepend": "boolean"
}
signatures["torch.jit.ScriptModule.register_load_state_dict_post_hook"] = {
    "hook": "Module", #Should be Callable
}
signatures["torch.jit.ScriptModule.register_load_state_dict_pre_hook"] = {
    "hook": "Module", #Should be Callable
}
signatures["torch.jit.ScriptModule.register_module"] = {
    "name": "string",
    "module": "Module" # Should be torch.nn.Module
}
signatures["torch.jit.ScriptModule.register_parameter"] = {
    "name": "string",
    "param": "tensor" # Should be nn.Parameter
}
signatures["torch.jit.ScriptModule.register_state_dict_post_hook"] = {
    "hook": "Module", #Should be Callable
}
signatures["torch.jit.ScriptModule.register_state_dict_pre_hook"] = {
    "hook": "Module", #Should be Callable
}
signatures["torch.jit.ScriptModule.requires_grad_"] = {
    "requires_grad": "boolean"
}
signatures["torch.jit.ScriptModule.save"] = {
    "f": "string", #Should be file-like object
    "kwargs": "list"
}
signatures["torch.jit.ScriptModule.set_extra_state"] = {
    "state": "list" #Should be dict
}
signatures["torch.jit.ScriptModule.set_submodule"] = {
    "target": "string",
    "module": "Module", # Should be torch.nn.Module
    "strict": "boolean"
}
signatures["torch.jit.ScriptModule.share_memory"] = {}
signatures["torch.jit.ScriptModule.state_dict"] = {
    "args": "list",
    "destination": "list", #Should be dict
    "prefix": "string",
    "keep_vars": "boolean"
}
signatures["torch.jit.ScriptModule.to"] = {
    "args": "list",
    "kwargs": "list"
}
signatures["torch.jit.ScriptModule.to_empty"] = {
    "device": "list", #Should be torch.device
    "recurse": "boolean"
}
signatures["torch.jit.ScriptModule.train"] = {
    "mode": "boolean"
}
signatures["torch.jit.ScriptModule.type"] = {
    "dst_type": "string" # Should be type
}
signatures["torch.jit.ScriptModule.xpu"] = {
    "device": "integer"
}
signatures["torch.jit.ScriptModule.zero_grad"] = {
    "set_to_none": "boolean"
}
signatures["torch.jit.ScriptWarning"] = {
}
signatures["torch.jit.TopLevelTracedModule"] = {
    "concrete_type_builder": "list" # Could also be a tuple but list seems more appropriate
}
signatures["torch.jit.TracedModule"] = {
    "f": "tensor", # Could also be callable, but tensor is more common for tracing
    "example_inputs": "tuple",
    "check_trace": "boolean",
    "check_inputs": "list",
    "strict": "boolean"
}
signatures["torch.jit.TracerWarning"] = {} # TracerWarning is a class, not a function. It doesn't have arguments.
signatures["torch.jit.TracingCheckError"] = {
    "message": "string"
}
signatures["torch.jit.annotate"] = {
    "the_type": "string", # Could be a type, but representing it as a string for simplicity
    "the_value": "tensor" # Could be any type, but assuming tensor is the most common case
}
signatures["torch.jit.annotations"] = {
    "obj": "string", # I am unsure about the proper type here.
    "ann": "string" # I am unsure about the proper type here.
}
signatures["torch.jit.contextmanager"] = {
    "fn": "string" # or maybe a function/callable type would be more appropriate?
}
signatures["torch.jit.enable_onednn_fusion"] = {
    "fusion": "boolean"
}
signatures["torch.jit.export"] = {
    "mod": "object", # Might be a Module instead of object
    "f": "string",
    "extra_files": "dictionary", # I am unsure about what should be put here
    "_disable_param_attr_dedup": "boolean",
    "_retain_param_name": "boolean"
}
signatures["torch.jit.export_opnames"] = {
    "mod": "object", # Assuming 'mod' is a module or a model which is an object
    "file_name": "string"
}
signatures["torch.jit.fork"] = {
    "func": "callable", # Could also be torch.nn.Module but callable is more general
    "args": "list", # Represents *args, a variable number of positional arguments
    "kwargs": "list" # Represents **kwargs, a variable number of keyword arguments
}
signatures["torch.jit.freeze"] = {
    "mod": "tensor", # Should be ScriptModule, but closest type is tensor
    "preserved_attrs": "list",
    "optimize_numerics": "boolean"
}
signatures["torch.jit.frontend.parse_ir"] = {
    "ir_string": "string"
}
signatures["torch.jit.frontend.parse_submodule"] = {
    "script_name": "string",
    "submodule_name": "string"
}
signatures["torch.jit.frontend.parse_type_annotation"] = {
    "type_str": "string"
}
signatures["torch.jit.frontend.SourceRange"] = {
    "source": "string", # could be refined to a Source object
    "start": "integer",
    "end": "integer"
}
signatures["torch.jit.frontend.get_jit_class_def"] = {
    "script_name": "string",
    "class_name": "string"
}
signatures["torch.jit.fuser"] = {
    "fuser_name": "string"
}
signatures["torch.jit.ignore"] = {
    "drop": "boolean",
    "kwargs": "tuple" # I'm unsure what type `kwargs` should be here, but it usually involves named arguments packed into a dictionary, which I'm approximating with a tuple for lack of a better fit.
}
signatures["torch.jit.interface"] = {
    "obj": "object" # Could also be "list" or "tuple" depending on use case, but "object" seems more general
}
signatures["torch.jit.is_scripting"] = {}
signatures["torch.jit.is_tracing"] = {}
signatures["torch.jit.isinstance"] = {
    "obj": "list", # Could be "any" but it seems to commonly handle containers.
    "target_type": "list" # Represents the type to check against, e.g., List[str]. Could be "any"
}
signatures["torch.jit.jit_module_from_flatbuffer"] = {
    "flatbuffer": "string", # Could also be a byte array but string seems more representative
    "extra_files": "list"
}
signatures["torch.jit.last_executed_optimized_graph"] = {}
signatures["torch.jit.load"] = {
    "f": "string", # Could also be a file-like object, but string (filename) is more common
    "map_location": "string", # Could also be torch.device, but string is more common in examples
    "_extra_files": "list", # Actually dictionary, but "list" is the closest acceptable type
    "_restore_shapes": "boolean"
}
signatures["torch.jit.mobile.LiteScriptModule"] = {} # Constructor, doesn't have arguments from Python perspective
signatures["torch.jit.mobile.os.cpu_count"] = {}
signatures["torch.jit.mobile.os.getenv"] = {
    "variable": "string"
}
signatures["torch.jit.mobile.os.uname"] = {}
signatures["torch.jit.mobile.torch.avg_pool2d"] = {
    "input": "tensor",
    "kernel_size": "list", # Could also be tuple or integer
    "stride": "list", # Could also be tuple or integer
    "padding": "list", # Could also be tuple or integer
    "ceil_mode": "boolean",
    "count_include_pad": "boolean",
    "divisor_override": "integer"
}
signatures["torch.jit.mobile.torch.batch_norm"] = {
    "input": "tensor",
    "weight": "tensor",
    "bias": "tensor",
    "running_mean": "tensor",
    "running_var": "tensor",
    "training": "boolean",
    "momentum": "float",
    "eps": "float",
}
signatures["torch.jit.mobile.torch.conv2d"] = {
    "input": "tensor",
    "weight": "tensor",
    "bias": "tensor",
    "stride": "tuple", #could be integer or tuple
    "padding": "tuple", #could be integer or tuple
    "dilation": "tuple", #could be integer or tuple
    "groups": "integer",
}
signatures["torch.jit.mobile.torch.dropout"] = {
    "input": "tensor",
    "p": "float",
    "training": "boolean",
}
signatures["torch.jit.mobile.torch.interpolate"] = {
    "input": "tensor",
    "size": "list", # could be None, int, or tuple of ints
    "scale_factor": "list", # could be None, float, or tuple of floats
    "mode": "string",
    "align_corners": "boolean",
    "recompute_scale_factor": "boolean"
}
signatures["torch.jit.mobile.torch.linear"] = {
    "input": "tensor",
    "weight": "tensor",
    "bias": "tensor",
}
signatures["torch.jit.mobile.torch.max_pool2d"] = {
    "input": "tensor",
    "kernel_size": "list", # Could also be tuple or integer
    "stride": "list", # Could also be tuple or integer
    "padding": "list", # Could also be tuple or integer
    "dilation": "list", # Could also be tuple or integer
    "ceil_mode": "boolean",
    "return_indices": "boolean"
}
signatures["torch.jit.mobile.torch.hardtanh"] = {
    "input": "tensor",
    "min_val": "float",
    "max_val": "float",
}
signatures["torch.jit.mobile.torch.relu"] = {
    "input": "tensor",
    "inplace": "boolean",
}
signatures["torch.jit.mobile.torch.sigmoid"] = {
    "input": "tensor"
}
signatures["torch.jit.mobile.validate_map_location"] = {
    "storage_or_tag": "string", # Could also be a dictionary or function, but string is most common.
    "location": "string" # Could also be a Storage object
}
signatures["torch.jit.onednn_fusion_enabled"] = {
    "enabled": "boolean"
}
signatures["torch.jit.optimize_for_inference"] = {
    "mod": "tensor" # I think it should be a `Module` but `tensor` seems closest
}
signatures["torch.jit.optimized_execution"] = {
    "enabled": "boolean"
}
signatures["torch.jit.run_frozen_optimizations"] = {
    "mod": "object" # Could potentially be "Module", but "object" is more general
}
signatures["torch.jit.save"] = {
    "m": "tensor", # Could also be ScriptModule, but tensor is more general
    "f": "string", # Could also be file-like object, but string is more common
    "_extra_files": "list" # Should be a dictionary, but list is closest option. The dictionary maps from filename to contents.
}
signatures["torch.jit.save_jit_module_to_flatbuffer"] = {
    "module": "object", # Should be "Module", but object is the closest we have
    "f": "string"
}
signatures["torch.jit.script"] = {
    "obj": "obj", # Could be Callable, class, or nn.Module, dictionary, or list. Assuming 'obj' is the closest match
    "optimize": "boolean", # Assuming boolean as optimize is a flag
    "_frames_up": "integer",
    "_rcb": "obj", # Assuming object as the type is not clear
    "example_inputs": "list" # Could also be dict of list of tuples or None. Assuming list for simplicity
}
signatures["torch.jit.script_if_tracing"] = {
    "condition": "boolean",
    "fn": "list", # Could also be a function, but we don't have that as a type
    "alternative_fn": "list" # Could also be a function, but we don't have that as a type
}
signatures["torch.jit.script_method"] = {
}
signatures["torch.jit.set_fusion_strategy"] = {
    "strategy": "list" # potentially should be list of strings, but most closely matches list
}
signatures["torch.jit.set_module"] = {
    "name": "string",
    "module": "module" # It should have been a Module type, but it is not in the list.
}
signatures["torch.jit.strict_fusion"] = {}
signatures["torch.jit.torch"] = {
}
signatures["torch.jit.trace"] = {
    "func": "callable", # Could also be torch.nn.Module, but callable seems more general
    "example_inputs": "tuple", # Can also be tensor or None, but tuple is the most common.
    "optimize": "boolean", # Assuming this is a boolean flag, though documentation doesn't explicitly say.
    "check_trace": "boolean",
    "check_inputs": "list", # List of tuples
    "check_tolerance": "float",
    "strict": "boolean",
    "_force_outplace": "boolean",
    "_module_class": "string", # Assuming string representation of class, though this is an internal argument.
    "_compilation_unit": "string", # Assuming string representation
    "example_kwarg_inputs": "dict",
    "_store_inputs": "boolean"
}
signatures["torch.jit.trace_module"] = {
    "mod": "object", # Should ideally be 'Module', but 'object' is the closest.
    "inputs": "tuple",
    "check_trace": "boolean",
    "check_inputs": "list", # This is a list of tuples, but 'list' is the closest.
    "strict": "boolean",
    "_force_outplace": "boolean"
}
signatures["torch.jit.unused"] = {
    "fn": "list" # Actually, it's a function, but it's being treated as a list of instructions by the compiler. Closest type is list
}
signatures["torch.jit.wait"] = {
    "future": "tensor" # Could also be a more specific type like "torch.jit.Future", but "tensor" is the closest option available
}
signatures["torch.jit.warnings"] = {
    "m": "string" # The documentation specifies it takes string and nodes, string seems more common
}
signatures["torch.kaiser_window"] = {
    "window_length": "integer",
    "periodic": "boolean",
    "beta": "float",
    "dtype": "dtype"
}
signatures["torch.kl_div"] = {
    "input": "tensor",
    "target": "tensor",
    "reduction": "string",
    "log_target": "boolean"
}
signatures["torch.layer_norm"] = {
    "input": "tensor",
    "normalized_shape": "list", # Could also be tuple, but list is more common
    "weight": "tensor",
    "bias": "tensor",
    "eps": "float",
    "elementwise_affine": "boolean"
}
signatures["torch.layout"] = {
}
signatures["torch.lcm_"] = {
    "input": "tensor",
    "other": "tensor" # Could also be integer, but tensor is more common
}
signatures["torch.ldexp"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.ldexp_"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.lerp"] = {
    "input": "tensor",
    "end": "tensor",
    "weight": "float" # Could be tensor as well, but float is more common
}
signatures["torch.less"] = {
    "input": "tensor",
    "other": "tensor" # Could also be float or integer, but tensor is the most general
}
signatures["torch.less_equal"] = {
    "input": "tensor",
    "other": "tensor" # Could also be float/integer, but tensor is most common
}
signatures["torch.library"] = {
    "name": "string",
    "dispatcher_lib": "string", # Should this be a custom type for libraries?
    "stub": "string" # Should this be a custom type for stubs?
}
signatures["torch.linalg.cholesky"] = {
    "A": "tensor",
    "upper": "boolean"
}
signatures["torch.linalg.cross"] = {
    "a": "tensor",
    "b": "tensor",
    "dim": "integer"
}
signatures["torch.linalg.det"] = {
    "A": "tensor"
}
signatures["torch.linalg.eig"] = {
    "A": "tensor"
}
signatures["torch.linalg.eigh"] = {
    "A": "tensor",
    "UPLO": "string" # Could also be a character, but string seems more appropriate.
}
signatures["torch.linalg.eigvalsh"] = {
    "A": "tensor",
    "UPLO": "string" # Could also be a character, but string seems more appropriate.
}
signatures["torch.linalg.householder_product"] = {
    "input": "tensor",
    "tau": "tensor"
}
signatures["torch.linalg.inv"] = {
    "A": "tensor"
}
signatures["torch.linalg.ldl_factor"] = {
    "A": "tensor",
    "hermitian": "boolean"
}
signatures["torch.linalg.ldl_solve"] = {
    "LDLF": "tensor",
    "pivots": "tensor",
    "B": "tensor",
    "hermitian": "boolean"
}
signatures["torch.linalg.lstsq"] = {
    "A": "tensor",
    "B": "tensor",
    "rcond": "float",
    "driver": "string"
}
signatures["torch.linalg.matmul"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.linalg.matrix_norm"] = {
    "A": "tensor",
    "ord": "string", # Can also accept int or tuple
    "dim": "tuple",
    "keepdim": "boolean",
    "dtype": "dtype"
}
signatures["torch.linalg.multi_dot"] = {
    "tensors": "tensor_list"
}
signatures["torch.linalg.norm"] = {
    "A": "tensor",
    "ord": "float", # Can also accept string, int, tuple, but float seems like a common default.
    "dim": "tuple",
    "keepdim": "boolean",
    "dtype": "dtype"
}
signatures["torch.linalg.pinv"] = {
    "A": "tensor",
    "rtol": "float",
    "hermitian": "boolean"
}
signatures["torch.linalg.qr"] = {
    "A": "tensor",
    "mode": "string"
}
signatures["torch.linalg.slogdet"] = {
    "A": "tensor"
}
signatures["torch.linalg.solve"] = {
    "A": "tensor",
    "B": "tensor"
}
signatures["torch.linalg.svd"] = {
    "A": "tensor",
    "compute_uv": "boolean"
}
signatures["torch.linalg.svdvals"] = {
    "A": "tensor"
}
signatures["torch.linalg.vector_norm"] = {
    "input": "tensor",
    "ord": "float", # Can also accept int, string, but float seems most common.
    "dim": "tuple",
    "keepdim": "boolean",
    "dtype": "dtype"
}
signatures["torch.linalg.LinAlgError"] = {
}
signatures["torch.linalg.cholesky"] = {
    "A": "tensor",
    "upper": "boolean"
}
signatures["torch.linalg.cond"] = {
    "A": "tensor",
    "p": "string" # Could also be integer or float depending on norm type
}
signatures["torch.linalg.det"] = {
    "A": "tensor"
}
signatures["torch.linalg.eig"] = {
    "A": "tensor"
}
signatures["torch.linalg.eigh"] = {
    "A": "tensor",
    "UPLO": "string"
}
signatures["torch.linalg.eigvals"] = {
    "A": "tensor"
}
signatures["torch.linalg.eigvalsh"] = {
    "A": "tensor",
    "UPLO": "string"
}
signatures["torch.linalg.householder_product"] = {
    "input": "tensor",
    "tau": "tensor"
}
signatures["torch.linalg.inv"] = {
    "A": "tensor"
}
signatures["torch.linalg.matmul"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.linalg.matrix_norm"] = {
    "A": "tensor",
    "ord": "string", #can be float or integer too
    "dim": "tuple",
    "keepdim": "boolean",
    "dtype": "dtype"
}
signatures["torch.linalg.norm"] = {
    "input": "tensor",
    "ord": "float", # could also be string/integer
    "dim": "list",
    "keepdim": "boolean",
    "dtype": "dtype"
}
signatures["torch.linalg.pinv"] = {
    "A": "tensor",
    "rcond": "float",
    "hermitian": "boolean"
}
signatures["torch.linalg.qr"] = {
    "A": "tensor",
    "mode": "string"
}
signatures["torch.linalg.slogdet"] = {
    "A": "tensor"
}
signatures["torch.linalg.solve"] = {
    "A": "tensor",
    "B": "tensor"
}
signatures["torch.linalg.svd"] = {
    "A": "tensor",
    "compute_uv": "boolean"
}
signatures["torch.linalg.vector_norm"] = {
    "input": "tensor",
    "ord": "float", # could also be string/integer
    "dim": "list",
    "keepdim": "boolean",
    "dtype": "dtype"
}
signatures["torch.linalg.lstsq"] = {
    "A": "tensor",
    "B": "tensor",
    "rcond": "float",
    "driver": "string"
}
signatures["torch.linalg.multi_dot"] = {
    "tensors": "tensor_list"
}
signatures["torch.linalg.cholesky_ex"] = {
    "A": "tensor",
    "upper": "boolean",
    "check_errors": "boolean"
}
signatures["torch.linalg.solve_ex"] = {
    "A": "tensor",
    "B": "tensor",
    "left": "boolean",
    "adjoint": "boolean" # I think this could also be tuple
}
signatures["torch.linalg.svdvals"] = {
    "A": "tensor"
}
signatures["torch.linalg.cross"] = {
    "input": "tensor",
    "other": "tensor",
    "dim": "integer"
}
signatures["torch.linalg.diagonal"] = {
    "input": "tensor",
    "offset": "integer",
    "dim1": "integer",
    "dim2": "integer"
}
signatures["torch.linalg.matmul"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.linalg.vecdot"] = {
    "x": "tensor",
    "y": "tensor",
    "dim": "list"
}
signatures["torch.linalg.pca"] = {
    "A": "tensor",
    "q": "integer",
    "center": "boolean"
}
signatures["torch.linalg.matrix_power"] = {
    "A": "tensor",
    "n": "integer"
}
signatures["torch.linalg.cholesky"] = {
    "A": "tensor",
    "upper": "boolean"
}
signatures["torch.linalg.cholesky_ex"] = {
    "input": "tensor",
    "upper": "boolean",
    "check_errors": "boolean"
}
signatures["torch.linalg.cond"] = {
    "A": "tensor",
    "p": "float" # Could also be string, int, inf, -inf but float seems most general
}
signatures["torch.linalg.cross"] = {
    "input": "tensor",
    "other": "tensor",
    "dim": "integer"
}
signatures["torch.linalg.det"] = {
    "A": "tensor"
}
signatures["torch.linalg.diagonal"] = {
    "A": "tensor",
    "offset": "integer",
    "dim1": "integer",
    "dim2": "integer"
}
signatures["torch.linalg.eig"] = {
    "A": "tensor"
}
signatures["torch.linalg.eigh"] = {
    "A": "tensor",
    "UPLO": "string" # Could also be a specific enum, but string is the closest available type
}
signatures["torch.linalg.eigvals"] = {
    "A": "tensor"
}
signatures["torch.linalg.eigvalsh"] = {
    "A": "tensor",
    "UPLO": "string" # Could also be some enum-like type but string seems closest
}
signatures["torch.linalg.householder_product"] = {
    "input": "tensor",
    "tau": "tensor"
}
signatures["torch.linalg.inv"] = {
    "A": "tensor"
}
signatures["torch.linalg.inv_ex"] = {
    "A": "tensor",
    "getri": "boolean" # could be None but boolean is more common
}
signatures["torch.linalg.ldl_factor"] = {
    "A": "tensor",
    "hermitian": "boolean"
}
signatures["torch.linalg.ldl_factor_ex"] = {
    "A": "tensor",
    "hermitian": "boolean",
    "check_errors": "boolean"
}
signatures["torch.linalg.ldl_solve"] = {
    "LD": "tensor",
    "pivots": "tensor",
    "B": "tensor"
}
signatures["torch.linalg.lstsq"] = {
    "A": "tensor",
    "B": "tensor",
    "rcond": "float",
    "driver": "string"
}
signatures["torch.linalg.lu"] = {
    "A": "tensor",
    "pivot": "boolean"
}
signatures["torch.linalg.lu_factor"] = {
    "A": "tensor",
    "pivot": "boolean"
}
signatures["torch.linalg.lu_factor_ex"] = {
    "A": "tensor",
    "pivot": "boolean"
}
signatures["torch.linalg.lu_solve"] = {
    "LU_data": "tensor",
    "LU_pivots": "tensor",
    "B": "tensor"
}
signatures["torch.linalg.matmul"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.linalg.matrix_exp"] = {
    "A": "tensor"
}
signatures["torch.linalg.matrix_norm"] = {
    "input": "tensor",
    "ord": "string", # Can also be integer, but string is the more common usage when specifying 'fro', 'nuc' etc.
    "dim": "tuple", # Can also be integer, but generally expects a tuple of dimensions
    "keepdim": "boolean",
    "dtype": "dtype"
}
signatures["torch.linalg.matrix_power"] = {
    "input": "tensor",
    "n": "integer"
}
signatures["torch.linalg.matrix_rank"] = {
    "input": "tensor",
    "tol": "float", # Could also accept tensor, but float is more common
    "rtol": "float",
    "hermitian": "boolean"
}
signatures["torch.linalg.multi_dot"] = {
    "tensors": "tensor_list"
}
signatures["torch.linalg.norm"] = {
    "A": "tensor",
    "ord": "float", # Could also be string or None, but float seems most common
    "dim": "tuple", # Could also be integer or None, but tuple seems most common when specified
    "keepdim": "boolean",
    "dtype": "dtype"
}
signatures["torch.linalg.pinv"] = {
    "A": "tensor",
    "atol": "float", # Could also be a tensor
    "rtol": "float", # Could also be a tensor
    "hermitian": "boolean",
    "rcond": "float" # Alias for rtol, keeping it for NumPy compatibility
}
signatures["torch.linalg.qr"] = {
    "A": "tensor",
    "mode": "string" # Could be an enum, but string is closest
}
signatures["torch.linalg.slogdet"] = {
    "A": "tensor"
}
signatures["torch.linalg.solve"] = {
    "A": "tensor",
    "B": "tensor",
    "left": "boolean"
}
signatures["torch.linalg.solve_ex"] = {
    "A": "tensor",
    "B": "tensor",
    "left": "boolean",
    "adjoint_a": "boolean"
}
signatures["torch.linalg.solve_triangular"] = {
    "A": "tensor",
    "b": "tensor",
    "upper": "boolean",
    "transpose": "boolean",
    "unitriangular": "boolean"
}
signatures["torch.linalg.svd"] = {
    "A": "tensor",
    "full_matrices": "boolean",
    "driver": "string"
}
signatures["torch.linalg.svdvals"] = {
    "A": "tensor",
    "driver": "string" # Could also be None, but string seems more appropriate given the options
}
signatures["torch.linalg.tensorinv"] = {
    "A": "tensor",
    "ind": "integer"
}
signatures["torch.linalg.tensorsolve"] = {
    "A": "tensor",
    "B": "tensor",
    "dims": "tuple" # Could also be None, but tuple seems more appropriate
}
signatures["torch.linalg.cholesky"] = {
    "A": "tensor",
    "upper": "boolean"
}
signatures["torch.linalg.cond"] = {
    "A": "tensor",
    "p": "string" # Could be a float or integer depending on the norm, using string for simplicity
}
signatures["torch.linalg.det"] = {
    "A": "tensor"
}
signatures["torch.linalg.eig"] = {
    "A": "tensor"
}
signatures["torch.linalg.eigh"] = {
    "A": "tensor",
    "UPLO": "string"
}
signatures["torch.linalg.eigvals"] = {
    "A": "tensor"
}
signatures["torch.linalg.eigvalsh"] = {
    "A": "tensor",
    "UPLO": "string"
}
signatures["torch.linalg.householder_product"] = {
    "input": "tensor",
    "tau": "tensor"
}
signatures["torch.linalg.inv"] = {
    "A": "tensor"
}
signatures["torch.linalg.inv_ex"] = {
    "A": "tensor",
    "check_errors": "boolean"
}
signatures["torch.linalg.matmul"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.linalg.matrix_norm"] = {
    "A": "tensor",
    "ord": "string", # Can be number or string
    "dim": "tuple",
    "keepdim": "boolean",
    "dtype": "dtype"
}
signatures["torch.linalg.matrix_power"] = {
    "A": "tensor",
    "n": "integer"
}
signatures["torch.linalg.matrix_rank"] = {
    "A": "tensor",
    "tol": "tensor",
    "hermitian": "boolean"
}
signatures["torch.linalg.norm"] = {
    "input": "tensor",
    "ord": "string", # Can be number or string
    "dim": "tuple",
    "keepdim": "boolean",
    "dtype": "dtype"
}
signatures["torch.linalg.pinv"] = {
    "A": "tensor",
    "rtol": "float",
    "atol": "float",
    "hermitian": "boolean"
}
signatures["torch.linalg.qr"] = {
    "A": "tensor",
    "mode": "string"
}
signatures["torch.linalg.slogdet"] = {
    "A": "tensor"
}
signatures["torch.linalg.solve"] = {
    "A": "tensor",
    "B": "tensor"
}
signatures["torch.linalg.solve_ex"] = {
    "A": "tensor",
    "B": "tensor",
    "check_errors": "boolean"
}
signatures["torch.linalg.svd"] = {
    "A": "tensor",
    "compute_uv": "boolean"
}
signatures["torch.linalg.svdvals"] = {
    "A": "tensor"
}
signatures["torch.linalg.tensorinv"] = {
    "a": "tensor",
    "ind": "integer"
}
signatures["torch.linalg.tensorsolve"] = {
    "a": "tensor",
    "b": "tensor",
    "dims": "tuple"
}
signatures["torch.linalg.vecdot"] = {
    "x": "tensor",
    "y": "tensor",
    "dim": "list"
}
signatures["torch.linalg.vander"] = {
    "x": "tensor",
    "N": "integer" # Could also be None, but integer is more common
}
signatures["torch.linalg.vecdot"] = {
    "x": "tensor",
    "y": "tensor",
    "dim": "integer"
}
signatures["torch.linalg.vector_norm"] = {
    "input": "tensor",
    "ord": "float", # Could be also integer, but float is more general as it includes inf values
    "dim": "integer",
    "keepdim": "boolean",
    "dtype": "dtype"
}
signatures["torch.load"] = {
    "f": "string", # Could also be a file-like object, but string filename is more common
    "map_location": "string", # Could be Callable, device, or dict, but string is most common single type
    "pickle_module": "list", # It is a module
    "weights_only": "boolean",
    "mmap": "boolean",
    "pickle_load_args": "list" # It is actually a dictionary, but list is more general
}
signatures["torch.lobpcg"] = {
    "A": "tensor",
    "k": "integer",
    "B": "tensor",
    "X": "tensor",
    "n": "integer",
    "iK": "tensor",
    "niter": "integer",
    "tol": "float",
    "largest": "boolean",
    "method": "string",
    "tracker": "callable", # Best match for a function
    "ortho_iparams": "dict", # Assuming dict is the best fit for "various parameters"
    "ortho_fparams": "dict",
    "ortho_bparams": "dict"
}
signatures["torch.log10"] = {
    "input": "tensor"
}
signatures["torch.log2"] = {
    "input": "tensor"
}
signatures["torch.log_"] = {
    "input": "tensor"
}
signatures["torch.log_softmax"] = {
    "input": "tensor",
    "dim": "integer", # Could be None, but integer is more common
    "dtype": "dtype" # optional
}
signatures["torch.logdet"] = {
    "input": "tensor"
}
signatures["torch.logical_and"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.logical_xor"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.logit_"] = {
    "input": "tensor",
    "eps": "float"
}
signatures["torch.lstm"] = {
    "input": "tensor",
    "hx": "tuple", # Could be tensor or tuple of tensors, but tuple seems more common with multiple layers
    "params": "list",
    "has_biases": "boolean",
    "num_layers": "integer",
    "dropout": "float",
    "train": "boolean",
    "bidirectional": "boolean",
    "batch_first": "boolean"
}
signatures["torch.lstm_cell"] = {
    "input": "tensor",
    "hx": "tuple", # Could be tensor or tuple[tensor, tensor]. Tuple seems more common as it represents (h_0, c_0)
    "weight_ih": "tensor",
    "weight_hh": "tensor",
    "bias_ih": "tensor",
    "bias_hh": "tensor"
}
signatures["torch.lt"] = {
    "input": "tensor",
    "other": "tensor" # Could also be float, but tensor seems more common
}
signatures["torch.lu"] = {
    "A": "tensor",
    "pivot": "boolean",
    "get_infos": "boolean"
}
signatures["torch.lu_unpack"] = {
    "LU_data": "tensor",
    "LU_pivots": "tensor",
    "unpack_data": "boolean",
    "unpack_pivots": "boolean"
}
signatures["torch.manual_seed"] = {
    "seed": "integer"
}
signatures["torch.margin_ranking_loss"] = {
    "input1": "tensor",
    "input2": "tensor",
    "target": "tensor",
    "margin": "float",
    "size_average": "boolean",
    "reduce": "boolean",
    "reduction": "string"
}
signatures["torch.masked"] = {
    "input": "tensor",
    "mask": "tensor",
    "value": "tensor" #Could potentially be "float" or "integer" in some cases, but defaulting to tensor since it can also be a tensor.
}
signatures["torch.masked_fill"] = {
    "input": "tensor",
    "mask": "tensor",
    "value": "float" # Could also be tensor, but float seems more common
}
signatures["torch.masked_scatter"] = {
    "input": "tensor",
    "mask": "tensor",
    "source": "tensor"
}
signatures["torch.masked_select"] = {
    "input": "tensor",
    "mask": "tensor"
}
signatures["torch.math.abs"] = {
    "input": "tensor"
}
signatures["torch.math.acos"] = {
    "input": "tensor"
}
signatures["torch.math.acosh"] = {
    "input": "tensor"
}
signatures["torch.math.add"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.math.angle"] = {
    "input": "tensor"
}
signatures["torch.math.asin"] = {
    "input": "tensor"
}
signatures["torch.math.asinh"] = {
    "input": "tensor"
}
signatures["torch.math.atan"] = {
    "input": "tensor"
}
signatures["torch.math.atan2"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.math.atanh"] = {
    "input": "tensor"
}
signatures["torch.math.ceil"] = {
    "input": "tensor"
}
signatures["torch.math.clamp"] = {
    "input": "tensor",
    "min": "float", # could also be tensor
    "max": "float" # could also be tensor
}
signatures["torch.math.conj"] = {
    "input": "tensor"
}
signatures["torch.math.cos"] = {
    "input": "tensor"
}
signatures["torch.math.cosh"] = {
    "input": "tensor"
}
signatures["torch.math.deg2rad"] = {
    "input": "tensor"
}
signatures["torch.math.div"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.math.erf"] = {
    "input": "tensor"
}
signatures["torch.math.erfc"] = {
    "input": "tensor"
}
signatures["torch.math.exp"] = {
    "input": "tensor"
}
signatures["torch.math.expm1"] = {
    "input": "tensor"
}
signatures["torch.math.floor"] = {
    "input": "tensor"
}
signatures["torch.math.floor_divide"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.math.fmod"] = {
    "input": "tensor",
    "divisor": "tensor"
}
signatures["torch.math.frac"] = {
    "input": "tensor"
}
signatures["torch.math.hypot"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.math.imag"] = {
    "input": "tensor"
}
signatures["torch.math.ldexp"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.math.log"] = {
    "input": "tensor"
}
signatures["torch.math.log10"] = {
    "input": "tensor"
}
signatures["torch.math.log1p"] = {
    "input": "tensor"
}
signatures["torch.math.log2"] = {
    "input": "tensor"
}
signatures["torch.math.mul"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.math.negative"] = {
    "input": "tensor"
}
signatures["torch.math.nextafter"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.math.positive"] = {
    "input": "tensor"
}
signatures["torch.math.pow"] = {
    "input": "tensor",
    "exponent": "tensor" # could also be float
}
signatures["torch.math.rad2deg"] = {
    "input": "tensor"
}
signatures["torch.math.real"] = {
    "input": "tensor"
}
signatures["torch.math.reciprocal"] = {
    "input": "tensor"
}
signatures["torch.math.remainder"] = {
    "input": "tensor",
    "divisor": "tensor"
}
signatures["torch.math.round"] = {
    "input": "tensor"
}
signatures["torch.math.rsqrt"] = {
    "input": "tensor"
}
signatures["torch.math.sigmoid"] = {
    "input": "tensor"
}
signatures["torch.math.sign"] = {
    "input": "tensor"
}
signatures["torch.math.sin"] = {
    "input": "tensor"
}
signatures["torch.math.sinc"] = {
    "input": "tensor"
}
signatures["torch.math.sinh"] = {
    "input": "tensor"
}
signatures["torch.math.sqrt"] = {
    "input": "tensor"
}
signatures["torch.math.square"] = {
    "input": "tensor"
}
signatures["torch.math.sub"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.math.tan"] = {
    "input": "tensor"
}
signatures["torch.math.tanh"] = {
    "input": "tensor"
}
signatures["torch.math.trunc"] = {
    "input": "tensor"
}
signatures["torch.median"] = {
    "input": "tensor",
    "dim": "integer",
    "keepdim": "boolean"
}
signatures["torch.memory_format"] = {
    "input": "tensor",
    "memory_format": "integer" # Could be a torch.memory_format enum, but representing it as an integer seems more common
}
signatures["torch.merge_type_from_type_comment"] = {
    "code": "string",
    "type_comment": "string"
}
signatures["torch.meshgrid"] = {
    "tensors": "tensor_list",
    "indexing": "string" # Could also be enum with values "xy" and "ij"
}
signatures["torch.miopen_batch_norm"] = {
    "input": "tensor",
    "weight": "tensor",
    "bias": "tensor",
    "running_mean": "tensor",
    "running_var": "tensor",
    "training": "boolean",
    "exponential_average_factor": "float",
    "epsilon": "float"
}
signatures["torch.miopen_convolution"] = {
    "input": "tensor",
    "weight": "tensor",
    "bias": "tensor",
    "padding": "list", # or tuple
    "stride": "list", # or tuple
    "dilation": "list", # or tuple
    "groups": "integer",
    "benchmark": "boolean",
    "deterministic": "boolean"
}
signatures["torch.miopen_convolution_add_relu"] = {
    "input": "tensor",
    "weight": "tensor",
    "bias": "tensor",
    "conv_param": "tensor", # Could potentially be a tuple or custom class, but tensor seems most fitting
    "alpha": "float",
    "beta": "float"
}
signatures["torch.miopen_convolution_relu"] = {
    "input": "tensor",
    "weight": "tensor",
    "bias": "tensor",
    "padding": "list", # Could also be tuple
    "stride": "list", # Could also be tuple
    "dilation": "list", # Could also be tuple
    "groups": "integer"
}
signatures["torch.miopen_convolution_transpose"] = {
    "input": "tensor",
    "weight": "tensor",
    "bias": "tensor",
    "padding": "list",
    "output_padding": "list",
    "stride": "list",
    "dilation": "list",
    "groups": "integer"
}
signatures["torch.miopen_depthwise_convolution"] = {
    "input": "tensor",
    "weight": "tensor",
    "bias": "tensor", # Could be None, but tensor is more common
    "padding": "list",
    "stride": "list",
    "dilation": "list",
    "groups": "integer"
}
signatures["torch.miopen_rnn"] = {
    "input": "tensor",
    "weight": "tensor",
    "weight_stride0": "integer",
    "weight_stride1": "integer",
    "hx": "tensor",
    "cx": "tensor",
    "mode": "integer",
    "hidden_size": "integer",
    "num_layers": "integer",
    "batch_first": "boolean",
    "dropout": "float",
    "train": "boolean",
    "bidirectional": "boolean",
    "batch_sizes": "tensor", # unsure if this should be integer
    "dropout_state": "tensor"
}
signatures["torch.mkldnn_convolution"] = {
    "input": "tensor",
    "weight": "tensor",
    "bias": "tensor",
    "padding": "list", # Could be tuple as well
    "stride": "list", # Could be tuple as well
    "dilation": "list", # Could be tuple as well
    "groups": "integer"
}
signatures["torch.mkldnn_linear_backward_weights"] = {
    "input": "tensor",
    "grad_output": "tensor",
    "weight": "tensor"
}
signatures["torch.mkldnn_rnn_layer"] = {
    "input": "tensor",
    "weight": "tensor_list",
    "bias": "tensor",
    "hx": "tensor",
    "cx": "tensor",
    "mode": "string",
    "hidden_size": "integer",
    "num_layers": "integer",
    "has_biases": "boolean",
    "batch_first": "boolean",
    "dropout": "float",
    "train": "boolean",
    "bidirectional": "boolean",
    "batch_sizes": "tensor", # unclear if it's really a tensor or list/tuple of integers
    "dropout_state": "tensor"
}
signatures["torch.mm"] = {
    "input": "tensor",
    "mat2": "tensor"
}
signatures["torch.mode"] = {
    "input": "tensor",
    "dim": "integer",
    "keepdim": "boolean"
}
signatures["torch.monitor"] = {
    "values": "tensor",
    "names": "list", # Assuming names is a list of strings. It could also be a tuple.
    "graph_name": "string",
    "priority": "integer",
    "monitors": "list" # List of torch.utils.tensorboard.writer.SummaryWriter objects
}
signatures["torch.moveaxis"] = {
    "input": "tensor",
    "source": "integer", # Can also be a tuple/list of integers
    "destination": "integer" # Can also be a tuple/list of integers
}
signatures["torch.mps"] = {} #torch.mps is a module, doesn't have a signature
signatures["torch.Tensor.to_metal"] = {
    "asynchronous": "boolean"
}
signatures["torch.mtia.empty"] = {
    "size": "list", # Could also be tuple, but list is more common
    "dtype": "dtype",
    "layout": "string",
    "requires_grad": "boolean",
    "pin_memory": "boolean",
    "memory_format": "string"
}
signatures["torch.mtia.is_available"] = {}
signatures["torch.mtia.synchronize"] = {}
signatures["torch.mul"] = {
    "input": "tensor",
    "other": "tensor" # Could also be float if Number means float
}
signatures["torch.multinomial"] = {
    "input": "tensor",
    "num_samples": "integer",
    "replacement": "boolean",
    "generator": "tensor" # I'm not sure about this one, it could also be a custom object type
}
signatures["torch.multiply"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.multiprocessing.Process"] = {
    "group": "string",  # Could be None as well, but string seems more common
    "target": "list",  # Callable function, represented as list here
    "name": "string",
    "args": "tuple",
    "kwargs": "list",  # Dictionary, represented as list here
    "daemon": "boolean"
}
signatures["torch.multiprocessing.Pool"] = {
    "processes": "integer",
    "initializer": "list", # callable function, represented as list here
    "initargs": "tuple",
    "maxtasksperchild": "integer"
}
signatures["torch.multiprocessing.Queue"] = {
    "maxsize": "integer"
}
signatures["torch.multiprocessing.Event"] = {}
signatures["torch.multiprocessing.Lock"] = {}
signatures["torch.multiprocessing.RLock"] = {}
signatures["torch.multiprocessing.Semaphore"] = {
    "value": "integer"
}
signatures["torch.multiprocessing.BoundedSemaphore"] = {
    "value": "integer"
}
signatures["torch.multiprocessing.Condition"] = {
    "lock": "Lock"  # It takes a Lock or RLock object, representing as Lock here
}
signatures["torch.multiprocessing.Value"] = {
    "typecode_or_type": "string", # or type, representing as string for simplicity
    "*args": "list"
}
signatures["torch.multiprocessing.Array"] = {
    "typecode_or_type": "string", # or type, representing as string for simplicity
    "size_or_initializer": "integer", #or list, representing as integer for simplicity
    "lock": "boolean"
}
signatures["torch.multiprocessing.Pipe"] = {
    "duplex": "boolean"
}
signatures["torch.mvlgamma"] = {
    "input": "tensor",
    "p": "integer" # Could potentially be a tensor as well, but integer seems more common based on context
}
signatures["torch.nan_to_num"] = {
    "input": "tensor",
    "nan": "float",
    "posinf": "float",
    "neginf": "float"
}
signatures["torch.nan_to_num_"] = {
    "input": "tensor",
    "nan": "float",
    "posinf": "float",
    "neginf": "float"
}
signatures["torch.nanmean"] = {
    "input": "tensor",
    "dim": "tuple", # Could also be an integer
    "keepdim": "boolean",
    "dtype": "dtype"
}
signatures["torch.narrow_copy"] = {
    "input": "tensor",
    "dim": "integer",
    "start": "integer",
    "length": "integer"
}
signatures["torch.native_batch_norm"] = {
    "input": "tensor",
    "weight": "tensor",
    "bias": "tensor",
    "running_mean": "tensor",
    "running_var": "tensor",
    "training": "boolean",
    "momentum": "float",
    "eps": "float"
}
signatures["torch.native_channel_shuffle"] = {
    "input": "tensor",
    "groups": "integer"
}
signatures["torch.native_dropout"] = {
    "input": "tensor",
    "p": "float",
    "train": "boolean"
}
signatures["torch.native_group_norm"] = {
    "input": "tensor",
    "weight": "tensor",
    "bias": "tensor",
    "N": "integer",
    "C": "integer",
    "H": "integer",
    "W": "integer",
    "group": "integer",
    "eps": "float"
}
signatures["torch.native_layer_norm"] = {
    "input": "tensor",
    "normalized_shape": "list", # Could also be tuple, but list is more common for shape parameters
    "weight": "tensor",
    "bias": "tensor",
    "eps": "float"
}
signatures["torch.native_norm"] = {
    "input": "tensor",
    "p": "float", # Can also be 'string' for 'fro', 'nuc', etc., but float (for numbers) is probably more common.
    "dim": "list", # Can also be "integer", but usually a tuple or list of dimensions
    "keepdim": "boolean"
}
signatures["torch.ne"] = {
    "input": "tensor",
    "other": "tensor" # Could also be float, but tensor is more general
}
signatures["torch.neg"] = {
    "input": "tensor"
}
signatures["torch.neg_"] = {
    "input": "tensor"
}
signatures["torch.negative"] = {
    "input": "tensor"
}
signatures["torch.negative_"] = {
    "input": "tensor"
}
signatures["torch.nested.nested_tensor"] = {
    "data": "list", # Could also be tuple
    "dtype": "dtype",
    "device": "string", # Although device is not allowed, there is a device argument in the signature
    "requires_grad": "boolean"
}
signatures["torch.nested.as_nested_tensor"] = {
    "data": "tensor" # Could also be a list or tuple. Choosing "tensor" for simplicity
}
signatures["torch.nested.to_padded_tensor"] = {
    "self": "tensor",
    "padding": "float" # or integer
}
signatures["torch.nested.from_padded_tensor"] = {
    "tensor": "tensor",
    "nested_size": "list"
}
signatures["torch.nn.Module.forward"] = {
}
signatures["torch.nn.Module.register_buffer"] = {
    "name": "string",
    "tensor": "tensor", # Could be NoneType but tensor is most common
    "persistent": "boolean"
}
signatures["torch.nn.Module.register_parameter"] = {
    "name": "string",
    "param": "tensor" # Could be NoneType but tensor is most common
}
signatures["torch.nn.Module.add_module"] = {
    "name": "string",
    "module": "tensor" # it is nn.Module object, but representing it with tensor
}
signatures["torch.nn.Module.apply"] = {
    "fn": "string" # it is function type, but representing it with string
}
signatures["torch.nn.Module.cuda"] = {
}
signatures["torch.nn.Module.cpu"] = {
}
signatures["torch.nn.Module.to"] = {
    "dtype": "dtype" # Could be device or tensor, but dtype is one of the most common calls
}
signatures["torch.nn.Module.parameters"] = {
    "recurse": "boolean"
}
signatures["torch.nn.Module.named_parameters"] = {
    "recurse": "boolean"
}
signatures["torch.nn.Module.children"] = {
}
signatures["torch.nn.Module.named_children"] = {
}
signatures["torch.nn.Module.modules"] = {
    "recurse": "boolean"
}
signatures["torch.nn.Module.named_modules"] = {
    "memo": "list", #OrderedDict
    "prefix": "string",
    "remove_duplicate": "boolean"
}
signatures["torch.nn.Module.zero_grad"] = {
    "set_to_none": "boolean"
}
signatures["torch.nn.Module.share_memory"] = {
}
signatures["torch.nn.Module.state_dict"] = {
    "destination": "list", #OrderedDict
    "prefix": "string",
    "keep_vars": "boolean"
}
signatures["torch.nn.Module.load_state_dict"] = {
    "state_dict": "list", #OrderedDict
    "strict": "boolean"
}
signatures["torch.nn.Module.buffers"] = {
    "recurse": "boolean"
}
signatures["torch.nn.Module.named_buffers"] = {
    "recurse": "boolean",
    "prefix": "string",
    "remove_duplicate": "boolean"
}
signatures["torch.nn.Module.train"] = {
    "mode": "boolean"
}
signatures["torch.nn.Module.eval"] = {
}
signatures["torch.nn.Module.requires_grad_"] = {
    "requires_grad": "boolean"
}
signatures["torch.nn.Module.float"] = {
}
signatures["torch.nn.Module.double"] = {
}
signatures["torch.nn.Module.half"] = {
}
signatures["torch.nn.Module.type"] = {
    "dst_type": "string"
}
signatures["torch.nn.Module.is_floating_point"] = {
}
signatures["torch.nn.Module.reset_parameters"] = {
}
signatures["torch.nn.Linear"] = {
    "in_features": "integer",
    "out_features": "integer",
    "bias": "boolean"
}
signatures["torch.nn.Conv2d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", #could also be int
    "stride": "tuple", #could also be int
    "padding": "tuple", #could also be int
    "dilation": "tuple", #could also be int
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.MaxPool2d"] = {
    "kernel_size": "tuple", #could also be int
    "stride": "tuple", #could also be int
    "padding": "tuple", #could also be int
    "dilation": "integer",
    "return_indices": "boolean",
    "ceil_mode": "boolean"
}
signatures["torch.nn.ReLU"] = {
    "inplace": "boolean"
}
signatures["torch.nn.Sigmoid"] = {
}
signatures["torch.nn.Tanh"] = {
}
signatures["torch.nn.BatchNorm2d"] = {
    "num_features": "integer",
    "eps": "float",
    "momentum": "float",
    "affine": "boolean",
    "track_running_stats": "boolean"
}
signatures["torch.nn.Dropout"] = {
    "p": "float",
    "inplace": "boolean"
}
signatures["torch.nn.Embedding"] = {
    "num_embeddings": "integer",
    "embedding_dim": "integer",
    "padding_idx": "integer", # Optional[int] = None
    "max_norm": "float", # Optional[float] = None
    "norm_type": "float",
    "scale_grad_by_freq": "boolean",
    "sparse": "boolean",
    "_weight": "tensor"
}
signatures["torch.nn.CrossEntropyLoss"] = {
    "weight": "tensor",
    "size_average": "boolean",
    "ignore_index": "integer",
    "reduce": "boolean",
    "reduction": "string",
    "label_smoothing": "float"
}
signatures["torch.nn.LSTM"] = {
    "input_size": "integer",
    "hidden_size": "integer",
    "num_layers": "integer",
    "bias": "boolean",
    "batch_first": "boolean",
    "dropout": "float",
    "bidirectional": "boolean",
    "proj_size": "integer"
}
signatures["torch.nn.RNN"] = {
    "input_size": "integer",
    "hidden_size": "integer",
    "num_layers": "integer",
    "nonlinearity": "string",
    "bias": "boolean",
    "batch_first": "boolean",
    "dropout": "float",
    "bidirectional": "boolean"
}
signatures["torch.nn.GRU"] = {
    "input_size": "integer",
    "hidden_size": "integer",
    "num_layers": "integer",
    "bias": "boolean",
    "batch_first": "boolean",
    "dropout": "float",
    "bidirectional": "boolean"
}
signatures["torch.nn.Sequential"] = {
    "*args": "tensor", #Module
}
signatures["torch.nn.AdaptiveAvgPool2d"] = {
    "output_size": "tuple" # Can be tuple or integer
}
signatures["torch.nn.AvgPool2d"] = {
    "kernel_size": "tuple", #Can be tuple or integer
    "stride": "tuple", #Can be tuple or integer
    "padding": "tuple", #Can be tuple or integer
    "ceil_mode": "boolean",
    "count_include_pad": "boolean",
    "divisor_override": "integer"
}
signatures["torch.nn.Transformer"] = {
    "d_model": "integer",
    "nhead": "integer",
    "num_encoder_layers": "integer",
    "num_decoder_layers": "integer",
    "dim_feedforward": "integer",
    "dropout": "float",
    "activation": "string",
    "custom_encoder": "tensor", #Module
    "custom_decoder": "tensor", #Module
    "layer_norm_eps": "float",
    "batch_first": "boolean",
    "norm_first": "boolean"
}
signatures["torch.nn.TransformerEncoder"] = {
    "encoder_layer": "tensor", #Module
    "num_layers": "integer",
    "norm": "tensor" #Module
}
signatures["torch.nn.TransformerDecoder"] = {
    "decoder_layer": "tensor", #Module
    "num_layers": "integer",
    "norm": "tensor" #Module
}
signatures["torch.nn.TransformerEncoderLayer"] = {
    "d_model": "integer",
    "nhead": "integer",
    "dim_feedforward": "integer",
    "dropout": "float",
    "activation": "string",
    "layer_norm_eps": "float",
    "batch_first": "boolean",
    "norm_first": "boolean"
}
signatures["torch.nn.TransformerDecoderLayer"] = {
    "d_model": "integer",
    "nhead": "integer",
    "dim_feedforward": "integer",
    "dropout": "float",
    "activation": "string",
    "layer_norm_eps": "float",
    "batch_first": "boolean",
    "norm_first": "boolean"
}
signatures["torch.nn.Softmax"] = {
    "dim": "integer"
}
signatures["torch.nn.LogSoftmax"] = {
    "dim": "integer"
}
signatures["torch.nn.MSELoss"] = {
    "size_average": "boolean",
    "reduce": "boolean",
    "reduction": "string"
}
signatures["torch.nn.L1Loss"] = {
    "size_average": "boolean",
    "reduce": "boolean",
    "reduction": "string"
}
signatures["torch.nn.BCELoss"] = {
    "weight": "tensor",
    "size_average": "boolean",
    "reduce": "boolean",
    "reduction": "string"
}
signatures["torch.nn.BCEWithLogitsLoss"] = {
    "weight": "tensor",
    "size_average": "boolean",
    "reduce": "boolean",
    "reduction": "string",
    "pos_weight": "tensor"
}
signatures["torch.nn.KLDivLoss"] = {
    "size_average": "boolean",
    "reduce": "boolean",
    "reduction": "string",
    "log_target": "boolean"
}
signatures["torch.nn.MarginRankingLoss"] = {
    "margin": "float",
    "size_average": "boolean",
    "reduce": "boolean",
    "reduction": "string"
}
signatures["torch.nn.MultiLabelMarginLoss"] = {
    "size_average": "boolean",
    "reduce": "boolean",
    "reduction": "string"
}
signatures["torch.nn.SoftMarginLoss"] = {
    "size_average": "boolean",
    "reduce": "boolean",
    "reduction": "string"
}
signatures["torch.nn.CosineEmbeddingLoss"] = {
    "margin": "float",
    "size_average": "boolean",
    "reduce": "boolean",
    "reduction": "string"
}
signatures["torch.nn.HingeEmbeddingLoss"] = {
    "margin": "float",
    "size_average": "boolean",
    "reduce": "boolean",
    "reduction": "string"
}
signatures["torch.nn.MultiLabelSoftMarginLoss"] = {
    "weight": "tensor",
    "size_average": "boolean",
    "reduce": "boolean",
    "reduction": "string"
}
signatures["torch.nn.PoissonNLLLoss"] = {
    "log_input": "boolean",
    "full": "boolean",
    "size_average": "boolean",
    "eps": "float",
    "reduce": "boolean",
    "reduction": "string"
}
signatures["torch.nn.CTCLoss"] = {
    "blank": "integer",
    "reduction": "string",
    "zero_infinity": "boolean"
}
signatures["torch.nn.TripletMarginLoss"] = {
    "margin": "float",
    "p": "float",
    "eps": "float",
    "swap": "boolean",
    "size_average": "boolean",
    "reduce": "boolean",
    "reduction": "string"
}
signatures["torch.nn.TripletMarginWithDistanceLoss"] = {
    "distance_function": "string", #Callable, it is difficult to determine
    "margin": "float",
    "eps": "float",
    "swap": "boolean",
    "size_average": "boolean",
    "reduce": "boolean",
    "reduction": "string"
}
signatures["torch.nn.NLLLoss"] = {
    "weight": "tensor",
    "size_average": "boolean",
    "ignore_index": "integer",
    "reduce": "boolean",
    "reduction": "string"
}
signatures["torch.nn.GaussianNLLLoss"] = {
    "full": "boolean",
    "eps": "float",
    "reduction": "string"
}
signatures["torch.nn.HuberLoss"] = {
    "delta": "float",
    "reduction": "string"
}
signatures["torch.nn.SmoothL1Loss"] = {
    "beta": "float",
    "reduction": "string"
}
signatures["torch.nn.Softplus"] = {
    "beta": "integer",
    "threshold": "integer"
}
signatures["torch.nn.PReLU"] = {
    "num_parameters": "integer",
    "init": "float"
}
signatures["torch.nn.LazyLinear"] = {
    "out_features": "integer",
    "bias": "boolean"
}
signatures["torch.nn.RNNBase.forward"] = {
    "input": "tensor",
    "hx": "tensor" # Optional[Tensor] = None
}
signatures["torch.nn.AdaptiveAvgPool1d"] = {
    "output_size": "integer" # could also be tuple[int], but integer is more common
}
signatures["torch.nn.AdaptiveAvgPool2d"] = {
    "output_size": "tuple" # Could also be an integer, but tuple is more general according to the documentation
}
signatures["torch.nn.AdaptiveAvgPool3d"] = {
    "output_size": "tuple" # Could also be an integer
}
signatures["torch.nn.AdaptiveLogSoftmaxWithLoss"] = {
    "in_features": "integer",
    "n_classes": "integer",
    "cutoffs": "list", # Could also be a tuple, but list seems more likely
    "div_value": "float",
    "head_bias": "boolean",
    # "device": "device", # Removed as requested
    # "dtype": "dtype" # Removed as requested
}
signatures["torch.nn.AdaptiveLogSoftmaxWithLoss.log_prob"] = {
    "input": "tensor"
}
signatures["torch.nn.AdaptiveLogSoftmaxWithLoss.predict"] = {
    "input": "tensor"
}
signatures["torch.nn.AdaptiveMaxPool1d"] = {
    "output_size": "integer", # Could also be tuple, but integer is more common
    "return_indices": "boolean"
}
signatures["torch.nn.AdaptiveMaxPool2d"] = {
    "output_size": "tuple", # Could also be an integer
    "return_indices": "boolean"
}
signatures["torch.nn.AdaptiveMaxPool3d"] = {
    "output_size": "tuple", # Could also be an integer or None, but tuple is most common
    "return_indices": "boolean"
}
signatures["torch.nn.AlphaDropout"] = {
    "p": "float",
    "inplace": "boolean"
}
signatures["torch.nn.AvgPool1d"] = {
    "kernel_size": "integer", # Could also be a tuple, but integer is more common
    "stride": "integer", # Could also be a tuple, but integer is more common
    "padding": "integer", # Could also be a tuple, but integer is more common
    "ceil_mode": "boolean",
    "count_include_pad": "boolean"
}
signatures["torch.nn.AvgPool2d"] = {
    "kernel_size": "tuple", # Could also be an integer
    "stride": "tuple", # Could also be an integer, default is kernel_size
    "padding": "tuple", # Could also be an integer
    "ceil_mode": "boolean",
    "count_include_pad": "boolean",
    "divisor_override": "integer" # Could be None
}
signatures["torch.nn.AvgPool3d"] = {
    "kernel_size": "tuple", # Can also be int, but tuple is more general
    "stride": "tuple", # Can also be int, but tuple is more general
    "padding": "tuple", # Can also be int, but tuple is more general
    "ceil_mode": "boolean",
    "count_include_pad": "boolean",
    "divisor_override": "integer" # Can be None, but integer is closest
}
signatures["torch.nn.BCELoss"] = {
    "weight": "tensor",
    "size_average": "boolean",
    "reduce": "boolean",
    "reduction": "string"
}
signatures["torch.nn.BCEWithLogitsLoss"] = {
    "weight": "tensor",
    "size_average": "boolean", # Deprecated, but still a boolean
    "reduce": "boolean", # Deprecated, but still a boolean
    "reduction": "string",
    "pos_weight": "tensor"
}
signatures["torch.nn.BatchNorm1d"] = {
    "num_features": "integer",
    "eps": "float",
    "momentum": "float", # Could be None, but float is more common
    "affine": "boolean",
    "track_running_stats": "boolean",
    "dtype": "dtype"
}
signatures["torch.nn.BatchNorm2d"] = {
    "num_features": "integer",
    "eps": "float",
    "momentum": "float",
    "affine": "boolean",
    "track_running_stats": "boolean",
    "dtype": "dtype" # Could be a more specific type, but dtype seems closest
}
signatures["torch.nn.BatchNorm3d"] = {
    "num_features": "integer",
    "eps": "float",
    "momentum": "float", # Could be None, but float is more common
    "affine": "boolean",
    "track_running_stats": "boolean",
    "dtype": "dtype"
}
signatures["torch.nn.Bilinear"] = {
    "in1_features": "integer",
    "in2_features": "integer",
    "out_features": "integer",
    "bias": "boolean",
    # device and dtype are omitted as per instructions
}
signatures["torch.nn.Buffer"] = {
    "data": "tensor",
    "persistent": "boolean"
}
signatures["torch.nn.CELU"] = {
    "alpha": "float",
    "inplace": "boolean"
}
signatures["torch.nn.CTCLoss"] = {
    "blank": "integer",
    "reduction": "string",
    "zero_infinity": "boolean",
    "log_probs": "tensor", # Can be (T, N, C) or (T, C)
    "targets": "tensor", # Can be (N, S) or (sum(target_lengths))
    "input_lengths": "tensor", # Can be tuple or tensor
    "target_lengths": "tensor" # Can be tuple or tensor
}
signatures["torch.nn.ChannelShuffle"] = {
    "groups": "integer"
}
signatures["torch.nn.CircularPad1d"] = {
    "padding": "tuple" # Could also be an integer, but tuple is more general according to the documentation.
}
signatures["torch.nn.CircularPad2d"] = {
    "padding": "tuple" # Could also be an integer, but tuple seems more general as per the docs.
}
signatures["torch.nn.CircularPad3d"] = {
    "padding": "tuple" # Could also be an integer, but tuple is more general
}
signatures["torch.nn.ConstantPad1d"] = {
    "padding": "tuple", # Could also be an integer, but tuple is more general
    "value": "float"
}
signatures["torch.nn.ConstantPad2d"] = {
    "padding": "tuple", # Could also be an integer, but tuple seems more common
    "value": "float" # Assuming constant value is a float
}
signatures["torch.nn.ConstantPad3d"] = {
    "padding": "tuple", # could also be integer
    "value": "float"
}
signatures["torch.nn.Container"] = {
}
signatures["torch.nn.Conv1d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "integer", # Could also be tuple, but integer is more common
    "stride": "integer", # Could also be tuple, but integer is more common
    "padding": "integer", # Could also be tuple or string, but integer is more common
    "dilation": "integer", # Could also be tuple, but integer is more common
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string",
    "dtype": "dtype"
}
signatures["torch.nn.Conv2d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # Could be integer, but tuple is more general and commonly used
    "stride": "tuple", # Could be integer, but tuple is more general
    "padding": "string", # Could be integer or tuple, but string 'valid', 'same' are also allowed
    "dilation": "tuple", # Could be integer, but tuple is more general
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string",
    "dtype": "dtype"
}
signatures["torch.nn.Conv3d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # Could also be an integer, but tuple is more general
    "stride": "tuple", # Could also be an integer, but tuple is more general
    "padding": "string", # Could also be an integer or tuple, but string is present as an option
    "dilation": "tuple", # Could also be an integer, but tuple is more general
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string",
    "dtype": "dtype"
}
signatures["torch.nn.ConvTranspose1d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "integer",  # Could also be tuple, but integer seems more common
    "stride": "integer",  # Could also be tuple, but integer seems more common
    "padding": "integer",  # Could also be tuple, but integer seems more common
    "output_padding": "integer",  # Could also be tuple, but integer seems more common
    "groups": "integer",
    "bias": "boolean",
    "dilation": "integer",  # Could also be tuple, but integer seems more common
    "padding_mode": "string",
    "dtype": "dtype"
}
signatures["torch.nn.ConvTranspose2d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # Could also be integer, but tuple seems more common for 2D
    "stride": "tuple", # Could also be integer, but tuple seems more common for 2D
    "padding": "tuple", # Could also be integer, but tuple seems more common for 2D
    "output_padding": "tuple", # Could also be integer, but tuple seems more common for 2D
    "groups": "integer",
    "bias": "boolean",
    "dilation": "tuple", # Could also be integer, but tuple seems more common for 2D
    "padding_mode": "string",
    "dtype": "dtype"
}
signatures["torch.nn.ConvTranspose3d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple",  # Could be integer as well, but tuple seems more common in examples
    "stride": "tuple",  # Could be integer as well, but tuple seems more common in examples
    "padding": "tuple",  # Could be integer as well, but tuple seems more common in examples
    "output_padding": "tuple",  # Could be integer as well, but tuple seems more common in examples
    "groups": "integer",
    "bias": "boolean",
    "dilation": "tuple",  # Could be integer as well, but tuple seems more common in examples
    "padding_mode": "string",
    "dtype": "dtype"
}
signatures["torch.nn.CosineEmbeddingLoss"] = {
    "margin": "float",
    "size_average": "boolean",
    "reduce": "boolean",
    "reduction": "string"
}
signatures["torch.nn.CosineSimilarity"] = {
    "dim": "integer",
    "eps": "float"
}
signatures["torch.nn.CrossEntropyLoss"] = {
    "weight": "tensor",
    "size_average": "boolean",
    "ignore_index": "integer",
    "reduce": "boolean",
    "reduction": "string",
    "label_smoothing": "float"
}
signatures["torch.nn.CrossMapLRN2d"] = {
    "size": "integer",
    "alpha": "float",
    "beta": "float",
    "k": "float"
}
signatures["torch.nn.DataParallel"] = {
    "module": "Module", # Should this be "object" or a more specific type?
    "device_ids": "list",
    "output_device": "integer", # Can also be torch.device, but integer seems more common
    "dim": "integer"
}
signatures["torch.nn.Dropout"] = {
    "p": "float",
    "inplace": "boolean"
}
signatures["torch.nn.Dropout1d"] = {
    "p": "float",
    "inplace": "boolean"
}
signatures["torch.nn.Dropout2d"] = {
    "p": "float",
    "inplace": "boolean"
}
signatures["torch.nn.Dropout3d"] = {
    "p": "float",
    "inplace": "boolean"
}
signatures["torch.nn.ELU"] = {
    "alpha": "float",
    "inplace": "boolean"
}
signatures["torch.nn.Embedding"] = {
    "num_embeddings": "integer",
    "embedding_dim": "integer",
    "padding_idx": "integer",
    "max_norm": "float",
    "norm_type": "float",
    "scale_grad_by_freq": "boolean",
    "sparse": "boolean",
    "_weight": "tensor",
    "_freeze": "boolean",
    "dtype": "dtype"
}
signatures["torch.nn.Embedding.from_pretrained"] = {
    "embeddings": "tensor",
    "freeze": "boolean",
    "padding_idx": "integer",
    "max_norm": "float",
    "norm_type": "float",
    "scale_grad_by_freq": "boolean",
    "sparse": "boolean"
}
signatures["torch.nn.EmbeddingBag"] = {
    "num_embeddings": "integer",
    "embedding_dim": "integer",
    "max_norm": "float",
    "norm_type": "float",
    "scale_grad_by_freq": "boolean",
    "mode": "string",
    "sparse": "boolean",
    "include_last_offset": "boolean",
    "padding_idx": "integer",
    "dtype": "dtype"
}
signatures["torch.nn.EmbeddingBag.forward"] = {
    "input": "tensor",
    "offsets": "tensor",
    "per_sample_weights": "tensor"
}
signatures["torch.nn.EmbeddingBag.from_pretrained"] = {
    "embeddings": "tensor",
    "freeze": "boolean",
    "max_norm": "float",
    "norm_type": "float",
    "scale_grad_by_freq": "boolean",
    "mode": "string",
    "sparse": "boolean",
    "include_last_offset": "boolean",
    "padding_idx": "integer"
}
signatures["torch.nn.FeatureAlphaDropout"] = {
    "p": "float",
    "inplace": "boolean"
}
signatures["torch.nn.Fold"] = {
    "output_size": "tuple",
    "kernel_size": "tuple",
    "dilation": "integer", # could also be tuple
    "padding": "integer", # could also be tuple
    "stride": "integer" # could also be tuple
}
signatures["torch.nn.FractionalMaxPool3d"] = {
    "kernel_size": "tuple", # Could also be an integer, but tuple is more general
    "output_size": "tuple", # Could also be an integer, but tuple is more general
    "output_ratio": "tuple", # Could also be a float, but tuple is more general
    "return_indices": "boolean",
    "_random_samples": "tensor"
}
signatures["torch.nn.GELU"] = {
    "approximate": "string" # could also be enum with "none" and "tanh" as options, but string seems more appropriate
}
signatures["torch.nn.GLU"] = {
    "dim": "integer", # Could also be None, but integer seems more common
    "input": "tensor"
}
signatures["torch.nn.GRU"] = {
    "input_size": "integer",
    "hidden_size": "integer",
    "num_layers": "integer",
    "bias": "boolean",
    "batch_first": "boolean",
    "dropout": "float",
    "bidirectional": "boolean",
    "dtype": "dtype",
    "input": "tensor",
    "h_0": "tensor"
}
signatures["torch.nn.GaussianNLLLoss"] = {
    "full": "boolean",
    "eps": "float",
    "reduction": "string"
}
signatures["torch.nn.Hardsigmoid"] = {
    "inplace": "boolean"
}
signatures["torch.nn.HingeEmbeddingLoss"] = {
    "margin": "float",
    "size_average": "boolean",
    "reduce": "boolean",
    "reduction": "string"
}
signatures["torch.nn.HuberLoss"] = {
    "reduction": "string",
    "delta": "float"
}
signatures["torch.nn.Identity"] = {
}
signatures["torch.nn.KLDivLoss"] = {
    "size_average": "boolean", # Deprecated, so type is not strictly defined. Assuming boolean for historical reasons.
    "reduce": "boolean", # Deprecated, so type is not strictly defined. Assuming boolean for historical reasons.
    "reduction": "string",
    "log_target": "boolean"
}
signatures["torch.nn.LPPool3d"] = {
    "norm_type": "float",  # Could also be interpreted as integer depending on the use case
    "kernel_size": "tuple", # Could also be an integer
    "stride": "tuple", # Could also be an integer
    "ceil_mode": "boolean"
}
signatures["torch.nn.LSTM"] = {
    "input_size": "integer",
    "hidden_size": "integer",
    "num_layers": "integer",
    "bias": "boolean",
    "batch_first": "boolean",
    "dropout": "float",
    "bidirectional": "boolean",
    "proj_size": "integer",
    "dtype": "dtype",
    "input": "tensor",
    "h_0": "tensor",
    "c_0": "tensor"
}
signatures["torch.nn.LazyBatchNorm1d"] = {
    "eps": "float",
    "momentum": "float",  # Could be None, but float is more common
    "affine": "boolean",
    "track_running_stats": "boolean"
}
signatures["torch.nn.LazyBatchNorm2d"] = {
    "eps": "float",
    "momentum": "float", # Could be None, but float is more common
    "affine": "boolean",
    "track_running_stats": "boolean",
    "dtype": "dtype"
}
signatures["torch.nn.LazyBatchNorm3d"] = {
    "eps": "float",
    "momentum": "float", # Could also be None, but float is more common
    "affine": "boolean",
    "track_running_stats": "boolean"
}
signatures["torch.nn.LazyConv1d"] = {
    "out_channels": "integer",
    "kernel_size": "tuple", # Could also be an integer, but tuple seems more general
    "stride": "integer", # Could also be a tuple, but integer seems more common
    "padding": "integer", # Could also be a tuple, but integer seems more common
    "dilation": "integer", # Could also be a tuple, but integer seems more common
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string",
    "dtype": "dtype"
}
signatures["torch.nn.LazyConv2d"] = {
    "out_channels": "integer",
    "kernel_size": "tuple", # could be integer too
    "stride": "tuple", # could be integer too
    "padding": "tuple", # could be integer too
    "dilation": "tuple", # could be integer too
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string",
    "dtype": "dtype"
}
signatures["torch.nn.LazyConv3d"] = {
    "out_channels": "integer",
    "kernel_size": "tuple", # Could also be integer, but tuple seems more common for kernel size
    "stride": "tuple", # Could also be integer
    "padding": "tuple", # Could also be integer
    "dilation": "tuple", # Could also be integer
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string",
    "dtype": "dtype"
}
signatures["torch.nn.LazyConvTranspose1d"] = {
    "out_channels": "integer",
    "kernel_size": "tuple", # Can be int or tuple, choosing tuple as more general
    "stride": "integer", # Can be int or tuple, choosing int as more common
    "padding": "integer", # Can be int or tuple, choosing int as more common
    "output_padding": "integer", # Can be int or tuple, choosing int as more common
    "groups": "integer",
    "bias": "boolean",
    "dilation": "integer", # Can be int or tuple, choosing int as more common
    "padding_mode": "string",
    "dtype": "dtype"
}
signatures["torch.nn.LazyConvTranspose2d"] = {
    "out_channels": "integer",
    "kernel_size": "tuple", # Could also be integer, but tuple seems more general
    "stride": "tuple", # Could also be integer
    "padding": "tuple", # Could also be integer
    "output_padding": "tuple", # Could also be integer
    "groups": "integer",
    "bias": "boolean",
    "dilation": "tuple", # Could also be integer
    "padding_mode": "string",
    "dtype": "dtype"
}
signatures["torch.nn.LazyConvTranspose3d"] = {
    "out_channels": "integer",
    "kernel_size": "tuple", # Could be integer as well, but tuple seems more common for kernel size
    "stride": "tuple", # Could be integer
    "padding": "tuple", # Could be integer
    "output_padding": "tuple", # Could be integer
    "groups": "integer",
    "bias": "boolean",
    "dilation": "tuple", # Could be integer
    "padding_mode": "string",
    "dtype": "dtype"
}
signatures["torch.nn.LazyInstanceNorm1d"] = {
    "eps": "float",
    "momentum": "float",
    "affine": "boolean",
    "track_running_stats": "boolean",
    "dtype": "dtype"
}
signatures["torch.nn.LazyInstanceNorm2d"] = {
    "eps": "float",
    "momentum": "float", # Optional[float]
    "affine": "boolean",
    "track_running_stats": "boolean",
    "dtype": "dtype"
}
signatures["torch.nn.LazyInstanceNorm3d"] = {
    "eps": "float",
    "momentum": "float",
    "affine": "boolean",
    "track_running_stats": "boolean",
    "dtype": "dtype"
}
signatures["torch.nn.LazyLinear"] = {
    "out_features": "integer",
    "bias": "boolean", # Could be UninitializedParameter but boolean is more common
    "device": "string", # Removed as per instructions
    "dtype": "dtype"
}
signatures["torch.nn.LocalResponseNorm"] = {
    "size": "integer",
    "alpha": "float",
    "beta": "float",
    "k": "float"
}
signatures["torch.nn.MaxPool1d"] = {
    "kernel_size": "integer", # Could also be tuple, but integer seems more common
    "stride": "integer", # Could also be tuple, but integer seems more common
    "padding": "integer", # Could also be tuple, but integer seems more common
    "dilation": "integer", # Could also be tuple, but integer seems more common
    "return_indices": "boolean",
    "ceil_mode": "boolean"
}
signatures["torch.nn.MaxUnpool1d"] = {
    "kernel_size": "integer", # Could be tuple but integer seems more common
    "stride": "integer", # Could be tuple but integer seems more common
    "padding": "integer", # Could be tuple but integer seems more common
    "input": "tensor",
    "indices": "tensor",
    "output_size": "tuple" # Optional, but the example uses tensor.size() which returns a tuple, so tuple seems appropriate
}
signatures["torch.nn.MaxUnpool3d"] = {
    "kernel_size": "tuple", # could be int
    "stride": "tuple", # could be int, default None
    "padding": "tuple", # could be int
    "input": "tensor",
    "indices": "tensor",
    "output_size": "tuple" # optional, could be missing
}
signatures["torch.nn.Mish"] = {
    "inplace": "boolean"
}
signatures["torch.nn.Module"] = {}
signatures["torch.nn.Module.add_module"] = {
    "name": "string",
    "module": "Module" # Should ideally be a more specific type but "Module" is the closest
}
signatures["torch.nn.Module.apply"] = {
    "fn": "list" # This should ideally be a function type. Closest is list
}
signatures["torch.nn.Module.bfloat16"] = {}
signatures["torch.nn.Module.buffers"] = {
    "recurse": "boolean"
}
signatures["torch.nn.Module.children"] = {}
signatures["torch.nn.Module.compile"] = {
    "args": "list",
    "kwargs": "list"
}
signatures["torch.nn.Module.cpu"] = {}
signatures["torch.nn.Module.cuda"] = {
    "device": "integer"
}
signatures["torch.nn.Module.double"] = {}
signatures["torch.nn.Module.eval"] = {}
signatures["torch.nn.Module.extra_repr"] = {}
signatures["torch.nn.Module.float"] = {}
signatures["torch.nn.Module.forward"] = {
    "input": "tensor" # or a tuple of tensors
}
signatures["torch.nn.Module.get_buffer"] = {
    "target": "string"
}
signatures["torch.nn.Module.get_extra_state"] = {}
signatures["torch.nn.Module.get_parameter"] = {
    "target": "string"
}
signatures["torch.nn.Module.get_submodule"] = {
    "target": "string"
}
signatures["torch.nn.Module.half"] = {}
signatures["torch.nn.Module.ipu"] = {
    "device": "integer"
}
signatures["torch.nn.Module.load_state_dict"] = {
    "state_dict": "list", # Should be dict, but list is closest
    "strict": "boolean",
    "assign": "boolean"
}
signatures["torch.nn.Module.modules"] = {}
signatures["torch.nn.Module.mtia"] = {
    "device": "integer"
}
signatures["torch.nn.Module.named_buffers"] = {
    "prefix": "string",
    "recurse": "boolean",
    "remove_duplicate": "boolean"
}
signatures["torch.nn.Module.named_children"] = {}
signatures["torch.nn.Module.named_modules"] = {
    "memo": "list", # Should be set, but list is closest
    "prefix": "string",
    "remove_duplicate": "boolean"
}
signatures["torch.nn.Module.named_parameters"] = {
    "prefix": "string",
    "recurse": "boolean",
    "remove_duplicate": "boolean"
}
signatures["torch.nn.Module.parameters"] = {
    "recurse": "boolean"
}
signatures["torch.nn.Module.register_backward_hook"] = {
    "hook": "list" # should be callable
}
signatures["torch.nn.Module.register_buffer"] = {
    "name": "string",
    "tensor": "tensor",
    "persistent": "boolean"
}
signatures["torch.nn.Module.register_forward_hook"] = {
    "hook": "list", # should be callable
    "prepend": "boolean",
    "with_kwargs": "boolean",
    "always_call": "boolean"
}
signatures["torch.nn.Module.register_forward_pre_hook"] = {
    "hook": "list", # should be callable
    "prepend": "boolean",
    "with_kwargs": "boolean"
}
signatures["torch.nn.Module.register_full_backward_hook"] = {
    "hook": "list", # should be callable
    "prepend": "boolean"
}
signatures["torch.nn.Module.register_full_backward_pre_hook"] = {
    "hook": "list", # should be callable
    "prepend": "boolean"
}
signatures["torch.nn.Module.register_load_state_dict_post_hook"] = {
    "hook": "list" # should be callable
}
signatures["torch.nn.Module.register_load_state_dict_pre_hook"] = {
    "hook": "list" # should be callable
}
signatures["torch.nn.Module.register_module"] = {
    "name": "string",
    "module": "Module" # Should ideally be a more specific type but "Module" is the closest
}
signatures["torch.nn.Module.register_parameter"] = {
    "name": "string",
    "param": "tensor" #Parameter or None, but tensor is closest
}
signatures["torch.nn.Module.register_state_dict_post_hook"] = {
    "hook": "list" # should be callable
}
signatures["torch.nn.Module.register_state_dict_pre_hook"] = {
    "hook": "list" # should be callable
}
signatures["torch.nn.Module.requires_grad_"] = {
    "requires_grad": "boolean"
}
signatures["torch.nn.Module.set_extra_state"] = {
    "state": "list" # should be dict, but list is closest
}
signatures["torch.nn.Module.set_submodule"] = {
    "target": "string",
    "module": "Module", # Should ideally be a more specific type but "Module" is the closest
    "strict": "boolean"
}
signatures["torch.nn.Module.share_memory"] = {}
signatures["torch.nn.Module.state_dict"] = {
    "prefix": "string",
    "keep_vars": "boolean"
}
signatures["torch.nn.Module.to"] = {
    "dtype": "dtype",
    "non_blocking": "boolean"
}
signatures["torch.nn.Module.to_empty"] = {
    "device": "integer",
    "recurse": "boolean"
}
signatures["torch.nn.Module.train"] = {
    "mode": "boolean"
}
signatures["torch.nn.Module.type"] = {
    "dst_type": "string" # Could be "dtype" but string is safer for "type or string"
}
signatures["torch.nn.Module.xpu"] = {
    "device": "integer"
}
signatures["torch.nn.Module.zero_grad"] = {
    "set_to_none": "boolean"
}
signatures["torch.nn.ModuleDict"] = {
    "modules": "iterable" # Could also be a dictionary of string: module. Choosing iterable since more general.
}
signatures["torch.nn.ModuleDict.clear"] = {}
signatures["torch.nn.ModuleDict.items"] = {}
signatures["torch.nn.ModuleDict.keys"] = {}
signatures["torch.nn.ModuleDict.pop"] = {
    "key": "string"
}
signatures["torch.nn.ModuleDict.update"] = {
    "modules": "iterable" # Could also be a dictionary of string: module. Choosing iterable since more general.
}
signatures["torch.nn.ModuleDict.values"] = {}
signatures["torch.nn.ModuleList"] = {
    "modules": "list" # Could also be tuple or iterable, but list seems most common
}
signatures["torch.nn.ModuleList.append"] = {
    "module": "list" # Should be nn.Module, but that's not an option
}
signatures["torch.nn.ModuleList.extend"] = {
    "modules": "list" # Should be iterable, but list seems most common
}
signatures["torch.nn.ModuleList.insert"] = {
    "index": "integer",
    "module": "list" # Should be nn.Module, but that's not an option
}
signatures["torch.nn.MultiLabelMarginLoss"] = {
    "size_average": "boolean", # Could be None, but boolean is more common
    "reduce": "boolean", # Could be None, but boolean is more common
    "reduction": "string"
}
signatures["torch.nn.MultiMarginLoss"] = {
    "p": "integer",
    "margin": "float",
    "weight": "tensor",
    "size_average": "boolean", # deprecated, but still a boolean
    "reduce": "boolean", # deprecated, but still a boolean
    "reduction": "string"
}
signatures["torch.nn.MultiheadAttention"] = {
    "embed_dim": "integer",
    "num_heads": "integer",
    "dropout": "float",
    "bias": "boolean",
    "add_bias_kv": "boolean",
    "add_zero_attn": "boolean",
    "kdim": "integer", # could be None, but integer is more common
    "vdim": "integer", # could be None, but integer is more common
    "batch_first": "boolean",
    "dtype": "dtype"
}
signatures["torch.nn.MultiheadAttention.forward"] = {
    "query": "tensor",
    "key": "tensor",
    "value": "tensor",
    "key_padding_mask": "tensor", #Optional[Tensor]
    "need_weights": "boolean",
    "attn_mask": "tensor", #Optional[Tensor]
    "average_attn_weights": "boolean",
    "is_causal": "boolean"
}
signatures["torch.nn.NLLLoss2d"] = {
    "weight": "tensor",
    "size_average": "boolean",
    "ignore_index": "integer",
    "reduce": "boolean",
    "reduction": "string"
}
signatures["torch.nn.Parameter"] = {
    "data": "tensor",
    "requires_grad": "boolean"
}
signatures["torch.nn.ParameterDict"] = {
    "parameters": "list" # Could also be a dictionary, but list of key-value pairs is also allowed
}
signatures["torch.nn.ParameterDict.clear"] = {}
signatures["torch.nn.ParameterDict.copy"] = {}
signatures["torch.nn.ParameterDict.fromkeys"] = {
    "keys": "list", # Iterable of strings
    "default": "tensor" # Should ideally be "Parameter" but using "tensor" for simplicity
}
signatures["torch.nn.ParameterDict.get"] = {
    "key": "string",
    "default": "tensor" # Should ideally be "Parameter" but using "tensor" for simplicity
}
signatures["torch.nn.ParameterDict.items"] = {}
signatures["torch.nn.ParameterDict.keys"] = {}
signatures["torch.nn.ParameterDict.pop"] = {
    "key": "string"
}
signatures["torch.nn.ParameterDict.popitem"] = {}
signatures["torch.nn.ParameterDict.setdefault"] = {
    "key": "string",
    "default": "tensor" # Should ideally be "Parameter" but using "tensor" for simplicity
}
signatures["torch.nn.ParameterDict.update"] = {
    "parameters": "list" # Could also be a dictionary, but list of key-value pairs is also allowed
}
signatures["torch.nn.ParameterDict.values"] = {}
signatures["torch.nn.ParameterList"] = {
    "values": "list" # Could also be a tuple, but list is more common
}
signatures["torch.nn.ParameterList.append"] = {
    "value": "list" # "Any" in the documentation, but assuming it's a list of parameters
}
signatures["torch.nn.ParameterList.extend"] = {
    "values": "list"
}
signatures["torch.nn.PixelUnshuffle"] = {
    "downscale_factor": "integer"
}
signatures["torch.nn.RMSNorm"] = {
    "normalized_shape": "list", # Could also be integer or torch.Size, but list seems more common
    "eps": "float",
    "elementwise_affine": "boolean",
    "dtype": "dtype"
}
signatures["torch.nn.RNN"] = {
    "input_size": "integer",
    "hidden_size": "integer",
    "num_layers": "integer",
    "nonlinearity": "string",
    "bias": "boolean",
    "batch_first": "boolean",
    "dropout": "float",
    "bidirectional": "boolean",
    "input": "tensor",
    "hx": "tensor",
    "dtype": "dtype" #Might be wrong
}
signatures["torch.nn.RNNBase"] = {
    "mode": "string",
    "input_size": "integer",
    "hidden_size": "integer",
    "num_layers": "integer",
    "bias": "boolean",
    "batch_first": "boolean",
    "dropout": "float",
    "bidirectional": "boolean",
    "proj_size": "integer",
    "dtype": "dtype"
}
signatures["torch.nn.RNNCell"] = {
    "input_size": "integer",
    "hidden_size": "integer",
    "bias": "boolean",
    "nonlinearity": "string",
    "input": "tensor",
    "hidden": "tensor",
    # device and dtype are omitted according to the instructions
}
signatures["torch.nn.RNNCellBase"] = {
    "input_size": "integer",
    "hidden_size": "integer",
    "bias": "boolean",
    "num_chunks": "integer" # could be integer or None, but integer is more common?
}
signatures["torch.nn.RReLU"] = {
    "lower": "float",
    "upper": "float",
    "inplace": "boolean"
}
signatures["torch.nn.ReflectionPad3d"] = {
    "padding": "tuple" # Could be integer as well, but tuple is more general
}
signatures["torch.nn.ReplicationPad2d"] = {
    "padding": "tuple" # Could also be an integer, but tuple seems more common
}
signatures["torch.nn.Sequential"] = {
    "args": "list" # Could also be OrderedDict, but list is more common
}
signatures["torch.nn.SmoothL1Loss"] = {
    "size_average": "boolean", # deprecated, but still a boolean
    "reduce": "boolean", # deprecated, but still a boolean
    "reduction": "string",
    "beta": "float"
}
signatures["torch.nn.SoftMarginLoss"] = {
    "size_average": "boolean", # could also be None
    "reduce": "boolean", # could also be None
    "reduction": "string"
}
signatures["torch.nn.SyncBatchNorm"] = {
    "num_features": "integer",
    "eps": "float",
    "momentum": "float", # Can be None, but float is more common
    "affine": "boolean",
    "track_running_stats": "boolean",
    "process_group": "list" # Could be "any", but process groups are often handled as lists of ranks
}
signatures["torch.nn.SyncBatchNorm.convert_sync_batchnorm"] = {
    "module": "list", #It should be nn.Module, but nn.Module is not available in the choice. List is a superclass of nn.Module.
    "process_group": "list" # Could be "any", but process groups are often handled as lists of ranks
}
signatures["torch.nn.Tanh"] = {}
signatures["torch.nn.Tanhshrink"] = {
}
signatures["torch.nn.Threshold"] = {
    "threshold": "float",
    "value": "float",
    "inplace": "boolean"
}
signatures["torch.nn.Transformer"] = {
    "d_model": "integer",
    "nhead": "integer",
    "num_encoder_layers": "integer",
    "num_decoder_layers": "integer",
    "dim_feedforward": "integer",
    "dropout": "float",
    "activation": "string", # Could be a callable, but string (relu or gelu) seems more common
    "custom_encoder": "list", # Assuming "Any" means it can be any kind of object/list
    "custom_decoder": "list", # Assuming "Any" means it can be any kind of object/list
    "layer_norm_eps": "float",
    "batch_first": "boolean",
    "norm_first": "boolean",
    "bias": "boolean"
}
signatures["torch.nn.Transformer.forward"] = {
    "src": "tensor",
    "tgt": "tensor",
    "src_mask": "tensor",
    "tgt_mask": "tensor",
    "memory_mask": "tensor",
    "src_key_padding_mask": "tensor",
    "tgt_key_padding_mask": "tensor",
    "memory_key_padding_mask": "tensor",
    "src_is_causal": "boolean", # Assuming Optional[bool] should be boolean
    "tgt_is_causal": "boolean", # Assuming Optional[bool] should be boolean
    "memory_is_causal": "boolean"
}
signatures["torch.nn.Transformer.generate_square_subsequent_mask"] = {
    "sz": "integer",
    "device": "string", # Could be device, but string seems more common/general if None
    "dtype": "dtype"
}
signatures["torch.nn.TransformerDecoder"] = {
    "decoder_layer": "list", # Should be TransformerDecoderLayer, but it's not in the options
    "num_layers": "integer",
    "norm": "list" # Should be Module, but it's not in the options
}
signatures["torch.nn.TransformerDecoder.forward"] = {
    "tgt": "tensor",
    "memory": "tensor",
    "tgt_mask": "tensor",
    "memory_mask": "tensor",
    "tgt_key_padding_mask": "tensor",
    "memory_key_padding_mask": "tensor",
    "tgt_is_causal": "boolean",
    "memory_is_causal": "boolean"
}
signatures["torch.nn.TransformerDecoderLayer"] = {
    "d_model": "integer",
    "nhead": "integer",
    "dim_feedforward": "integer",
    "dropout": "float",
    "activation": "string", # could also be a callable, but string (relu or gelu) seems more common
    "layer_norm_eps": "float",
    "batch_first": "boolean",
    "norm_first": "boolean",
    "bias": "boolean"
}
signatures["torch.nn.TransformerDecoderLayer.forward"] = {
    "tgt": "tensor",
    "memory": "tensor",
    "tgt_mask": "tensor",
    "memory_mask": "tensor",
    "tgt_key_padding_mask": "tensor",
    "memory_key_padding_mask": "tensor",
    "tgt_is_causal": "boolean",
    "memory_is_causal": "boolean"
}
signatures["torch.nn.TransformerEncoder"] = {
    "encoder_layer": "tensor", # Should be TransformerEncoderLayer, but it's a tensor-like object
    "num_layers": "integer",
    "norm": "tensor", # Should be Module, but it's a tensor-like object
    "enable_nested_tensor": "boolean",
    "mask_check": "boolean"
}
signatures["torch.nn.TransformerEncoder.forward"] = {
    "src": "tensor",
    "mask": "tensor",
    "src_key_padding_mask": "tensor",
    "is_causal": "boolean"
}
signatures["torch.nn.TransformerEncoderLayer"] = {
    "d_model": "integer",
    "nhead": "integer",
    "dim_feedforward": "integer",
    "dropout": "float",
    "activation": "string", # Could also be Callable[[Tensor], Tensor], but string is more common according to the documentation
    "layer_norm_eps": "float",
    "batch_first": "boolean",
    "norm_first": "boolean",
    "bias": "boolean"
}
signatures["torch.nn.TransformerEncoderLayer.forward"] = {
    "src": "tensor",
    "src_mask": "tensor",
    "src_key_padding_mask": "tensor",
    "is_causal": "boolean"
}
signatures["torch.nn.TripletMarginLoss"] = {
    "margin": "float",
    "p": "integer",
    "eps": "float",
    "swap": "boolean",
    "size_average": "boolean",
    "reduce": "boolean",
    "reduction": "string"
}
signatures["torch.nn.TripletMarginWithDistanceLoss"] = {
    "distance_function": "list", # Could be a Callable, but list is more general
    "margin": "float",
    "swap": "boolean",
    "reduction": "string"
}
signatures["torch.nn.Unflatten"] = {
    "dim": "integer", # Could be string for NamedTensor, but integer is more common
    "unflattened_size": "tuple" # Could be torch.Size, list or NamedShape, but tuple is more common
}
signatures["torch.nn.Unfold"] = {
    "kernel_size": "tuple", # Could also be an int, but tuple is more general
    "dilation": "tuple", # Could also be an int, but tuple is more general
    "padding": "tuple", # Could also be an int, but tuple is more general
    "stride": "tuple" # Could also be an int, but tuple is more general
}
signatures["torch.nn.UninitializedBuffer"] = {
    "size": "tuple",
    "dtype": "dtype",
    "layout": "string", # Could potentially be "layout" if that was a valid type, unsure though
    "requires_grad": "boolean"
}
signatures["torch.nn.UninitializedParameter"] = {
    "dtype": "dtype",
    "requires_grad": "boolean"
}
signatures["torch.nn.Upsample"] = {
    "size": "tuple", # Could also be integer, but tuple is more common for spatial dimensions
    "scale_factor": "tuple", # Could also be float, but tuple is more common for spatial dimensions
    "mode": "string",
    "align_corners": "boolean",
    "recompute_scale_factor": "boolean"
}
signatures["torch.nn.UpsamplingBilinear2d"] = {
    "size": "tuple", # Could also be an integer, but tuple is more common for (h, w)
    "scale_factor": "float" # Could also be a tuple, but float is more common when scaling both dimensions equally
}
signatures["torch.nn.UpsamplingNearest2d"] = {
    "size": "tuple", # Can also be an integer
    "scale_factor": "float" # Can also be a tuple
}
signatures["torch.nn.ZeroPad1d"] = {
    "padding": "tuple" # Could also be an integer, but tuple is more general and includes integer as a special case
}
signatures["torch.nn.ZeroPad2d"] = {
    "padding": "tuple" # could also be an integer, but tuple seems more specific given the description
}
signatures["torch.nn.ZeroPad3d"] = {
    "padding": "tuple" # Could also be an integer, but tuple seems more common for specifying different padding values
}
signatures["torch.nn.functional.attention"] = {
    "query": "tensor",
    "key": "tensor",
    "value": "tensor",
    "dropout_p": "float",
    "need_weights": "boolean",
    "average_attn_weights": "boolean"
}
signatures["torch.nn.attention.Iterable"] = {
    "module": "string", # unsure, but module seems like a string
    "name": "string"
}
signatures["torch.nn.attention.List"] = {
    "module": "list" # Could potentially be "list" or "tuple", but list seems more appropriate here
}
signatures["torch.nn.attention.SDPAParams"] = {
    "dropout_p": "float",
    "attn_mask": "tensor",
    "is_causal": "boolean",
    "scale": "float" # could also be boolean depending on use case
}
signatures["torch.nn.attention.SDPBackend"] = {} # This is an enum-like class, so it doesn't have arguments in the traditional sense.
signatures["torch.nn.attention.Union"] = {
    "query": "tensor",
    "key": "tensor",
    "value": "tensor",
    "key_padding_mask": "tensor",
    "need_weights": "boolean",
    "attn_mask": "tensor",
    "average_attn_weights": "boolean",
    "is_causal": "boolean",
    "bias": "tensor",
    "add_zero_attn": "boolean"
}
signatures["torch.nn.attention.can_use_efficient_attention"] = {
    "query": "tensor",
    "key": "tensor",
    "value": "tensor",
    "attn_mask": "tensor",
    "need_weights": "boolean"
}
signatures["torch.nn.attention.can_use_flash_attention"] = {
    "query": "tensor",
    "key": "tensor",
    "value": "tensor",
    "mask": "tensor",
    "dropout_p": "float",
    "training": "boolean",
    "scale": "float"
}
signatures["torch.nn.attention.contextlib.attn_bias_fwd"] = {
    "query": "tensor",
    "key": "tensor",
    "bias": "tensor",
}
signatures["torch.nn.attention.contextlib.attn_bias_bwd"] = {
    "grad_out": "tensor",
    "query": "tensor",
    "key": "tensor",
    "bias": "tensor",
}
signatures["torch.nn.attention.contextlib.fused_sdpa_cuda"] = {
    "query": "tensor",
    "key": "tensor",
    "value": "tensor",
    "mask": "tensor",
    "dropout_p": "float"
}
signatures["torch.nn.attention.contextlib.fused_sdpa_bwd"] = {
    "grad_out": "tensor",
    "query": "tensor",
    "key": "tensor",
    "value": "tensor",
    "out": "tensor",
    "mask": "tensor",
    "dropout_p": "float"
}
signatures["torch.nn.attention.contextlib.scaled_dot_product_attention"] = {
    "query": "tensor",
    "key": "tensor",
    "value": "tensor",
    "attn_mask": "tensor",
    "dropout_p": "float",
    "is_causal": "boolean"
}
signatures["torch.nn.attention.sdpa_kernel"] = {
    "query": "tensor",
    "key": "tensor",
    "value": "tensor",
    "attn_mask": "tensor",
    "dropout_p": "float",
    "is_causal": "boolean",
    "training": "boolean"
}
signatures["torch.nn.functional.scaled_dot_product_attention"] = {
    "query": "tensor",
    "key": "tensor",
    "value": "tensor",
    "attn_mask": "tensor",
    "dropout_p": "float",
    "is_causal": "boolean"
}
signatures["torch.nn.MultiheadAttention"] = {
    "embed_dim": "integer",
    "num_heads": "integer",
    "dropout": "float",
    "bias": "boolean",
    "add_bias_kv": "boolean",
    "add_zero_attn": "boolean",
    "kdim": "integer",
    "vdim": "integer",
    "batch_first": "boolean"
}
signatures["torch.nn.attention.warn"] = {
    "attn_mask": "tensor",
    "dropout": "float",
    "is_causal": "boolean"
}
signatures["torch.nn.common_types.size2_t"] = { # It is used for specifying 2D sizes, representing integers
    "arg": "tuple" # Could be a list too, but tuple is more common for sizes
}
signatures["torch.nn.common_types.size3_t"] = { # It is used for specifying 3D sizes, representing integers
    "arg": "tuple" # Could be a list too, but tuple is more common for sizes
}
signatures["torch.nn.common_types.size4_t"] = { # It is used for specifying 4D sizes, representing integers
    "arg": "tuple" # Could be a list too, but tuple is more common for sizes
}
signatures["torch.nn.common_types.size5_t"] = { # It is used for specifying 5D sizes, representing integers
    "arg": "tuple" # Could be a list too, but tuple is more common for sizes
}
signatures["torch.nn.factory_kwargs"] = {
    "module_class": "type", # Could also be "string" if it accepts the module name as a string. Type seems more appropriate
    "kwargs": "dict" #Should be a dict, no predefined type.
}
signatures["relu"] = {
    "input": "tensor",
    "inplace": "boolean"
}
signatures["leaky_relu"] = {
    "input": "tensor",
    "negative_slope": "float",
    "inplace": "boolean"
}
signatures["elu"] = {
    "input": "tensor",
    "alpha": "float",
    "inplace": "boolean"
}
signatures["gelu"] = {
    "input": "tensor",
    "approximate": "string"
}
signatures["selu"] = {
    "input": "tensor",
    "inplace": "boolean"
}
signatures["celu"] = {
    "input": "tensor",
    "alpha": "float",
    "inplace": "boolean"
}
signatures["silu"] = {
    "input": "tensor",
    "inplace": "boolean"
}
signatures["sigmoid"] = {
    "input": "tensor"
}
signatures["tanh"] = {
    "input": "tensor"
}
signatures["softmax"] = {
    "input": "tensor",
    "dim": "integer",
    "dtype": "dtype"
}
signatures["log_softmax"] = {
    "input": "tensor",
    "dim": "integer",
    "dtype": "dtype"
}
signatures["adaptive_avg_pool1d"] = {
    "input": "tensor",
    "output_size": "integer" # could be list or tuple
}
signatures["adaptive_avg_pool2d"] = {
    "input": "tensor",
    "output_size": "tuple" # could be list or integer
}
signatures["adaptive_avg_pool3d"] = {
    "input": "tensor",
    "output_size": "tuple" # could be list or integer
}
signatures["avg_pool1d"] = {
    "input": "tensor",
    "kernel_size": "integer", # could be tuple
    "stride": "integer", # could be tuple
    "padding": "integer", # could be tuple
    "ceil_mode": "boolean",
    "count_include_pad": "boolean"
}
signatures["avg_pool2d"] = {
    "input": "tensor",
    "kernel_size": "tuple", # could be integer
    "stride": "tuple", # could be integer
    "padding": "tuple", # could be integer
    "ceil_mode": "boolean",
    "count_include_pad": "boolean",
    "divisor_override": "integer"
}
signatures["avg_pool3d"] = {
    "input": "tensor",
    "kernel_size": "tuple", # could be integer
    "stride": "tuple", # could be integer
    "padding": "tuple", # could be integer
    "ceil_mode": "boolean",
    "count_include_pad": "boolean"
}
signatures["max_pool1d"] = {
    "input": "tensor",
    "kernel_size": "integer", # could be tuple
    "stride": "integer", # could be tuple
    "padding": "integer", # could be tuple
    "dilation": "integer",
    "ceil_mode": "boolean",
    "return_indices": "boolean"
}
signatures["max_pool2d"] = {
    "input": "tensor",
    "kernel_size": "tuple", # could be integer
    "stride": "tuple", # could be integer
    "padding": "tuple", # could be integer
    "dilation": "tuple", # could be integer
    "ceil_mode": "boolean",
    "return_indices": "boolean"
}
signatures["max_pool3d"] = {
    "input": "tensor",
    "kernel_size": "tuple", # could be integer
    "stride": "tuple", # could be integer
    "padding": "tuple", # could be integer
    "dilation": "tuple", # could be integer
    "ceil_mode": "boolean",
    "return_indices": "boolean"
}
signatures["fractional_max_pool2d"] = {
    "input": "tensor",
    "kernel_size": "tuple", # could be integer
    "output_size": "tuple", # could be integer
    "output_ratio": "tuple",
    "return_indices": "boolean",
    "generator": "tensor"
}
signatures["fractional_max_pool3d"] = {
    "input": "tensor",
    "kernel_size": "tuple", # could be integer
    "output_size": "tuple", # could be integer
    "output_ratio": "tuple",
    "return_indices": "boolean",
    "generator": "tensor"
}
signatures["lp_pool1d"] = {
    "input": "tensor",
    "norm_type": "float",
    "kernel_size": "integer",
    "stride": "integer", # could be tuple
    "padding": "integer",
    "ceil_mode": "boolean"
}
signatures["lp_pool2d"] = {
    "input": "tensor",
    "norm_type": "float",
    "kernel_size": "tuple", # could be integer
    "stride": "tuple", # could be integer
    "padding": "tuple",
    "ceil_mode": "boolean"
}
signatures["adaptive_max_pool1d"] = {
    "input": "tensor",
    "output_size": "integer", # could be tuple or list
    "return_indices": "boolean"
}
signatures["adaptive_max_pool2d"] = {
    "input": "tensor",
    "output_size": "tuple", # could be integer or list
    "return_indices": "boolean"
}
signatures["adaptive_max_pool3d"] = {
    "input": "tensor",
    "output_size": "tuple", # could be integer or list
    "return_indices": "boolean"
}
signatures["unfold"] = {
    "input": "tensor",
    "kernel_size": "tuple",
    "dilation": "tuple",
    "padding": "tuple",
    "stride": "tuple"
}
signatures["fold"] = {
    "input": "tensor",
    "output_size": "tuple",
    "kernel_size": "tuple",
    "dilation": "tuple",
    "padding": "tuple",
    "stride": "tuple"
}
signatures["embedding"] = {
    "input": "tensor",
    "weight": "tensor",
    "padding_idx": "integer",
    "max_norm": "float",
    "norm_type": "float",
    "scale_grad_by_freq": "boolean",
    "sparse": "boolean"
}
signatures["embedding_bag"] = {
    "input": "tensor",
    "weight": "tensor",
    "offsets": "tensor",
    "max_norm": "float",
    "norm_type": "float",
    "scale_grad_by_freq": "boolean",
    "mode": "string",
    "sparse": "boolean",
    "per_sample_weights": "tensor",
    "include_last_offset": "boolean"
}
signatures["linear"] = {
    "input": "tensor",
    "weight": "tensor",
    "bias": "tensor"
}
signatures["dropout"] = {
    "input": "tensor",
    "p": "float",
    "training": "boolean",
    "inplace": "boolean"
}
signatures["dropout2d"] = {
    "input": "tensor",
    "p": "float",
    "training": "boolean",
    "inplace": "boolean"
}
signatures["dropout3d"] = {
    "input": "tensor",
    "p": "float",
    "training": "boolean",
    "inplace": "boolean"
}
signatures["alpha_dropout"] = {
    "input": "tensor",
    "p": "float",
    "training": "boolean"
}
signatures["feature_alpha_dropout"] = {
    "input": "tensor",
    "p": "float",
    "training": "boolean"
}
signatures["conv1d"] = {
    "input": "tensor",
    "weight": "tensor",
    "bias": "tensor",
    "stride": "tuple", # could be integer
    "padding": "tuple", # could be string or integer
    "dilation": "tuple", # could be integer
    "groups": "integer"
}
signatures["conv2d"] = {
    "input": "tensor",
    "weight": "tensor",
    "bias": "tensor",
    "stride": "tuple", # could be integer
    "padding": "tuple", # could be string or integer
    "dilation": "tuple", # could be integer
    "groups": "integer"
}
signatures["conv3d"] = {
    "input": "tensor",
    "weight": "tensor",
    "bias": "tensor",
    "stride": "tuple", # could be integer
    "padding": "tuple", # could be string or integer
    "dilation": "tuple", # could be integer
    "groups": "integer"
}
signatures["conv_transpose1d"] = {
    "input": "tensor",
    "weight": "tensor",
    "bias": "tensor",
    "stride": "tuple", # could be integer
    "padding": "tuple", # could be string or integer
    "output_padding": "tuple", # could be integer
    "groups": "integer",
    "dilation": "tuple" # could be integer
}
signatures["conv_transpose2d"] = {
    "input": "tensor",
    "weight": "tensor",
    "bias": "tensor",
    "stride": "tuple", # could be integer
    "padding": "tuple", # could be string or integer
    "output_padding": "tuple", # could be integer
    "groups": "integer",
    "dilation": "tuple" # could be integer
}
signatures["conv_transpose3d"] = {
    "input": "tensor",
    "weight": "tensor",
    "bias": "tensor",
    "stride": "tuple", # could be integer
    "padding": "tuple", # could be string or integer
    "output_padding": "tuple", # could be integer
    "groups": "integer",
    "dilation": "tuple" # could be integer
}
signatures["batch_norm"] = {
    "input": "tensor",
    "running_mean": "tensor",
    "running_var": "tensor",
    "weight": "tensor",
    "bias": "tensor",
    "training": "boolean",
    "momentum": "float",
    "eps": "float"
}
signatures["instance_norm"] = {
    "input": "tensor",
    "running_mean": "tensor",
    "running_var": "tensor",
    "weight": "tensor",
    "bias": "tensor",
    "use_input_stats": "boolean",
    "momentum": "float",
    "eps": "float"
}
signatures["layer_norm"] = {
    "input": "tensor",
    "normalized_shape": "tuple", # can be list
    "weight": "tensor",
    "bias": "tensor",
    "eps": "float"
}
signatures["local_response_norm"] = {
    "input": "tensor",
    "size": "integer",
    "alpha": "float",
    "beta": "float",
    "k": "float"
}
signatures["rnn_tanh"] = {
    "input": "tensor",
    "hx": "tensor",
    "weight_ih": "tensor",
    "weight_hh": "tensor",
    "bias_ih": "tensor",
    "bias_hh": "tensor"
}
signatures["rnn_relu"] = {
    "input": "tensor",
    "hx": "tensor",
    "weight_ih": "tensor",
    "weight_hh": "tensor",
    "bias_ih": "tensor",
    "bias_hh": "tensor"
}
signatures["lstm_cell"] = {
    "input": "tensor",
    "hx": "tuple",
    "weight_ih": "tensor",
    "weight_hh": "tensor",
    "bias_ih": "tensor",
    "bias_hh": "tensor"
}
signatures["gru_cell"] = {
    "input": "tensor",
    "hx": "tensor",
    "weight_ih": "tensor",
    "weight_hh": "tensor",
    "bias_ih": "tensor",
    "bias_hh": "tensor"
}
signatures["binary_cross_entropy"] = {
    "input": "tensor",
    "target": "tensor",
    "weight": "tensor",
    "size_average": "boolean",
    "reduce": "boolean",
    "reduction": "string"
}
signatures["binary_cross_entropy_with_logits"] = {
    "input": "tensor",
    "target": "tensor",
    "weight": "tensor",
    "pos_weight": "tensor",
    "reduction": "string"
}
signatures["nll_loss"] = {
    "input": "tensor",
    "target": "tensor",
    "weight": "tensor",
    "ignore_index": "integer",
    "reduction": "string"
}
signatures["cross_entropy"] = {
    "input": "tensor",
    "target": "tensor",
    "weight": "tensor",
    "ignore_index": "integer",
    "reduction": "string",
    "label_smoothing": "float"
}
signatures["cosine_embedding_loss"] = {
    "input1": "tensor",
    "input2": "tensor",
    "target": "tensor",
    "margin": "float",
    "size_average": "boolean",
    "reduce": "boolean",
    "reduction": "string"
}
signatures["hinge_embedding_loss"] = {
    "input": "tensor",
    "target": "tensor",
    "margin": "float",
    "size_average": "boolean",
    "reduce": "boolean",
    "reduction": "string"
}
signatures["kl_div"] = {
    "input": "tensor",
    "target": "tensor",
    "size_average": "boolean",
    "reduce": "boolean",
    "reduction": "string",
    "log_target": "boolean"
}
signatures["margin_ranking_loss"] = {
    "input1": "tensor",
    "input2": "tensor",
    "target": "tensor",
    "margin": "float",
    "size_average": "boolean",
    "reduce": "boolean",
    "reduction": "string"
}
signatures["mse_loss"] = {
    "input": "tensor",
    "target": "tensor",
    "size_average": "boolean",
    "reduce": "boolean",
    "reduction": "string"
}
signatures["poisson_nll_loss"] = {
    "input": "tensor",
    "target": "tensor",
    "log_input": "boolean",
    "full": "boolean",
    "size_average": "boolean",
    "eps": "float",
    "reduce": "boolean",
    "reduction": "string"
}
signatures["smooth_l1_loss"] = {
    "input": "tensor",
    "target": "tensor",
    "size_average": "boolean",
    "reduce": "boolean",
    "reduction": "string",
    "beta": "float"
}
signatures["l1_loss"] = {
    "input": "tensor",
    "target": "tensor",
    "size_average": "boolean",
    "reduce": "boolean",
    "reduction": "string"
}
signatures["triplet_margin_loss"] = {
    "anchor": "tensor",
    "positive": "tensor",
    "negative": "tensor",
    "margin": "float",
    "p": "float",
    "eps": "float",
    "swap": "boolean",
    "size_average": "boolean",
    "reduce": "boolean",
    "reduction": "string"
}
signatures["triplet_margin_with_distance_loss"] = {
    "anchor": "tensor",
    "positive": "tensor",
    "negative": "tensor",
    "distance_function": "string", # callable could be another option
    "margin": "float",
    "eps": "float",
    "swap": "boolean",
    "reduction": "string"
}
signatures["margin_ranking_loss"] = {
    "input1": "tensor",
    "input2": "tensor",
    "target": "tensor",
    "margin": "float",
    "size_average": "boolean",
    "reduce": "boolean",
    "reduction": "string"
}
signatures["ctc_loss"] = {
    "log_probs": "tensor",
    "targets": "tensor",
    "input_lengths": "tensor",
    "target_lengths": "tensor",
    "blank": "integer",
    "reduction": "string",
    "zero_infinity": "boolean"
}
signatures["pad"] = {
    "input": "tensor",
    "pad": "tuple", # could be a list
    "mode": "string",
    "value": "float"
}
signatures["interpolate"] = {
    "input": "tensor",
    "size": "tuple", # can be list or integer
    "scale_factor": "float", # can be tuple or list
    "mode": "string",
    "align_corners": "boolean",
    "recompute_scale_factor": "boolean"
}
signatures["grid_sample"] = {
    "input": "tensor",
    "grid": "tensor",
    "mode": "string",
    "padding_mode": "string",
    "align_corners": "boolean"
}
signatures["conv_tbc"] = {
    "input": "tensor",
    "weight": "tensor",
    "bias": "tensor",
    "pad": "integer"
}
signatures["pdist"] = {
    "input": "tensor",
    "p": "float"
}
signatures["cosine_similarity"] = {
    "x1": "tensor",
    "x2": "tensor",
    "dim": "integer",
    "eps": "float"
}
signatures["dropout"] = {
    "input": "tensor",
    "p": "float",
    "training": "boolean",
    "inplace": "boolean"
}
signatures["one_hot"] = {
    "tensor": "tensor",
    "num_classes": "integer"
}
signatures["torch.nn.functional.cross_entropy"] = {
    "input": "tensor",
    "target": "tensor",
    "weight": "tensor",
    "ignore_index": "integer",
    "reduction": "string",
    "label_smoothing": "float"
}
signatures["torch.nn.functional.embedding"] = {
    "input": "tensor",
    "weight": "tensor",
    "padding_idx": "integer",
    "max_norm": "float",
    "norm_type": "float",
    "scale_grad_by_freq": "boolean",
    "sparse": "boolean"
}
signatures["torch.nn.functional.interpolate"] = {
    "input": "tensor",
    "size": "tuple", # Can be int or tuple, choosing tuple as more general
    "scale_factor": "tuple", # Can be float or tuple, choosing tuple as more general
    "mode": "string",
    "align_corners": "boolean",
    "recompute_scale_factor": "boolean",
    "antialias": "boolean"
}
signatures["torch.nn.functional.log_softmax"] = {
    "input": "tensor",
    "dim": "integer", # Could be 'integer' or 'None' commonly
    "dtype": "dtype" # optional argument, but included as the most common signature is requested
}
signatures["torch.nn.functional.softmax"] = {
    "input": "tensor",
    "dim": "integer",
    "dtype": "dtype"
}
signatures["torch.nn.functional.softmin"] = {
    "input": "tensor",
    "dim": "integer",
    "dtype": "dtype"
}
signatures["torch.nn.grad"] = {
    "outputs": "tensor_list",
    "inputs": "tensor_list",
    "grad_outputs": "tensor_list",
    "retain_graph": "boolean",
    "create_graph": "boolean",
    "only_inputs": "boolean",
    "allow_unused": "boolean" # could be a list of tensors?
}
signatures["torch.nn.init.calculate_gain"] = {
    "nonlinearity": "string",
    "param": "float"
}
signatures["torch.nn.init.uniform_"] = {
    "tensor": "tensor",
    "a": "float",
    "b": "float"
}
signatures["torch.nn.init.normal_"] = {
    "tensor": "tensor",
    "mean": "float",
    "std": "float"
}
signatures["torch.nn.init.trunc_normal_"] = {
    "tensor": "tensor",
    "mean": "float",
    "std": "float",
    "a": "float",
    "b": "float"
}
signatures["torch.nn.init.constant_"] = {
    "tensor": "tensor",
    "val": "float" # Could potentially be an integer as well, but float seems more common
}
signatures["torch.nn.init.ones_"] = {
    "tensor": "tensor"
}
signatures["torch.nn.init.zeros_"] = {
    "tensor": "tensor"
}
signatures["torch.nn.init.eye_"] = {
    "tensor": "tensor"
}
signatures["torch.nn.init.dirac_"] = {
    "tensor": "tensor",
    "offset": "integer"
}
signatures["torch.nn.init.xavier_uniform_"] = {
    "tensor": "tensor",
    "gain": "float"
}
signatures["torch.nn.init.xavier_normal_"] = {
    "tensor": "tensor",
    "gain": "float"
}
signatures["torch.nn.init.kaiming_uniform_"] = {
    "tensor": "tensor",
    "a": "float",
    "mode": "string",
    "nonlinearity": "string"
}
signatures["torch.nn.init.kaiming_normal_"] = {
    "tensor": "tensor",
    "a": "float",
    "mode": "string",
    "nonlinearity": "string"
}
signatures["torch.nn.init.orthogonal_"] = {
    "tensor": "tensor",
    "gain": "float"
}
signatures["torch.nn.init.sparse_"] = {
    "tensor": "tensor",
    "sparsity": "float",
    "std": "float"
}
 # Assuming signatures is already initialized
signatures["torch.nn.intrinsic.quantized.ConvReLU2d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # could also be int, but tuple is more common for kernel size
    "stride": "tuple", # could also be int
    "padding": "tuple", # could also be int
    "dilation": "tuple", # could also be int
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string",
    "qconfig": "string" # qconfig is an object, but representing it as a string for simplicity
}
signatures["torch.nn.intrinsic.quantized.ConvBn2d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # could also be int, but tuple is more common for kernel size
    "stride": "tuple", # could also be int
    "padding": "tuple", # could also be int
    "dilation": "tuple", # could also be int
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.intrinsic.quantized.ConvBnReLU2d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # could also be int, but tuple is more common for kernel size
    "stride": "tuple", # could also be int
    "padding": "tuple", # could also be int
    "dilation": "tuple", # could also be int
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.intrinsic.ConvReLU1d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # could also be int, but tuple is more common for kernel size
    "stride": "tuple", # could also be int
    "padding": "tuple", # could also be int
    "dilation": "tuple", # could also be int
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.intrinsic.ConvReLU2d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # could also be int, but tuple is more common for kernel size
    "stride": "tuple", # could also be int
    "padding": "tuple", # could also be int
    "dilation": "tuple", # could also be int
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.intrinsic.ConvReLU3d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # could also be int, but tuple is more common for kernel size
    "stride": "tuple", # could also be int
    "padding": "tuple", # could also be int
    "dilation": "tuple", # could also be int
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.intrinsic.LinearReLU"] = {
    "in_features": "integer",
    "out_features": "integer",
    "bias": "boolean"
}
signatures["torch.nn.intrinsic.BNReLU2d"] = {
    "num_features": "integer",
    "eps": "float",
    "momentum": "float",
    "affine": "boolean",
    "track_running_stats": "boolean"
}
signatures["torch.nn.intrinsic.BNReLU3d"] = {
    "num_features": "integer",
    "eps": "float",
    "momentum": "float",
    "affine": "boolean",
    "track_running_stats": "boolean"
}
signatures["torch.nn.intrinsic.ConvBn1d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "integer", # could also be tuple
    "stride": "integer", # could also be tuple
    "padding": "integer", # could also be tuple
    "dilation": "integer", # could also be tuple
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.intrinsic.ConvBn2d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # could also be integer
    "stride": "tuple", # could also be integer
    "padding": "tuple", # could also be integer
    "dilation": "tuple", # could also be integer
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.intrinsic.ConvBn3d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # could also be an integer
    "stride": "tuple", # could also be an integer
    "padding": "tuple", # could also be an integer
    "dilation": "tuple", # could also be an integer
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.intrinsic.ConvBnReLU1d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "integer", # or tuple
    "stride": "integer", # or tuple
    "padding": "integer", # or tuple
    "dilation": "integer", # or tuple
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.intrinsic.ConvBnReLU2d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # Could be integer as well, but tuple is more common
    "stride": "tuple", # Could be integer as well, but tuple is more common
    "padding": "tuple", # Could be integer as well, but tuple is more common
    "dilation": "tuple", # Could be integer as well, but tuple is more common
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.intrinsic.ConvBnReLU3d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple",  # Can also be an integer, choosing the most common
    "stride": "tuple",  # Can also be an integer, choosing the most common
    "padding": "tuple",  # Can also be an integer, choosing the most common
    "dilation": "tuple",  # Can also be an integer, choosing the most common
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.intrinsic.ConvReLU1d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "integer", # could be tuple as well
    "stride": "integer", # could be tuple as well
    "padding": "integer", # could be tuple as well
    "dilation": "integer", # could be tuple as well
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.intrinsic.ConvReLU2d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # could also be an integer, tuple is more common
    "stride": "tuple", # could also be an integer, tuple is more common
    "padding": "tuple", # could also be an integer, tuple is more common
    "dilation": "tuple", # could also be an integer, tuple is more common
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.intrinsic.ConvReLU3d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # Could also be integer but tuple is more common
    "stride": "tuple", # Could also be integer but tuple is more common
    "padding": "tuple", # Could also be integer but tuple is more common
    "dilation": "tuple", # Could also be integer but tuple is more common
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.intrinsic.LinearBn1d"] = {
    "in_features": "integer",
    "out_features": "integer",
    "bias": "boolean"
}
signatures["torch.nn.intrinsic.LinearReLU"] = {
    "in_features": "integer",
    "out_features": "integer",
    "bias": "boolean"
}
signatures["torch.nn.intrinsic.modules.ConvReLU1d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # Could be integer as well, but tuple is probably more common
    "stride": "tuple", # Could be integer as well, but tuple is probably more common
    "padding": "tuple", # Could be integer as well, but tuple is probably more common
    "dilation": "tuple", # Could be integer as well, but tuple is probably more common
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.intrinsic.modules.ConvReLU2d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # Could be integer as well, but tuple is probably more common
    "stride": "tuple", # Could be integer as well, but tuple is probably more common
    "padding": "tuple", # Could be integer as well, but tuple is probably more common
    "dilation": "tuple", # Could be integer as well, but tuple is probably more common
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.intrinsic.modules.ConvReLU3d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # Could be integer as well, but tuple is probably more common
    "stride": "tuple", # Could be integer as well, but tuple is probably more common
    "padding": "tuple", # Could be integer as well, but tuple is probably more common
    "dilation": "tuple", # Could be integer as well, but tuple is probably more common
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.intrinsic.modules.LinearReLU"] = {
    "in_features": "integer",
    "out_features": "integer",
    "bias": "boolean"
}
signatures["torch.nn.intrinsic.modules.BNReLU2d"] = {
    "num_features": "integer",
    "eps": "float",
    "momentum": "float",
    "affine": "boolean",
    "track_running_stats": "boolean"
}
signatures["torch.nn.intrinsic.modules.BNReLU3d"] = {
    "num_features": "integer",
    "eps": "float",
    "momentum": "float",
    "affine": "boolean"
}
signatures["torch.nn.intrinsic.modules.ConvBn1d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "integer", # Could also be tuple
    "stride": "integer", # Could also be tuple
    "padding": "integer", # Could also be tuple
    "dilation": "integer", # Could also be tuple
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.intrinsic.modules.ConvBn2d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # could also be an integer
    "stride": "tuple", # could also be an integer
    "padding": "tuple", # could also be an integer
    "dilation": "tuple", # could also be an integer
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.intrinsic.modules.ConvBn3d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # Could be integer or tuple, assuming tuple is more common
    "stride": "tuple", # Could be integer or tuple, assuming tuple is more common
    "padding": "tuple", # Could be integer or tuple, assuming tuple is more common
    "dilation": "tuple", # Could be integer or tuple, assuming tuple is more common
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.intrinsic.modules.ConvBnReLU1d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # could be integer as well but tuple is more common
    "stride": "tuple", # could be integer as well but tuple is more common
    "padding": "tuple", # could be integer as well but tuple is more common
    "dilation": "tuple", # could be integer as well but tuple is more common
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.intrinsic.modules.ConvBnReLU2d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # Could also be an integer, but tuple is more common
    "stride": "tuple", # Could also be an integer, but tuple is more common
    "padding": "tuple", # Could also be an integer, but tuple is more common
    "dilation": "tuple", # Could also be an integer, but tuple is more common
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.intrinsic.modules.ConvBnReLU3d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple",  # Could be integer as well, but tuple is more common
    "stride": "tuple", # Could be integer as well, but tuple is more common
    "padding": "tuple", # Could be integer as well, but tuple is more common
    "dilation": "tuple", # Could be integer as well, but tuple is more common
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.intrinsic.modules.ConvReLU1d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "integer",
    "stride": "integer",
    "padding": "integer",
    "dilation": "integer",
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.intrinsic.modules.ConvReLU2d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # Could be integer as well, but tuple is more common
    "stride": "tuple", # Could be integer as well, but tuple is more common
    "padding": "tuple", # Could be integer as well, but tuple is more common
    "dilation": "tuple", # Could be integer as well, but tuple is more common
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.intrinsic.modules.ConvReLU3d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # Could be integer too, but tuple is more common
    "stride": "tuple", # Could be integer too, but tuple is more common
    "padding": "tuple", # Could be integer too, but tuple is more common
    "dilation": "tuple", # Could be integer too, but tuple is more common
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.intrinsic.modules.LinearBn1d"] = {
    "in_features": "integer",
    "out_features": "integer",
    "bias": "boolean"
}
signatures["torch.nn.intrinsic.modules.LinearReLU"] = {
    "in_features": "integer",
    "out_features": "integer",
    "bias": "boolean"
}
signatures["torch.nn.intrinsic.modules.fused.ConvReLU1d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "integer",
    "stride": "integer",
    "padding": "integer",
    "dilation": "integer",
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.intrinsic.modules.fused.ConvReLU2d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "integer",
    "stride": "integer",
    "padding": "integer",
    "dilation": "integer",
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.intrinsic.modules.fused.ConvReLU3d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "integer",
    "stride": "integer",
    "padding": "integer",
    "dilation": "integer",
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.intrinsic.modules.fused.LinearReLU"] = {
    "in_features": "integer",
    "out_features": "integer",
    "bias": "boolean"
}
signatures["torch.nn.intrinsic.modules.fused.BatchNormReLU1d"] = {
    "num_features": "integer",
    "eps": "float",
    "momentum": "float",
    "affine": "boolean",
    "track_running_stats": "boolean"
}
signatures["torch.nn.intrinsic.modules.fused.BatchNormReLU2d"] = {
    "num_features": "integer",
    "eps": "float",
    "momentum": "float",
    "affine": "boolean",
    "track_running_stats": "boolean"
}
signatures["torch.nn.intrinsic.modules.fused.BatchNormReLU3d"] = {
    "num_features": "integer",
    "eps": "float",
    "momentum": "float",
    "affine": "boolean",
    "track_running_stats": "boolean"
}
signatures["torch.nn.intrinsic.qat.ConvBn1d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "integer",
    "stride": "integer",
    "padding": "integer",
    "dilation": "integer",
    "groups": "integer",
    "padding_mode": "string",
    "bias": "boolean",
    "qconfig": "string" # Could also be a QConfig object, but "string" is more general.
}
signatures["torch.nn.intrinsic.qat.ConvBn2d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "integer",
    "stride": "integer",
    "padding": "integer",
    "dilation": "integer",
    "groups": "integer",
    "padding_mode": "string",
    "bias": "boolean",
    "qconfig": "string" # Could also be a QConfig object, but "string" is more general.
}
signatures["torch.nn.intrinsic.qat.ConvBn3d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "integer",
    "stride": "integer",
    "padding": "integer",
    "dilation": "integer",
    "groups": "integer",
    "padding_mode": "string",
    "bias": "boolean",
    "qconfig": "string" # Could also be a QConfig object, but "string" is more general.
}
signatures["torch.nn.intrinsic.qat.ConvReLU1d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "integer",
    "stride": "integer",
    "padding": "integer",
    "dilation": "integer",
    "groups": "integer",
    "padding_mode": "string",
    "bias": "boolean",
    "qconfig": "string" # Could also be a QConfig object, but "string" is more general.
}
signatures["torch.nn.intrinsic.qat.ConvReLU2d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "integer",
    "stride": "integer",
    "padding": "integer",
    "dilation": "integer",
    "groups": "integer",
    "padding_mode": "string",
    "bias": "boolean",
    "qconfig": "string" # Could also be a QConfig object, but "string" is more general.
}
signatures["torch.nn.intrinsic.qat.ConvReLU3d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "integer",
    "stride": "integer",
    "padding": "integer",
    "dilation": "integer",
    "groups": "integer",
    "padding_mode": "string",
    "bias": "boolean",
    "qconfig": "string" # Could also be a QConfig object, but "string" is more general.
}
signatures["torch.nn.intrinsic.qat.LinearReLU"] = {
    "in_features": "integer",
    "out_features": "integer",
    "bias": "boolean",
    "qconfig": "string" # Could also be a QConfig object, but "string" is more general.
}
signatures["torch.nn.intrinsic.qat.BNReLU2d"] = {
    "num_features": "integer",
    "eps": "float",
    "momentum": "float",
    "qconfig": "string" # Could also be a QConfig object, but "string" is more general.
}
signatures["torch.nn.intrinsic.qat.BNReLU3d"] = {
    "num_features": "integer",
    "eps": "float",
    "momentum": "float",
    "qconfig": "string" # Could also be a QConfig object, but "string" is more general.
}
signatures["torch.nn.intrinsic.qat.ConvBn1d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "integer", # or tuple
    "stride": "integer", # or tuple
    "padding": "integer", # or tuple
    "dilation": "integer", # or tuple
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string",
    "eps": "float",
    "momentum": "float",
    "qconfig": "string" # I am not sure about qconfig, but string seems most appropriate
}
signatures["torch.nn.intrinsic.qat.ConvBn2d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # could be integer too
    "stride": "tuple", # could be integer too
    "padding": "tuple", # could be integer too
    "dilation": "tuple", # could be integer too
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string",
    "eps": "float",
    "momentum": "float",
    "qconfig": "object" # unsure, but it's an object for specifying quantization configurations
}
signatures["torch.nn.intrinsic.qat.ConvBn3d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # could also be integer
    "stride": "tuple", # could also be integer
    "padding": "tuple", # could also be integer
    "dilation": "tuple", # could also be integer
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.intrinsic.qat.ConvBnReLU1d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "integer", # or tuple
    "stride": "integer", # or tuple
    "padding": "integer", # or tuple
    "dilation": "integer", # or tuple
    "groups": "integer",
    "padding_mode": "string",
    "bias": "boolean",
    "qconfig": "object" # Should be qconfig_cls or QConfig, but can't specify 'object'
}
signatures["torch.nn.intrinsic.qat.ConvBnReLU2d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # Can also be an integer
    "stride": "tuple", # Can also be an integer
    "padding": "tuple", # Can also be an integer
    "dilation": "tuple", # Can also be an integer
    "groups": "integer",
    "padding_mode": "string",
    "bias": "boolean",
    "qconfig": "string" # I'm not sure about this one, but it seems like it should be a string representing a quantization configuration
}
signatures["torch.nn.intrinsic.qat.ConvBnReLU3d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # Could also be an integer
    "stride": "tuple", # Could also be an integer
    "padding": "tuple", # Could also be an integer
    "dilation": "tuple", # Could also be an integer
    "groups": "integer",
    "padding_mode": "string",
    "bias": "boolean",
    "qconfig": "string" # Should ideally be `object` or a custom type.
}
signatures["torch.nn.intrinsic.qat.ConvReLU1d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "integer", # could also be tuple, but integer is more common
    "stride": "integer", # could also be tuple, but integer is more common
    "padding": "integer", # could also be tuple, but integer is more common
    "dilation": "integer", # could also be tuple, but integer is more common
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.intrinsic.qat.ConvReLU2d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # could also be integer
    "stride": "tuple", # could also be integer
    "padding": "tuple", # could also be integer
    "dilation": "tuple", # could also be integer
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string",
    "qconfig": "string" #This should ideally be a quantization config object, but string is the closest we can get
}
signatures["torch.nn.intrinsic.qat.ConvReLU3d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # Could also be an integer, but tuple is more common
    "stride": "tuple", # Could also be an integer, but tuple is more common
    "padding": "tuple", # Could also be an integer, but tuple is more common
    "dilation": "tuple", # Could also be an integer, but tuple is more common
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string",
    "qconfig": "string" # This is actually a QConfig object, but representing it as a string for simplicity since it is the most general string format.
}
signatures["torch.nn.intrinsic.qat.LinearBn1d"] = {
    "in_features": "integer",
    "out_features": "integer",
    "bias": "boolean",
    "eps": "float",
    "momentum": "float",
    "qconfig": "object" # should ideally be qconfig but object seems to be the next best thing
}
signatures["torch.nn.intrinsic.qat.LinearReLU"] = {
    "in_features": "integer",
    "out_features": "integer",
    "bias": "boolean"
}
signatures["torch.nn.intrinsic.qat.freeze_bn_stats"] = {
    "module": "tensor" # It's a module, but can also be a tensor, choosing tensor as it seems more common
}
signatures["torch.nn.intrinsic.qat.modules.ConvBn1d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # could also be integer
    "stride": "tuple", # could also be integer
    "padding": "tuple", # could also be integer
    "dilation": "tuple", # could also be integer
    "groups": "integer",
    "padding_mode": "string",
    "bias": "boolean"
}
signatures["torch.nn.intrinsic.qat.modules.ConvBn2d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # could also be integer
    "stride": "tuple", # could also be integer
    "padding": "tuple", # could also be integer
    "dilation": "tuple", # could also be integer
    "groups": "integer",
    "padding_mode": "string",
    "bias": "boolean"
}
signatures["torch.nn.intrinsic.qat.modules.ConvBn3d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # could also be integer
    "stride": "tuple", # could also be integer
    "padding": "tuple", # could also be integer
    "dilation": "tuple", # could also be integer
    "groups": "integer",
    "padding_mode": "string",
    "bias": "boolean"
}
signatures["torch.nn.intrinsic.qat.modules.ConvReLU1d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # could also be integer
    "stride": "tuple", # could also be integer
    "padding": "tuple", # could also be integer
    "dilation": "tuple", # could also be integer
    "groups": "integer",
    "padding_mode": "string",
    "bias": "boolean"
}
signatures["torch.nn.intrinsic.qat.modules.ConvReLU2d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # could also be integer
    "stride": "tuple", # could also be integer
    "padding": "tuple", # could also be integer
    "dilation": "tuple", # could also be integer
    "groups": "integer",
    "padding_mode": "string",
    "bias": "boolean"
}
signatures["torch.nn.intrinsic.qat.modules.ConvReLU3d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # could also be integer
    "stride": "tuple", # could also be integer
    "padding": "tuple", # could also be integer
    "dilation": "tuple", # could also be integer
    "groups": "integer",
    "padding_mode": "string",
    "bias": "boolean"
}
signatures["torch.nn.intrinsic.qat.modules.LinearReLU"] = {
    "in_features": "integer",
    "out_features": "integer",
    "bias": "boolean"
}
signatures["torch.nn.intrinsic.qat.modules.BNReLU2d"] = {
    "num_features": "integer"
}
signatures["torch.nn.intrinsic.qat.modules.BNReLU3d"] = {
    "num_features": "integer"
}
signatures["torch.nn.intrinsic.qat.modules.ConvBnReLU1d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # could also be integer
    "stride": "tuple", # could also be integer
    "padding": "tuple", # could also be integer
    "dilation": "tuple", # could also be integer
    "groups": "integer",
    "padding_mode": "string",
    "bias": "boolean"
}
signatures["torch.nn.intrinsic.qat.modules.ConvBnReLU2d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # could also be integer
    "stride": "tuple", # could also be integer
    "padding": "tuple", # could also be integer
    "dilation": "tuple", # could also be integer
    "groups": "integer",
    "padding_mode": "string",
    "bias": "boolean"
}
signatures["torch.nn.intrinsic.qat.modules.ConvBnReLU3d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # could also be integer
    "stride": "tuple", # could also be integer
    "padding": "tuple", # could also be integer
    "dilation": "tuple", # could also be integer
    "groups": "integer",
    "padding_mode": "string",
    "bias": "boolean"
}
signatures["torch.nn.intrinsic.qat.modules.ConvBn1d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "integer", # Could also be tuple
    "stride": "integer", # Could also be tuple
    "padding": "integer", # Could also be tuple
    "dilation": "integer", # Could also be tuple
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.intrinsic.qat.modules.ConvBn2d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "integer", # Could also be tuple, but integer seems more common
    "stride": "integer", # Could also be tuple, but integer seems more common
    "padding": "integer", # Could also be tuple, but integer seems more common
    "dilation": "integer", # Could also be tuple, but integer seems more common
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.intrinsic.qat.modules.ConvBn3d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # could also be an integer but tuple is more common
    "stride": "tuple", # could also be an integer but tuple is more common
    "padding": "tuple", # could also be an integer but tuple is more common
    "dilation": "tuple", # could also be an integer but tuple is more common
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.intrinsic.qat.modules.ConvBnReLU1d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "integer", # Can also be a tuple
    "stride": "integer", # Can also be a tuple
    "padding": "integer", # Can also be a tuple
    "dilation": "integer", # Can also be a tuple
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.intrinsic.qat.modules.ConvBnReLU2d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple",  # Could be integer too, but tuple is more common
    "stride": "tuple", # Could be integer too, but tuple is more common
    "padding": "tuple", # Could be integer too, but tuple is more common
    "dilation": "tuple", # Could be integer too, but tuple is more common
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string",
    "qconfig": "string" # Ideally this should be QConfig type, but closest we have is string
}
signatures["torch.nn.intrinsic.qat.modules.ConvBnReLU3d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # Could also be integer
    "stride": "tuple", # Could also be integer
    "padding": "tuple", # Could also be integer
    "dilation": "tuple", # Could also be integer
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string",
    "qconfig": "string" # Should be a QConfig type but can only be string for this exercise
}
signatures["torch.nn.intrinsic.qat.modules.ConvReLU1d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "integer", # Could also be tuple, but integer seems more common
    "stride": "integer", # Could also be tuple, but integer seems more common
    "padding": "integer", # Could also be tuple, but integer seems more common
    "dilation": "integer", # Could also be tuple, but integer seems more common
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.intrinsic.qat.modules.ConvReLU2d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # can be integer too, but tuple is more common
    "stride": "tuple", # can be integer too, but tuple is more common
    "padding": "tuple", # can be integer too, but tuple is more common
    "dilation": "tuple", # can be integer too, but tuple is more common
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.intrinsic.qat.modules.ConvReLU3d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # could be integer as well but tuple is more common
    "stride": "tuple", # could be integer as well but tuple is more common
    "padding": "tuple", # could be integer as well but tuple is more common
    "dilation": "tuple", # could be integer as well but tuple is more common
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.intrinsic.qat.modules.LinearBn1d"] = {
    "in_features": "integer",
    "out_features": "integer",
    "bias": "boolean",
    "eps": "float",
    "momentum": "float",
    "qconfig": "object" # Could also be a string, but object is a more general option
}
signatures["torch.nn.intrinsic.qat.modules.LinearReLU"] = {
    "in_features": "integer",
    "out_features": "integer",
    "bias": "boolean"
}
signatures["torch.nn.intrinsic.qat.modules.conv_fused"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # could be integer as well, but tuple is more common
    "stride": "tuple", # could be integer as well, but tuple is more common
    "padding": "tuple", # could be integer as well, but tuple is more common
    "dilation": "tuple", # could be integer as well, but tuple is more common
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.intrinsic.qat.modules.freeze_bn_stats"] = {
}
signatures["torch.nn.intrinsic.qat.modules.linear_fused"] = {
    "in_features": "integer",
    "out_features": "integer",
    "bias": "boolean"
}
signatures["torch.nn.intrinsic.qat.modules.linear_relu"] = {
    "in_features": "integer",
    "out_features": "integer",
    "bias": "boolean"
}
signatures["torch.nn.intrinsic.qat.modules.update_bn_stats"] = {
    "bn_module": "tensor", # This might need to be a module type but sticking to provided types
    "input": "tensor"
}
signatures["torch.nn.intrinsic.qat.update_bn_stats"] = {
    "module": "tensor", # or maybe it's a module type, unclear
    "X": "tensor"
}
signatures["torch.nn.intrinsic.quantized.ConvReLU2d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # Could also be integer
    "stride": "tuple", # Could also be integer
    "padding": "tuple", # Could also be integer
    "dilation": "tuple", # Could also be integer
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.intrinsic.quantized.ConvBn2d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # Could also be integer
    "stride": "tuple", # Could also be integer
    "padding": "tuple", # Could also be integer
    "dilation": "tuple", # Could also be integer
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.intrinsic.quantized.ConvBnReLU2d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # Could also be integer
    "stride": "tuple", # Could also be integer
    "padding": "tuple", # Could also be integer
    "dilation": "tuple", # Could also be integer
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.intrinsic.quantized.LinearReLU"] = {
    "in_features": "integer",
    "out_features": "integer",
    "bias": "boolean"
}
signatures["torch.nn.intrinsic.quantized.BNReLU2d"] = {
    "num_features": "integer",
    "eps": "float",
    "momentum": "float",
    "qconfig": "string" # Could also be an object, but string seems like a reasonable representation of config
}
signatures["torch.nn.intrinsic.quantized.BNReLU3d"] = {
    "num_features": "integer",
    "eps": "float",
    "momentum": "float",
    "qconfig": "object" # This should ideally be qconfig or QConfig but it is hard to tell without documentation
}
signatures["torch.nn.intrinsic.quantized.ConvReLU1d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "integer", # or tuple
    "stride": "integer", # or tuple
    "padding": "integer", # or tuple
    "dilation": "integer", # or tuple
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.intrinsic.quantized.ConvReLU2d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # Could also be integer, but tuple is more common
    "stride": "tuple", # Could also be integer, but tuple is more common
    "padding": "tuple", # Could also be integer, but tuple is more common
    "dilation": "tuple", # Could also be integer, but tuple is more common
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.intrinsic.quantized.ConvReLU3d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # Can also be an integer
    "stride": "tuple", # Can also be an integer
    "padding": "tuple", # Can also be an integer
    "dilation": "tuple", # Can also be an integer
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string",
    "qconfig": "string" # I am not sure about this type, maybe there's a more precise one
}
signatures["torch.nn.intrinsic.quantized.LinearReLU"] = {
    "in_features": "integer",
    "out_features": "integer",
    "bias": "boolean"
    # I believe bias can also be a Tensor, but boolean is more common.
}
signatures["torch.nn.intrinsic.quantized.dynamic.LinearReLU"] = {
    "in_features": "integer",
    "out_features": "integer",
    "bias": "boolean"
}
signatures["torch.nn.intrinsic.quantized.dynamic.Linear"] = {
    "in_features": "integer",
    "out_features": "integer",
    "bias": "boolean"
}
signatures["torch.nn.intrinsic.quantized.dynamic.ConvReLU2d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # Could be integer, but usually it is tuple
    "stride": "tuple", # Could be integer, but usually it is tuple
    "padding": "tuple", # Could be integer, but usually it is tuple
    "dilation": "tuple", # Could be integer, but usually it is tuple
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.intrinsic.quantized.dynamic.Conv2d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # Could be integer, but usually it is tuple
    "stride": "tuple", # Could be integer, but usually it is tuple
    "padding": "tuple", # Could be integer, but usually it is tuple
    "dilation": "tuple", # Could be integer, but usually it is tuple
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.intrinsic.quantized.dynamic.LinearReLU"] = {
    "in_features": "integer",
    "out_features": "integer",
    "bias": "boolean" # Could also be None, which should ideally be a special type in the future.
}
signatures["torch.nn.intrinsic.quantized.dynamic.modules.ConvReLU2d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "integer", # Could also be tuple
    "stride": "integer", # Could also be tuple
    "padding": "integer", # Could also be tuple
    "dilation": "integer", # Could also be tuple
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.intrinsic.quantized.dynamic.modules.LinearReLU"] = {
    "in_features": "integer",
    "out_features": "integer",
    "bias": "boolean"
}
signatures["torch.nn.intrinsic.quantized.dynamic.modules.BNReLU2d"] = {
    "num_features": "integer",
    "eps": "float",
    "momentum": "float",
    "affine": "boolean"
}
signatures["torch.nn.intrinsic.quantized.dynamic.modules.ConvBnReLU2d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "integer", # Could also be tuple
    "stride": "integer", # Could also be tuple
    "padding": "integer", # Could also be tuple
    "dilation": "integer", # Could also be tuple
    "groups": "integer",
    "bias": "boolean",
    "eps": "float",
    "momentum": "float",
    "padding_mode": "string"
}
signatures["torch.nn.intrinsic.quantized.dynamic.modules.LinearReLU"] = {
    "in_features": "integer",
    "out_features": "integer",
    "bias": "boolean"
}
signatures["torch.nn.intrinsic.quantized.dynamic.modules.linear_relu"] = {
    "in_features": "integer",
    "out_features": "integer",
    "bias": "boolean" # Could also be None, but boolean seems more common for specifying whether to use bias
}
signatures["torch.nn.intrinsic.quantized.modules.ConvReLU2d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "integer", # Could also be tuple
    "stride": "integer", # Could also be tuple
    "padding": "integer", # Could also be tuple
    "dilation": "integer", # Could also be tuple
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.intrinsic.quantized.modules.ConvBn2d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "integer", # Could also be tuple
    "stride": "integer", # Could also be tuple
    "padding": "integer", # Could also be tuple
    "dilation": "integer", # Could also be tuple
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.intrinsic.quantized.modules.ConvBnReLU2d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "integer", # Could also be tuple
    "stride": "integer", # Could also be tuple
    "padding": "integer", # Could also be tuple
    "dilation": "integer", # Could also be tuple
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.intrinsic.quantized.modules.LinearReLU"] = {
    "in_features": "integer",
    "out_features": "integer",
    "bias": "boolean"
}
signatures["torch.nn.intrinsic.quantized.modules.BNReLU2d"] = {
    "num_features": "integer",
    "eps": "float",
    "momentum": "float",
    "qconfig": "object" # Assuming qconfig is a custom object type. Could be string?
}
signatures["torch.nn.intrinsic.quantized.modules.BNReLU3d"] = {
    "num_features": "integer",
    "eps": "float",
    "momentum": "float",
    "affine": "boolean",
    "track_running_stats": "boolean",
    "quantization_scheme": "dtype" # could also be 'string', unsure about the exact type here
}
signatures["torch.nn.intrinsic.quantized.modules.ConvReLU1d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "integer", # Could also be tuple, but integer is more common
    "stride": "integer", # Could also be tuple, but integer is more common
    "padding": "integer", # Could also be tuple, but integer is more common
    "dilation": "integer", # Could also be tuple, but integer is more common
    "groups": "integer",
    "padding_mode": "string",
    "qconfig": "dtype" # Assuming qconfig is a specific dtype like QConfig
}
signatures["torch.nn.intrinsic.quantized.modules.ConvReLU2d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # Could also be integer
    "stride": "tuple", # Could also be integer
    "padding": "tuple", # Could also be integer
    "dilation": "tuple", # Could also be integer
    "groups": "integer",
    "padding_mode": "string",
    "bias": "boolean"
}
signatures["torch.nn.intrinsic.quantized.modules.ConvReLU3d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # could be integer as well
    "stride": "tuple", # could be integer as well
    "padding": "tuple", # could be integer as well
    "dilation": "tuple", # could be integer as well
    "groups": "integer",
    "padding_mode": "string",
    "qconfig": "string" # Actually a QConfig, but representing it as a string
}
signatures["torch.nn.intrinsic.quantized.modules.LinearReLU"] = {
    "in_features": "integer",
    "out_features": "integer",
    "bias": "boolean" # Could also be None, but boolean is more common
}
signatures["torch.nn.intrinsic.quantized.modules.bn_relu"] = {
    "num_features": "integer",
    "eps": "float",
    "momentum": "float",
    "qconfig": "object" # Assuming qconfig is a class, but "object" is the closest match
}
signatures["torch.nn.intrinsic.quantized.modules.conv_relu"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # Could be integer as well, but tuple is more common
    "stride": "tuple", # Could be integer as well, but tuple is more common
    "padding": "tuple", # Could be integer as well, but tuple is more common
    "dilation": "tuple", # Could be integer as well, but tuple is more common
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.intrinsic.quantized.modules.linear_relu"] = {
    "in_features": "integer",
    "out_features": "integer",
    "bias": "boolean"
}
signatures["torch.nn.Conv2d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # Can be an integer or a tuple
    "stride": "tuple", # Can be an integer or a tuple
    "padding": "tuple", # Can be an integer or a tuple
    "dilation": "tuple", # Can be an integer or a tuple
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.Linear"] = {
    "in_features": "integer",
    "out_features": "integer",
    "bias": "boolean"
}
signatures["torch.nn.ReLU"] = {
    "inplace": "boolean"
}
signatures["torch.nn.MaxPool2d"] = {
    "kernel_size": "tuple", # Can be an integer or a tuple
    "stride": "tuple", # Can be an integer or a tuple
    "padding": "tuple", # Can be an integer or a tuple
    "dilation": "tuple", # Can be an integer or a tuple
    "return_indices": "boolean",
    "ceil_mode": "boolean"
}
signatures["torch.nn.BatchNorm2d"] = {
    "num_features": "integer",
    "eps": "float",
    "momentum": "float",
    "affine": "boolean",
    "track_running_stats": "boolean"
}
signatures["torch.nn.Dropout"] = {
    "p": "float",
    "inplace": "boolean"
}
signatures["torch.nn.CrossEntropyLoss"] = {
    "weight": "tensor",
    "ignore_index": "integer",
    "reduction": "string",
    "label_smoothing": "float"
}
signatures["torch.nn.Module.forward"] = {
    "input": "tensor" # Assuming 'input' is the most common argument name for forward
}
signatures["torch.nn.LSTM"] = {
    "input_size": "integer",
    "hidden_size": "integer",
    "num_layers": "integer",
    "bias": "boolean",
    "batch_first": "boolean",
    "dropout": "float",
    "bidirectional": "boolean"
}
signatures["torch.nn.Embedding"] = {
    "num_embeddings": "integer",
    "embedding_dim": "integer",
    "padding_idx": "integer",
    "max_norm": "float",
    "norm_type": "float",
    "scale_grad_by_freq": "boolean",
    "sparse": "boolean",
    "_weight": "tensor" #Should it be a tensor?
}
signatures["torch.nn.Sequential"] = {
    "*args": "list" # a list of modules
}
signatures["torch.nn.AdaptiveAvgPool2d"] = {
    "output_size": "tuple" #Can be int or tuple
}
signatures["torch.nn.MSELoss"] = {
    "reduction": "string"
}
signatures["torch.nn.modules.AdaptiveAvgPool1d"] = {
    "output_size": "integer" # Can also be a tuple of ints, but integer is more common
}
signatures["torch.nn.modules.AdaptiveAvgPool2d"] = {
    "output_size": "tuple" # Can be integer or tuple, but tuple is more common
}
signatures["torch.nn.modules.AdaptiveAvgPool3d"] = {
    "output_size": "tuple" # Could also be an integer or tuple of integers
}
signatures["torch.nn.modules.AdaptiveLogSoftmaxWithLoss"] = {
    "in_features": "integer",
    "n_classes": "integer",
    "cutoffs": "list",
    "div_value": "float",
    "head_bias": "boolean",
    "adaptive_float16": "boolean"
}
signatures["torch.nn.modules.AdaptiveMaxPool1d"] = {
    "output_size": "integer" # Could also be a tuple, but integer is more common
}
signatures["torch.nn.modules.AdaptiveMaxPool2d"] = {
    "output_size": "tuple" # Could also be an integer, but tuple is more common
}
signatures["torch.nn.modules.AdaptiveMaxPool3d"] = {
    "output_size": "tuple" # Can also be integer or tuple of integers, but tuple is more general
}
signatures["torch.nn.modules.AlphaDropout"] = {
    "p": "float",
    "inplace": "boolean"
}
signatures["torch.nn.modules.AvgPool1d"] = {
    "kernel_size": "integer", # Can also be tuple
    "stride": "integer", # Can also be None or tuple
    "padding": "integer", # Can also be tuple
    "ceil_mode": "boolean",
    "count_include_pad": "boolean"
}
signatures["torch.nn.modules.AvgPool2d"] = {
    "kernel_size": "integer", # Could also be tuple
    "stride": "integer", # Could also be tuple
    "padding": "integer", # Could also be tuple
    "ceil_mode": "boolean",
    "count_include_pad": "boolean",
    "divisor_override": "integer" # Could be None
}
signatures["torch.nn.modules.AvgPool3d"] = {
    "kernel_size": "tuple", # Could also be an integer
    "stride": "tuple", # Could also be an integer
    "padding": "tuple", # Could also be an integer
    "ceil_mode": "boolean",
    "count_include_pad": "boolean",
    "divisor_override": "integer" # Could be None
}
signatures["torch.nn.modules.BCELoss"] = {
    "weight": "tensor",
    "size_average": "boolean",
    "reduce": "boolean",
    "reduction": "string"
}
signatures["torch.nn.modules.BCEWithLogitsLoss"] = {
    "weight": "tensor",
    "size_average": "boolean",
    "reduce": "boolean",
    "reduction": "string",
    "pos_weight": "tensor"
}
signatures["torch.nn.modules.BatchNorm1d"] = {
    "num_features": "integer",
    "eps": "float",
    "momentum": "float",
    "affine": "boolean",
    "track_running_stats": "boolean"
}
signatures["torch.nn.modules.BatchNorm2d"] = {
    "num_features": "integer",
    "eps": "float",
    "momentum": "float",
    "affine": "boolean",
    "track_running_stats": "boolean"
}
signatures["torch.nn.modules.BatchNorm3d"] = {
    "num_features": "integer",
    "eps": "float",
    "momentum": "float",
    "affine": "boolean",
    "track_running_stats": "boolean"
}
signatures["torch.nn.modules.Bilinear"] = {
    "in1_features": "integer",
    "in2_features": "integer",
    "out_features": "integer",
    "bias": "boolean"
}
signatures["torch.nn.modules.CELU"] = {
    "alpha": "float",
    "inplace": "boolean"
}
signatures["torch.nn.modules.CTCLoss"] = {
    "blank": "integer",
    "reduction": "string",
    "zero_infinity": "boolean"
}
signatures["torch.nn.functional.ctc_loss"] = {
    "log_probs": "tensor",
    "targets": "tensor",
    "input_lengths": "tensor",
    "target_lengths": "tensor",
    "blank": "integer",
    "reduction": "string",
    "zero_infinity": "boolean"
}
signatures["torch.nn.modules.ChannelShuffle"] = {
    "groups": "integer"
}
signatures["torch.nn.modules.CircularPad1d"] = {
    "padding": "integer" # could also be tuple
}
signatures["torch.nn.modules.CircularPad2d"] = {
    "padding": "tuple" # Could also be integer or list/tuple of integers
}
signatures["torch.nn.modules.CircularPad3d"] = {
    "padding": "tuple" # Could also be an integer or list, but tuple seems most common
}
signatures["torch.nn.modules.ConstantPad1d"] = {
    "padding": "tuple", # Could also be an integer, but tuple is probably more common/explicit
    "value": "float"
}
signatures["torch.nn.modules.ConstantPad2d"] = {
    "padding": "integer", # Could also be tuple/list
    "value": "float"
}
signatures["torch.nn.modules.ConstantPad3d"] = {
    "padding": "tuple", # Could also be integer or list/tuple of integers
    "value": "float"
}
signatures["torch.nn.modules.Container"] = {}
signatures["torch.nn.modules.Conv1d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "integer", # Could be tuple as well, but integer is more common
    "stride": "integer", # Could be tuple as well, but integer is more common
    "padding": "integer", # Could be string or tuple as well, but integer is more common
    "dilation": "integer", # Could be tuple as well, but integer is more common
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.modules.Conv2d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # Could also be an integer, but tuple is more common for specifying height and width
    "stride": "tuple", # Could also be an integer, but tuple is more common for specifying height and width
    "padding": "tuple", # Could also be an integer, but tuple is more common for specifying height and width
    "dilation": "tuple", # Could also be an integer, but tuple is more common for specifying height and width
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.modules.Conv3d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # Can be tuple or integer, defaulting to tuple
    "stride": "tuple", # Can be tuple or integer, defaulting to tuple
    "padding": "tuple", # Can be tuple or string or integer, defaulting to tuple
    "dilation": "tuple", # Can be tuple or integer, defaulting to tuple
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.modules.ConvTranspose1d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # Can also be integer
    "stride": "tuple", # Can also be integer
    "padding": "tuple", # Can also be integer
    "output_padding": "tuple", # Can also be integer
    "groups": "integer",
    "bias": "boolean",
    "dilation": "tuple", # Can also be integer
    "padding_mode": "string"
}
signatures["torch.nn.modules.ConvTranspose2d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # Could also be integer, but tuple is more common
    "stride": "tuple", # Could also be integer, but tuple is more common
    "padding": "tuple", # Could also be integer, but tuple is more common
    "output_padding": "tuple", # Could also be integer, but tuple is more common
    "groups": "integer",
    "bias": "boolean",
    "dilation": "tuple", # Could also be integer, but tuple is more common
    "padding_mode": "string"
}
signatures["torch.nn.modules.ConvTranspose3d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # could also be integer
    "stride": "tuple", # could also be integer
    "padding": "tuple", # could also be integer
    "output_padding": "tuple", # could also be integer
    "dilation": "tuple", # could also be integer
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.modules.CosineEmbeddingLoss"] = {
    "margin": "float",
    "reduction": "string"
}
signatures["torch.nn.modules.CosineSimilarity"] = {
    "dim": "integer",
    "eps": "float"
}
signatures["torch.nn.modules.CrossEntropyLoss"] = {
    "weight": "tensor",
    "ignore_index": "integer",
    "reduction": "string",
    "label_smoothing": "float"
}
signatures["torch.nn.modules.CrossMapLRN2d"] = {
    "size": "integer",
    "alpha": "float",
    "beta": "float",
    "k": "float"
}
signatures["torch.nn.modules.Dropout"] = {
    "p": "float",
    "inplace": "boolean"
}
signatures["torch.nn.modules.Dropout1d"] = {
    "p": "float",
    "inplace": "boolean"
}
signatures["torch.nn.modules.Dropout2d"] = {
    "p": "float",
    "inplace": "boolean"
}
signatures["torch.nn.modules.Dropout3d"] = {
    "p": "float",
    "inplace": "boolean"
}
signatures["torch.nn.modules.ELU"] = {
    "alpha": "float",
    "inplace": "boolean"
}
signatures["torch.nn.modules.Embedding"] = {
    "num_embeddings": "integer",
    "embedding_dim": "integer",
    "padding_idx": "integer",
    "max_norm": "float",
    "norm_type": "float",
    "scale_grad_by_freq": "boolean",
    "sparse": "boolean",
    "_weight": "tensor" #guessing this is the weight tensor. Could also be a string if its the name of the tensor to be passed
}
signatures["torch.nn.modules.EmbeddingBag"] = {
    "num_embeddings": "integer",
    "embedding_dim": "integer",
    "max_norm": "float",
    "norm_type": "float",
    "scale_grad_by_freq": "boolean",
    "mode": "string",
    "sparse": "boolean",
    "_weight": "tensor", # should this be tensor? it is not exposed to users directly
    "include_last_offset": "boolean",
    "padding_idx": "integer"
}
signatures["torch.nn.modules.FeatureAlphaDropout"] = {
    "p": "float" # Could also be None, but float is more common
}
signatures["torch.nn.modules.Flatten"] = {
    "start_dim": "integer",
    "end_dim": "integer"
}
signatures["torch.nn.modules.Fold"] = {
    "output_size": "tuple",
    "kernel_size": "tuple",
    "dilation": "tuple",
    "padding": "tuple",
    "stride": "tuple"
}
signatures["torch.nn.modules.FractionalMaxPool2d"] = {
    "kernel_size": "tuple",
    "output_size": "tuple",
    "output_ratio": "tuple",
    "return_indices": "boolean"
}
signatures["torch.nn.modules.FractionalMaxPool3d"] = {
    "kernel_size": "tuple",
    "output_size": "tuple",
    "output_ratio": "tuple",
    "return_indices": "boolean"
}
signatures["torch.nn.modules.GELU"] = {
    "approximate": "string" # Could be boolean as well
}
signatures["torch.nn.modules.GLU"] = {
    "dim": "integer"
}
signatures["torch.nn.modules.GRU"] = {
    "input_size": "integer",
    "hidden_size": "integer",
    "num_layers": "integer",
    "bias": "boolean",
    "batch_first": "boolean",
    "dropout": "float",
    "bidirectional": "boolean"
}
signatures["torch.nn.modules.GRUCell"] = {
    "input": "tensor",
    "hx": "tensor"
}
signatures["torch.nn.modules.GaussianNLLLoss"] = {
    "full": "boolean",
    "eps": "float",
    "reduction": "string"
}
signatures["torch.nn.modules.GroupNorm"] = {
    "num_groups": "integer",
    "num_channels": "integer",
    "eps": "float",
    "affine": "boolean"
}
signatures["torch.nn.modules.Hardshrink"] = {
    "lambd": "float"
}
signatures["torch.nn.modules.Hardsigmoid"] = {
    "inplace": "boolean"
}
signatures["torch.nn.modules.Hardswish"] = {
    "input": "tensor"
}
signatures["torch.nn.modules.Hardtanh"] = {
    "min_val": "float", # Could be integer as well
    "max_val": "float" # Could be integer as well
}
signatures["torch.nn.modules.HingeEmbeddingLoss"] = {
    "margin": "float",
    "reduction": "string"
}
signatures["torch.nn.modules.HuberLoss"] = {
    "delta": "float",
    "reduction": "string"
}
signatures["torch.nn.modules.Identity"] = {}
signatures["torch.nn.modules.InstanceNorm1d"] = {
    "num_features": "integer",
    "eps": "float",
    "momentum": "float",
    "affine": "boolean",
    "track_running_stats": "boolean"
}
signatures["torch.nn.modules.InstanceNorm2d"] = {
    "num_features": "integer",
    "eps": "float",
    "momentum": "float",
    "affine": "boolean",
    "track_running_stats": "boolean"
}
signatures["torch.nn.modules.InstanceNorm3d"] = {
    "num_features": "integer",
    "eps": "float",
    "momentum": "float",
    "affine": "boolean",
    "track_running_stats": "boolean"
}
signatures["torch.nn.modules.KLDivLoss"] = {
    "reduction": "string",
    "log_target": "boolean" # Could be argued to be None as well
}
signatures["torch.nn.modules.L1Loss"] = {
    "size_average": "boolean", # Deprecated, should ideally not be in the signature
    "reduce": "boolean", # Deprecated, should ideally not be in the signature
    "reduction": "string"
}
signatures["torch.nn.modules.LPPool1d"] = {
    "norm_type": "integer", # could be float as well
    "kernel_size": "integer",
    "stride": "integer",
    "ceil_mode": "boolean"
}
signatures["torch.nn.modules.LPPool2d"] = {
    "norm_type": "float", # Can also be int
    "kernel_size": "integer", # Can also be tuple
    "stride": "integer", # Can also be tuple
    "ceil_mode": "boolean"
}
signatures["torch.nn.modules.LPPool3d"] = {
    "norm_type": "integer", # Could be float, but usually integer
    "kernel_size": "tuple",
    "stride": "tuple",
    "ceil_mode": "boolean"
}
signatures["torch.nn.modules.LSTM"] = {
    "input_size": "integer",
    "hidden_size": "integer",
    "num_layers": "integer",
    "bias": "boolean",
    "batch_first": "boolean",
    "dropout": "float",
    "bidirectional": "boolean",
    "proj_size": "integer"
}
signatures["torch.nn.modules.LSTMCell"] = {
    "input": "tensor",
    "hx": "tuple", # Could be "tensor_list" if always a list, but seems like a tuple is common
    "cx": "tensor",
}
signatures["torch.nn.modules.LayerNorm"] = {
    "normalized_shape": "list", # Could also be tuple, but list is more common
    "eps": "float",
    "elementwise_affine": "boolean"
}
signatures["torch.nn.modules.LazyBatchNorm1d"] = {
    "num_features": "integer", # Could also be None which represents lazy evaluation of features
    "eps": "float",
    "momentum": "float",
    "affine": "boolean",
    "track_running_stats": "boolean"
}
signatures["torch.nn.modules.LazyBatchNorm2d"] = {
    "num_features": "integer", # This is inferred from the input, so it could also be considered None or Any
    "eps": "float",
    "momentum": "float",
    "affine": "boolean",
    "track_running_stats": "boolean"
}
signatures["torch.nn.modules.LazyBatchNorm3d"] = {
    "num_features": "integer", # Ideally would be 'None' initially, but integer is the closest type
    "eps": "float",
    "momentum": "float",
    "affine": "boolean",
    "track_running_stats": "boolean"
}
signatures["torch.nn.modules.LazyConv1d"] = {
    "in_channels": "integer", # Could potentially also be None
    "out_channels": "integer",
    "kernel_size": "integer", # or tuple
    "stride": "integer", # or tuple
    "padding": "integer", # or string or tuple
    "dilation": "integer", # or tuple
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.modules.LazyConv2d"] = {
    "in_channels": "integer", # Could be None, representing lazy initialization. But integer is closest.
    "out_channels": "integer",
    "kernel_size": "integer", # Could be tuple, but integer is more common as a single number for both dimensions
    "stride": "integer", # Could be tuple, but integer is more common
    "padding": "integer", # Could be string or tuple, but integer is more common
    "dilation": "integer", # Could be tuple, but integer is more common
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.modules.LazyConv3d"] = {
    "in_channels": "integer", # Could be None, but integer is more common when it's initialized
    "out_channels": "integer",
    "kernel_size": "tuple", # Can also be integer, but tuple is more common
    "stride": "tuple", # Can also be integer, but tuple is more common
    "padding": "tuple", # Can also be integer or string, but tuple is more common
    "dilation": "tuple", # Can also be integer, but tuple is more common
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.modules.LazyConvTranspose1d"] = {
    "in_channels": "integer", # Could be None, defaulting to inferring from input, but integer is the closest type
    "out_channels": "integer",
    "kernel_size": "integer", # Could be tuple, but integer is more common
    "stride": "integer", # Could be tuple, but integer is more common
    "padding": "integer", # Could be tuple, but integer is more common
    "output_padding": "integer", # Could be tuple, but integer is more common
    "groups": "integer",
    "bias": "boolean",
    "dilation": "integer" # Could be tuple, but integer is more common
}
signatures["torch.nn.modules.LazyConvTranspose2d"] = {
    "in_channels": "integer", # Could be None, but assuming most common case is integer
    "out_channels": "integer",
    "kernel_size": "tuple", # Can be int or tuple, choosing tuple for consistency with other Conv APIs
    "stride": "tuple", # Can be int or tuple, choosing tuple for consistency with other Conv APIs
    "padding": "tuple", # Can be int or tuple, choosing tuple for consistency with other Conv APIs
    "output_padding": "tuple", # Can be int or tuple, choosing tuple for consistency with other Conv APIs
    "groups": "integer",
    "dilation": "tuple", # Can be int or tuple, choosing tuple for consistency with other Conv APIs
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.modules.LazyConvTranspose3d"] = {
    "in_channels": "integer", # Could be None, but integer is most common after initialization
    "out_channels": "integer",
    "kernel_size": "tuple", # Can also accept int, but tuple is more general.
    "stride": "tuple", # Can also accept int, but tuple is more general.
    "padding": "tuple", # Can also accept int, but tuple is more general.
    "output_padding": "tuple", # Can also accept int, but tuple is more general.
    "groups": "integer",
    "dilation": "tuple", # Can also accept int, but tuple is more general.
    "bias": "boolean"
}
signatures["torch.nn.modules.LazyInstanceNorm1d"] = {
    "eps": "float",
    "momentum": "float",
    "affine": "boolean",
    "track_running_stats": "boolean"
}
signatures["torch.nn.modules.LazyInstanceNorm2d"] = {
    "eps": "float",
    "momentum": "float",
    "affine": "boolean",
    "track_running_stats": "boolean"
}
signatures["torch.nn.modules.LazyInstanceNorm3d"] = {
    "eps": "float",
    "momentum": "float",
    "affine": "boolean",
    "track_running_stats": "boolean"
}
signatures["torch.nn.modules.LazyLinear"] = {
    "in_features": "integer", # could also be None but most commonly it's initialized lazily
    "out_features": "integer",
    "bias": "boolean"
}
signatures["torch.nn.modules.LeakyReLU"] = {
    "negative_slope": "float",
    "inplace": "boolean"
}
signatures["torch.nn.modules.Linear"] = {
    "in_features": "integer",
    "out_features": "integer",
    "bias": "boolean"
}
signatures["torch.nn.modules.LocalResponseNorm"] = {
    "size": "integer",
    "alpha": "float",
    "beta": "float",
    "k": "float"
}
signatures["torch.nn.modules.LogSigmoid"] = {}
signatures["torch.nn.modules.LogSoftmax"] = {
    "dim": "integer" # Could also accept a tuple of ints, but integer is more common
}
signatures["torch.nn.modules.MSELoss"] = {
    "size_average": "boolean", # Could be deprecated boolean, but keeping it boolean for now
    "reduce": "boolean", # Could be deprecated boolean, but keeping it boolean for now
    "reduction": "string"
}
signatures["torch.nn.modules.MarginRankingLoss"] = {
    "margin": "float",
    "reduction": "string"
}
signatures["torch.nn.modules.MaxPool1d"] = {
    "kernel_size": "integer", # Could also be a tuple, but integer is more common
    "stride": "integer", # Could also be a tuple, but integer is more common
    "padding": "integer", # Could also be a tuple, but integer is more common
    "dilation": "integer",
    "return_indices": "boolean",
    "ceil_mode": "boolean"
}
signatures["torch.nn.modules.MaxPool2d"] = {
    "kernel_size": "integer", # Can also be a tuple, but integer is more common
    "stride": "integer", # Can also be a tuple, but integer is more common
    "padding": "integer", # Can also be a tuple, but integer is more common
    "dilation": "integer",
    "return_indices": "boolean",
    "ceil_mode": "boolean"
}
signatures["torch.nn.modules.MaxPool3d"] = {
    "kernel_size": "integer", # Could be tuple, but integer is more common
    "stride": "integer", # Could be tuple, but integer is more common
    "padding": "integer", # Could be tuple, but integer is more common
    "dilation": "integer", # Could be tuple, but integer is more common
    "return_indices": "boolean",
    "ceil_mode": "boolean"
}
signatures["torch.nn.modules.MaxUnpool1d"] = {
    "kernel_size": "integer", # could also be tuple, but integer is more common
    "stride": "integer", # could also be tuple, but integer is more common
    "padding": "integer" # could also be tuple, but integer is more common
}
signatures["torch.nn.modules.MaxUnpool2d"] = {
    "kernel_size": "integer", # could be tuple
    "stride": "integer", # could be tuple
    "padding": "integer" # could be tuple
}
signatures["torch.nn.modules.MaxUnpool3d"] = {
    "kernel_size": "integer", # can also be tuple
    "stride": "integer", # can also be tuple
    "padding": "integer" # can also be tuple
}
signatures["torch.nn.modules.Mish"] = {
    "inplace": "boolean" # Documentation doesn't specify a type explicitly, but the argument suggests a boolean.
}
signatures["torch.nn.modules.Module"] = {
}
signatures["torch.nn.modules.ModuleDict"] = {
    "modules": "dict" # Could also be None, but dict is more common
}
signatures["torch.nn.modules.ModuleList"] = {
    "modules": "list" # Should ideally be Module or a list of Modules, but "list" is the closest type
}
signatures["torch.nn.modules.MultiLabelMarginLoss"] = {
    "margin": "float",
    "weight": "tensor",
    "size_average": "boolean", #deprecated
    "reduce": "boolean", #deprecated
    "reduction": "string"
}
signatures["torch.nn.modules.MultiLabelSoftMarginLoss"] = {
    "weight": "tensor",
    "size_average": "boolean",
    "reduce": "boolean",
    "reduction": "string"
}
signatures["torch.nn.modules.MultiMarginLoss"] = {
    "p": "integer", # could be float
    "weight": "tensor",
    "margin": "float",
    "reduction": "string"
}
signatures["torch.nn.modules.MultiheadAttention"] = {
    "embed_dim": "integer",
    "num_heads": "integer",
    "dropout": "float",
    "bias": "boolean",
    "add_bias_kv": "boolean",
    "add_zero_attn": "boolean",
    "kdim": "integer",
    "vdim": "integer",
    "batch_first": "boolean",
    "device": "string", # Should probably not include, but spec said to choose the most common
    "dtype": "dtype"
}
signatures["torch.nn.modules.NLLLoss"] = {
    "weight": "tensor",
    "size_average": "boolean", # Deprecated
    "ignore_index": "integer",
    "reduce": "boolean", # Deprecated
    "reduction": "string"
}
signatures["torch.nn.modules.NLLLoss2d"] = {
    "weight": "tensor",
    "size_average": "boolean", # Deprecated
    "ignore_index": "integer",
    "reduce": "boolean", # Deprecated
    "reduction": "string"
}
signatures["torch.nn.modules.PReLU"] = {
    "num_parameters": "integer",
    "init": "float"
}
signatures["torch.nn.modules.PairwiseDistance"] = {
    "p": "float",
    "eps": "float", # Should this be a number?
    "keepdim": "boolean"
}
signatures["torch.nn.modules.ParameterDict"] = {
    "parameters": "list" # Could also be OrderedDict, but list of Parameters seems more accurate as input
}
signatures["torch.nn.modules.ParameterList"] = {
    "parameters": "list" # could also be tuple, but list seems more common
}
signatures["torch.nn.modules.PixelShuffle"] = {
    "upscale_factor": "integer"
}
signatures["torch.nn.modules.PixelUnshuffle"] = {
    "downscale_factor": "integer"
}
signatures["torch.nn.modules.PoissonNLLLoss"] = {
    "log_input": "boolean",
    "full": "boolean",
    "eps": "float",
    "reduction": "string"
}
signatures["torch.nn.modules.RMSNorm"] = {
    "normalized_shape": "integer", # Could also be tuple/list, but integer seems most common based on examples.
    "eps": "float",
    "elementwise_affine": "boolean"
}
signatures["torch.nn.modules.RNN"] = {
    "input_size": "integer",
    "hidden_size": "integer",
    "num_layers": "integer",
    "nonlinearity": "string",
    "bias": "boolean",
    "batch_first": "boolean",
    "dropout": "float",
    "bidirectional": "boolean"
}
signatures["torch.nn.modules.RNNBase"] = {
    "mode": "string",
    "input_size": "integer",
    "hidden_size": "integer",
    "num_layers": "integer",
    "bias": "boolean",
    "batch_first": "boolean",
    "dropout": "float",
    "bidirectional": "boolean",
    "proj_size": "integer" # I'm not 100% sure on this, it's an integer
}
signatures["torch.nn.modules.RNNCell"] = {
    "input": "tensor",
    "hx": "tensor",
}
signatures["torch.nn.modules.RNNCellBase"] = {
    "input_size": "integer",
    "hidden_size": "integer",
    "bias": "boolean",
    "num_chunks": "integer" # This might be better suited as optional[integer] but I don't have that as an option.
}
signatures["torch.nn.modules.RReLU"] = {
    "lower": "float",
    "upper": "float",
    "inplace": "boolean"
}
signatures["torch.nn.modules.ReLU"] = {
    "inplace": "boolean"
}
signatures["torch.nn.modules.ReLU6"] = {
    "inplace": "boolean"
}
signatures["torch.nn.modules.ReflectionPad1d"] = {
    "padding": "integer" # Can also accept tuple of ints, choosing most common int
}
signatures["torch.nn.modules.ReflectionPad2d"] = {
    "padding": "integer" # can also be tuple
}
signatures["torch.nn.modules.ReflectionPad3d"] = {
    "padding": "integer" # can also be tuple/list of integers
}
signatures["torch.nn.modules.ReplicationPad1d"] = {
    "padding": "tuple", # Could also be an integer
}
signatures["torch.nn.modules.ReplicationPad2d"] = {
    "padding": "tuple" # Can also be an int, but tuple is more common
}
signatures["torch.nn.modules.ReplicationPad3d"] = {
    "padding": "integer" # Could also be tuple or list, but integer is more common
}
signatures["torch.nn.modules.SELU"] = {
    "inplace": "boolean"
}
signatures["torch.nn.modules.Sequential"] = {
    "args": "list" # Could also be OrderedDict, but list is more common for passing layers
}
signatures["torch.nn.modules.SiLU"] = {
    "inplace": "boolean" # Could also be None, but boolean is more common
}
signatures["torch.nn.modules.Sigmoid"] = {}
signatures["torch.nn.modules.SmoothL1Loss"] = {
    "reduction": "string",
    "beta": "float" # Originally float, but could also potentially be a tensor
}
signatures["torch.nn.modules.SoftMarginLoss"] = {
    "reduction": "string"
}
signatures["torch.nn.modules.Softmax"] = {
    "dim": "integer" # Could be None, but integer is most common
}
signatures["torch.nn.modules.Softmax2d"] = {
}
signatures["torch.nn.modules.Softmin"] = {
    "dim": "integer" # Could be None, but integer is more common
}
signatures["torch.nn.modules.Softplus"] = {
    "beta": "float",
    "threshold": "float"
}
signatures["torch.nn.modules.Softshrink"] = {
    "lambd": "float"
}
signatures["torch.nn.modules.Softsign"] = {
    "input": "tensor"
}
signatures["torch.nn.modules.SyncBatchNorm"] = {
    "num_features": "integer",
    "eps": "float",
    "momentum": "float",
    "affine": "boolean",
    "track_running_stats": "boolean"
}
signatures["torch.nn.modules.Tanh"] = {
    "input": "tensor"
}
signatures["torch.nn.modules.Tanhshrink"] = {
    "input": "tensor"
}
signatures["torch.nn.modules.Threshold"] = {
    "threshold": "float",
    "value": "float",
    "inplace": "boolean"
}
signatures["torch.nn.modules.Transformer"] = {
    "d_model": "integer",
    "nhead": "integer",
    "num_encoder_layers": "integer",
    "num_decoder_layers": "integer",
    "dim_feedforward": "integer",
    "dropout": "float",
    "activation": "string",
    "custom_encoder": "tensor", # could also be module
    "custom_decoder": "tensor", # could also be module
    "layer_norm_eps": "float",
    "batch_first": "boolean",
    "norm_first": "boolean",
    "memory_mask": "boolean" # actually, its a Tensor, but based on documentation, it is boolean mask
}
signatures["torch.nn.modules.TransformerDecoder"] = {
    "decoder_layer": "object", # should ideally be nn.TransformerDecoderLayer, but object is closest
    "num_layers": "integer",
    "norm": "object" # should ideally be nn.LayerNorm, but object is closest
}
signatures["torch.nn.modules.TransformerDecoderLayer"] = {
    "d_model": "integer",
    "nhead": "integer",
    "dim_feedforward": "integer",
    "dropout": "float",
    "activation": "string",
    "layer_norm_eps": "float",
    "batch_first": "boolean",
    "norm_first": "boolean"
}
signatures["torch.nn.modules.TransformerEncoder"] = {
    "encoder_layer": "object", # Actually a TransformerEncoderLayer object
    "num_layers": "integer",
    "norm": "object" # Actually a LayerNorm object
}
signatures["torch.nn.modules.TransformerEncoderLayer"] = {
    "d_model": "integer",
    "nhead": "integer",
    "dim_feedforward": "integer",
    "dropout": "float",
    "activation": "string",
    "layer_norm_eps": "float",
    "batch_first": "boolean",
    "norm_first": "boolean"
}
signatures["torch.nn.modules.TripletMarginLoss"] = {
    "margin": "float",
    "p": "float",
    "eps": "float",
    "swap": "boolean",
    "size_average": "boolean", # deprecated
    "reduce": "boolean", # deprecated
    "reduction": "string"
}
signatures["torch.nn.modules.TripletMarginWithDistanceLoss"] = {
    "distance_function": "string", # Could potentially be a callable, but most commonly a string.
    "margin": "float",
    "swap": "boolean",
    "reduction": "string"
}
signatures["torch.nn.modules.Unflatten"] = {
    "dim": "integer",
    "unflattened_size": "tuple" # Could also be an integer, but tuple is more common
}
signatures["torch.nn.modules.Unfold"] = {
    "kernel_size": "tuple", # Could also be int, but tuple is more common
    "dilation": "tuple", # Could also be int, but tuple is more common
    "padding": "tuple", # Could also be int, but tuple is more common
    "stride": "tuple" # Could also be int, but tuple is more common
}
signatures["torch.nn.modules.Upsample"] = {
    "size": "tuple", # Could also be integer or None
    "scale_factor": "float", # Could also be tuple or list or None
    "mode": "string",
    "align_corners": "boolean",
    "recompute_scale_factor": "boolean" # Available since v1.1.0
}
signatures["torch.nn.modules.UpsamplingBilinear2d"] = {
    "size": "tuple", # could also be integer list/tuple
    "scale_factor": "float", # could also be a tuple of floats
}
signatures["torch.nn.modules.UpsamplingNearest2d"] = {
    "size": "tuple", # can also be an integer, but tuple is more common for spatial dimensions
    "scale_factor": "float" # can also be a tuple, but float is more common for single factor upsampling
}
signatures["torch.nn.modules.ZeroPad1d"] = {
    "padding": "tuple", # Could be integer or tuple. Choosing tuple as the more general option.
}
signatures["torch.nn.modules.ZeroPad2d"] = {
    "padding": "tuple" # can also be int or tuple of int, choosing tuple for simplicity
}
signatures["torch.nn.modules.ZeroPad3d"] = {
    "padding": "tuple", # Could also be integer or list, but tuple seems most common
}
signatures["torch.nn.functional.relu"] = {
    "input": "tensor",
    "inplace": "boolean"
}
signatures["torch.nn.functional.relu6"] = {
    "input": "tensor",
    "inplace": "boolean"
}
signatures["torch.nn.functional.elu"] = {
    "input": "tensor",
    "alpha": "float",
    "inplace": "boolean"
}
signatures["torch.nn.functional.celu"] = {
    "input": "tensor",
    "alpha": "float",
    "inplace": "boolean"
}
signatures["torch.nn.functional.selu"] = {
    "input": "tensor",
    "inplace": "boolean"
}
signatures["torch.nn.functional.gelu"] = {
    "input": "tensor",
    "approximate": "string" # Should probably be an enum of 'none' or 'tanh'
}
signatures["torch.nn.functional.hardshrink"] = {
    "input": "tensor",
    "lambd": "float"
}
signatures["torch.nn.functional.tanhshrink"] = {
    "input": "tensor"
}
signatures["torch.nn.functional.hardtanh"] = {
    "input": "tensor",
    "min_val": "float",
    "max_val": "float",
    "inplace": "boolean"
}
signatures["torch.nn.functional.sigmoid"] = {
    "input": "tensor"
}
signatures["torch.nn.functional.hardsigmoid"] = {
    "input": "tensor",
    "inplace": "boolean"
}
signatures["torch.nn.functional.silu"] = {
    "input": "tensor",
    "inplace": "boolean"
}
signatures["torch.nn.functional.mish"] = {
    "input": "tensor",
    "inplace": "boolean"
}
signatures["torch.nn.functional.softsign"] = {
    "input": "tensor"
}
signatures["torch.nn.functional.softplus"] = {
    "input": "tensor",
    "beta": "float",
    "threshold": "float"
}
signatures["torch.nn.functional.softmin"] = {
    "input": "tensor",
    "dim": "integer",
    "dtype": "dtype"
}
signatures["torch.nn.functional.softmax"] = {
    "input": "tensor",
    "dim": "integer",
    "dtype": "dtype"
}
signatures["torch.nn.functional.gumbel_softmax"] = {
    "logits": "tensor",
    "tau": "float",
    "hard": "boolean",
    "dim": "integer"
}
signatures["torch.nn.functional.log_softmax"] = {
    "input": "tensor",
    "dim": "integer",
    "dtype": "dtype"
}
signatures["torch.nn.functional.tanh"] = {
    "input": "tensor"
}
signatures["torch.nn.functional.threshold"] = {
    "input": "tensor",
    "threshold": "float",
    "value": "float",
    "inplace": "boolean"
}
signatures["torch.nn.functional.leaky_relu"] = {
    "input": "tensor",
    "negative_slope": "float",
    "inplace": "boolean"
}
signatures["torch.nn.functional.prelu"] = {
    "input": "tensor",
    "weight": "tensor"
}
signatures["torch.nn.functional.rrelu"] = {
    "input": "tensor",
    "lower": "float",
    "upper": "float",
    "training": "boolean", # Can be None
    "inplace": "boolean"
}
signatures["torch.nn.functional.elu"] = {
    "input": "tensor",
    "alpha": "float",
    "inplace": "boolean"
}
signatures["torch.nn.AdaptiveLogSoftmaxWithLoss"] = {
    "in_features": "integer",
    "n_classes": "integer",
    "cutoffs": "list", # could be tuple as well?
    "div_value": "float",
    "head_bias": "boolean",
    "adaptive_float16": "boolean"
}
signatures["torch.nn.AdaptiveAvgPool1d"] = {
    "output_size": "integer"
}
signatures["torch.nn.AdaptiveAvgPool2d"] = {
    "output_size": "tuple" # can also be integer, but tuple seems more common
}
signatures["torch.nn.AdaptiveAvgPool3d"] = {
    "output_size": "tuple" # can also be integer, but tuple seems more common
}
signatures["torch.nn.modules.batchnorm.BatchNorm1d"] = {
    "num_features": "integer",
    "eps": "float",
    "momentum": "float",
    "affine": "boolean",
    "track_running_stats": "boolean"
}
signatures["torch.nn.modules.batchnorm.BatchNorm2d"] = {
    "num_features": "integer",
    "eps": "float",
    "momentum": "float",
    "affine": "boolean",
    "track_running_stats": "boolean"
}
signatures["torch.nn.modules.batchnorm.BatchNorm3d"] = {
    "num_features": "integer",
    "eps": "float",
    "momentum": "float",
    "affine": "boolean",
    "track_running_stats": "boolean"
}
signatures["torch.nn.modules.batchnorm.SyncBatchNorm"] = {
    "num_features": "integer",
    "eps": "float",
    "momentum": "float",
    "affine": "boolean",
    "track_running_stats": "boolean",
    "process_group": "list" # I'm not entirely sure what the type of this argument is, but it is some kind of group, so I am choosing list.
}
signatures["torch.nn.modules.channelshuffle"] = {
    "groups": "integer"
}
signatures["torch.nn.modules.container.Module"] = {} # Base class, no specific signature
signatures["torch.nn.Sequential"] = {
    "args": "list" # or "tuple", but list seems more common in examples
}
signatures["torch.nn.ModuleList"] = {
    "modules": "list"
}
signatures["torch.nn.ModuleDict"] = {
    "modules": "dict"
}
signatures["torch.nn.ParameterList"] = {
    "parameters": "list"
}
signatures["torch.nn.ParameterDict"] = {
    "parameters": "dict"
}
signatures["torch.nn.modules.conv.Conv2d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # Can be int or tuple, but tuple is more common
    "stride": "tuple", # Can be int or tuple, but tuple is more common
    "padding": "tuple", # Can be string, int or tuple, but tuple is most common
    "dilation": "tuple", # Can be int or tuple, but tuple is more common
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.modules.conv.ConvTranspose2d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # Can be int or tuple, but tuple is more common
    "stride": "tuple", # Can be int or tuple, but tuple is more common
    "padding": "tuple", # Can be int or tuple, but tuple is most common
    "output_padding": "tuple", # Can be int or tuple, but tuple is more common
    "groups": "integer",
    "bias": "boolean",
    "dilation": "tuple",
    "padding_mode": "string"
}
signatures["torch.nn.modules.conv.Conv3d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # Can be int or tuple, but tuple is more common
    "stride": "tuple", # Can be int or tuple, but tuple is more common
    "padding": "tuple", # Can be string, int or tuple, but tuple is most common
    "dilation": "tuple", # Can be int or tuple, but tuple is more common
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.modules.conv.Conv1d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "integer", # Can be int or tuple, but int is more common here
    "stride": "integer", # Can be int or tuple, but int is more common here
    "padding": "string", # Can be string or int or tuple, but string is added to match documentation
    "dilation": "integer", # Can be int or tuple, but int is more common here
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.modules.conv.ConvTranspose3d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # Can be int or tuple, but tuple is more common
    "stride": "tuple", # Can be int or tuple, but tuple is more common
    "padding": "tuple", # Can be int or tuple, but tuple is most common
    "output_padding": "tuple", # Can be int or tuple, but tuple is more common
    "groups": "integer",
    "bias": "boolean",
    "dilation": "tuple",
    "padding_mode": "string"
}
signatures["torch.nn.modules.conv.Conv"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple",
    "stride": "tuple",
    "padding": "tuple",
    "dilation": "tuple",
    "transposed": "boolean",
    "output_padding": "tuple",
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.modules.conv.LazyConv2d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple",
    "stride": "tuple",
    "padding": "tuple",
    "dilation": "tuple",
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.modules.conv.LazyConv1d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "integer",
    "stride": "integer",
    "padding": "string",
    "dilation": "integer",
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.modules.conv.LazyConv3d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple",
    "stride": "tuple",
    "padding": "tuple",
    "dilation": "tuple",
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.modules.conv.Unfold"] = {
    "kernel_size": "tuple",
    "dilation": "tuple",
    "padding": "tuple",
    "stride": "tuple"
}
signatures["torch.nn.modules.conv.Fold"] = {
    "output_size": "tuple",
    "kernel_size": "tuple",
    "dilation": "tuple",
    "padding": "tuple",
    "stride": "tuple"
}
signatures["torch.nn.modules.distance.PairwiseDistance"] = {
    "p": "float", # Could also be integer? Not sure which is more common
    "eps": "float",
    "keepdim": "boolean"
}
signatures["torch.nn.modules.distance.CosineSimilarity"] = {
    "dim": "integer",
    "eps": "float"
}
signatures["torch.nn.modules.dropout.Dropout"] = {
    "p": "float",
    "inplace": "boolean"
}
signatures["torch.nn.modules.dropout.Dropout2d"] = {
    "p": "float",
    "inplace": "boolean"
}
signatures["torch.nn.modules.dropout.Dropout3d"] = {
    "p": "float",
    "inplace": "boolean"
}
signatures["torch.nn.modules.flatten.Flatten"] = {
    "start_dim": "integer",
    "end_dim": "integer"
}
signatures["torch.nn.modules.fold"] = {
    "input": "tensor",
    "output_size": "tuple", # Could also be integer_list, but tuple is more common
    "kernel_size": "tuple", # Could also be integer_list, but tuple is more common
    "dilation": "tuple", # Could also be integer_list, but tuple is more common
    "padding": "tuple", # Could also be integer_list, but tuple is more common
    "stride": "tuple" # Could also be integer_list, but tuple is more common
}
signatures["torch.nn.modules.instancenorm"] = {
    "num_features": "integer",
    "eps": "float",
    "momentum": "float",
    "affine": "boolean",
    "track_running_stats": "boolean"
}
signatures["torch.nn.modules.lazy.LazyModuleMixin"] = {} # This is a mixin class, it doesn't have an explicit signature.
signatures["torch.nn.modules.lazy.LazyLinear"] = {
    "in_features": "integer", # Can be None, but usually integer
    "out_features": "integer",
    "bias": "boolean"
}
signatures["torch.nn.modules.lazy.LazyConv1d"] = {
    "in_channels": "integer", # Can be None, but usually integer
    "out_channels": "integer",
    "kernel_size": "tuple", # or integer
    "stride": "tuple", # or integer
    "padding": "string", # or tuple or integer
    "dilation": "tuple", # or integer
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.modules.lazy.LazyConv2d"] = {
    "in_channels": "integer", # Can be None, but usually integer
    "out_channels": "integer",
    "kernel_size": "tuple", # or integer
    "stride": "tuple", # or integer
    "padding": "string", # or tuple or integer
    "dilation": "tuple", # or integer
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.modules.lazy.LazyConv3d"] = {
    "in_channels": "integer", # Can be None, but usually integer
    "out_channels": "integer",
    "kernel_size": "tuple", # or integer
    "stride": "tuple", # or integer
    "padding": "string", # or tuple or integer
    "dilation": "tuple", # or integer
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.modules.lazy.LazyConvTranspose1d"] = {
    "in_channels": "integer", # Can be None, but usually integer
    "out_channels": "integer",
    "kernel_size": "tuple", # or integer
    "stride": "tuple", # or integer
    "padding": "string", # or tuple or integer
    "output_padding": "tuple", # or integer
    "groups": "integer",
    "bias": "boolean",
    "dilation": "tuple" # or integer
}
signatures["torch.nn.modules.lazy.LazyConvTranspose2d"] = {
    "in_channels": "integer", # Can be None, but usually integer
    "out_channels": "integer",
    "kernel_size": "tuple", # or integer
    "stride": "tuple", # or integer
    "padding": "string", # or tuple or integer
    "output_padding": "tuple", # or integer
    "groups": "integer",
    "bias": "boolean",
    "dilation": "tuple" # or integer
}
signatures["torch.nn.modules.lazy.LazyConvTranspose3d"] = {
    "in_channels": "integer", # Can be None, but usually integer
    "out_channels": "integer",
    "kernel_size": "tuple", # or integer
    "stride": "tuple", # or integer
    "padding": "string", # or tuple or integer
    "output_padding": "tuple", # or integer
    "groups": "integer",
    "bias": "boolean",
    "dilation": "tuple" # or integer
}
signatures["torch.nn.modules.lazy.LazyBatchNorm1d"] = {
    "num_features": "integer", # Can be None, but usually integer
    "eps": "float",
    "momentum": "float",
    "affine": "boolean",
    "track_running_stats": "boolean"
}
signatures["torch.nn.modules.lazy.LazyBatchNorm2d"] = {
    "num_features": "integer", # Can be None, but usually integer
    "eps": "float",
    "momentum": "float",
    "affine": "boolean",
    "track_running_stats": "boolean"
}
signatures["torch.nn.modules.lazy.LazyBatchNorm3d"] = {
    "num_features": "integer", # Can be None, but usually integer
    "eps": "float",
    "momentum": "float",
    "affine": "boolean",
    "track_running_stats": "boolean"
}
signatures["torch.nn.modules.linear.Linear"] = {
    "in_features": "integer",
    "out_features": "integer",
    "bias": "boolean"
}
signatures["torch.nn.modules.loss.L1Loss"] = {
    "reduction": "string"
}
signatures["torch.nn.modules.loss.MSELoss"] = {
    "reduction": "string"
}
signatures["torch.nn.modules.loss.CrossEntropyLoss"] = {
    "weight": "tensor",
    "ignore_index": "integer",
    "reduction": "string",
    "label_smoothing": "float"
}
signatures["torch.nn.modules.loss.CTCLoss"] = {
    "blank": "integer",
    "reduction": "string",
    "zero_infinity": "boolean"
}
signatures["torch.nn.modules.loss.NLLLoss"] = {
    "weight": "tensor",
    "ignore_index": "integer",
    "reduction": "string"
}
signatures["torch.nn.modules.loss.PoissonNLLLoss"] = {
    "log_input": "boolean",
    "full": "boolean",
    "eps": "float",
    "reduction": "string"
}
signatures["torch.nn.modules.loss.KLDivLoss"] = {
    "reduction": "string",
    "log_target": "boolean"
}
signatures["torch.nn.modules.loss.BCELoss"] = {
    "weight": "tensor",
    "reduction": "string"
}
signatures["torch.nn.modules.loss.BCEWithLogitsLoss"] = {
    "weight": "tensor",
    "pos_weight": "tensor",
    "reduction": "string"
}
signatures["torch.nn.modules.loss.MarginRankingLoss"] = {
    "margin": "float",
    "reduction": "string"
}
signatures["torch.nn.modules.loss.HingeEmbeddingLoss"] = {
    "margin": "float",
    "reduction": "string"
}
signatures["torch.nn.modules.loss.CosineEmbeddingLoss"] = {
    "margin": "float",
    "reduction": "string"
}
signatures["torch.nn.modules.loss.MultiLabelMarginLoss"] = {
    "reduction": "string"
}
signatures["torch.nn.modules.loss.MultiLabelSoftMarginLoss"] = {
    "weight": "tensor",
    "reduction": "string"
}
signatures["torch.nn.modules.loss.SoftMarginLoss"] = {
    "reduction": "string"
}
signatures["torch.nn.modules.loss.MultiMarginLoss"] = {
    "p": "integer",
    "weight": "tensor",
    "margin": "float",
    "reduction": "string"
}
signatures["torch.nn.modules.loss.TripletMarginLoss"] = {
    "margin": "float",
    "p": "float",
    "eps": "float",
    "swap": "boolean",
    "reduction": "string"
}
signatures["torch.nn.modules.loss.TripletMarginWithDistanceLoss"] = {
    "distance_function": "string", # should probably be a callable, but string is the closest match
    "margin": "float",
    "swap": "boolean",
    "reduction": "string"
}
signatures["torch.nn.modules.loss.HuberLoss"] = {
    "delta": "float",
    "reduction": "string"
}
signatures["torch.nn.modules.loss.SmoothL1Loss"] = {
    "beta": "float",
    "reduction": "string"
}
signatures["torch.nn.modules.loss.GaussianNLLLoss"] = {
    "full": "boolean",
    "eps": "float",
    "reduction": "string"
}
signatures["torch.nn.modules.loss.LaplacianNLLLoss"] = {
    "reduction": "string"
}
signatures["torch.nn.modules.loss.PReLU"] = {
}
signatures["torch.nn.modules.module"] = {} # this is a base class, so there's no real signature to provide
signatures["torch.nn.BatchNorm1d"] = {
    "num_features": "integer",
    "eps": "float",
    "momentum": "float",
    "affine": "boolean",
    "track_running_stats": "boolean"
}
signatures["torch.nn.BatchNorm2d"] = {
    "num_features": "integer",
    "eps": "float",
    "momentum": "float",
    "affine": "boolean",
    "track_running_stats": "boolean"
}
signatures["torch.nn.BatchNorm3d"] = {
    "num_features": "integer",
    "eps": "float",
    "momentum": "float",
    "affine": "boolean",
    "track_running_stats": "boolean"
}
signatures["torch.nn.GroupNorm"] = {
    "num_groups": "integer",
    "num_channels": "integer",
    "eps": "float",
    "affine": "boolean"
}
signatures["torch.nn.InstanceNorm1d"] = {
    "num_features": "integer",
    "eps": "float",
    "momentum": "float",
    "affine": "boolean",
    "track_running_stats": "boolean"
}
signatures["torch.nn.InstanceNorm2d"] = {
    "num_features": "integer",
    "eps": "float",
    "momentum": "float",
    "affine": "boolean",
    "track_running_stats": "boolean"
}
signatures["torch.nn.InstanceNorm3d"] = {
    "num_features": "integer",
    "eps": "float",
    "momentum": "float",
    "affine": "boolean",
    "track_running_stats": "boolean"
}
signatures["torch.nn.LayerNorm"] = {
    "normalized_shape": "list", # could be tuple as well, but list seems more appropriate
    "eps": "float",
    "elementwise_affine": "boolean"
}
signatures["torch.nn.LocalResponseNorm"] = {
    "size": "integer",
    "alpha": "float",
    "beta": "float",
    "k": "float"
}
signatures["torch.nn.SpectralNorm"] = {
    "module": "tensor", #this should really be nn.Module, but tensor is the closest available type
    "name": "string",
    "n_power_iterations": "integer",
    "eps": "float",
    "dim": "integer"
}
signatures["torch.nn.WeightNorm"] = {
    "module": "tensor", #this should really be nn.Module, but tensor is the closest available type
    "name": "string",
    "dim": "integer"
}
signatures["torch.nn.functional.pad"] = {
    "input": "tensor",
    "pad": "tuple", # Can be a list or tuple, choosing tuple since it's usually immutable
    "mode": "string",
    "value": "float"
}
signatures["torch.nn.ConstantPad1d"] = {
    "padding": "tuple", # Can be int or tuple
    "value": "float"
}
signatures["torch.nn.ConstantPad2d"] = {
    "padding": "tuple", # Can be int or tuple
    "value": "float"
}
signatures["torch.nn.ConstantPad3d"] = {
    "padding": "tuple", # Can be int or tuple
    "value": "float"
}
signatures["torch.nn.ReflectionPad1d"] = {
    "padding": "tuple" # Can be int or tuple
}
signatures["torch.nn.ReflectionPad2d"] = {
    "padding": "tuple" # Can be int or tuple
}
signatures["torch.nn.ReflectionPad3d"] = {
    "padding": "tuple" # Can be int or tuple
}
signatures["torch.nn.ReplicationPad1d"] = {
    "padding": "tuple" # Can be int or tuple
}
signatures["torch.nn.ReplicationPad2d"] = {
    "padding": "tuple" # Can be int or tuple
}
signatures["torch.nn.ReplicationPad3d"] = {
    "padding": "tuple" # Can be int or tuple
}
signatures["torch.nn.ZeroPad2d"] = {
    "padding": "tuple" # Can be int or tuple
}
signatures["torch.nn.modules.pixelshuffle"] = {
    "upscale_factor": "integer"
}
signatures["MaxPool1d"] = {
    "kernel_size": "integer",
    "stride": "integer",
    "padding": "integer",
    "dilation": "integer",
    "return_indices": "boolean",
    "ceil_mode": "boolean"
}
signatures["MaxPool2d"] = {
    "kernel_size": "integer",
    "stride": "integer",
    "padding": "integer",
    "dilation": "integer",
    "return_indices": "boolean",
    "ceil_mode": "boolean"
}
signatures["MaxPool3d"] = {
    "kernel_size": "integer",
    "stride": "integer",
    "padding": "integer",
    "dilation": "integer",
    "return_indices": "boolean",
    "ceil_mode": "boolean"
}
signatures["MaxUnpool1d"] = {
    "kernel_size": "integer",
    "stride": "integer",
    "padding": "integer"
}
signatures["MaxUnpool2d"] = {
    "kernel_size": "integer",
    "stride": "integer",
    "padding": "integer"
}
signatures["MaxUnpool3d"] = {
    "kernel_size": "integer",
    "stride": "integer",
    "padding": "integer"
}
signatures["AvgPool1d"] = {
    "kernel_size": "integer",
    "stride": "integer",
    "padding": "integer",
    "ceil_mode": "boolean",
    "count_include_pad": "boolean"
}
signatures["AvgPool2d"] = {
    "kernel_size": "integer",
    "stride": "integer",
    "padding": "integer",
    "ceil_mode": "boolean",
    "count_include_pad": "boolean",
    "divisor_override": "integer" # could be none, which I'd consider float but integer is more appropriate
}
signatures["AvgPool3d"] = {
    "kernel_size": "integer",
    "stride": "integer",
    "padding": "integer",
    "ceil_mode": "boolean",
    "count_include_pad": "boolean",
    "divisor_override": "integer" # could be none, which I'd consider float but integer is more appropriate
}
signatures["FractionalMaxPool2d"] = {
    "kernel_size": "integer",
    "output_size": "integer",
    "output_ratio": "tuple", # could be single float
    "return_indices": "boolean"
}
signatures["LPPool1d"] = {
    "norm_type": "float",
    "kernel_size": "integer",
    "stride": "integer",
    "ceil_mode": "boolean"
}
signatures["LPPool2d"] = {
    "norm_type": "float",
    "kernel_size": "integer",
    "stride": "integer",
    "ceil_mode": "boolean"
}
signatures["AdaptiveMaxPool1d"] = {
    "output_size": "integer",
    "return_indices": "boolean"
}
signatures["AdaptiveMaxPool2d"] = {
    "output_size": "integer",
    "return_indices": "boolean"
}
signatures["AdaptiveMaxPool3d"] = {
    "output_size": "tuple",
    "return_indices": "boolean"
}
signatures["AdaptiveAvgPool1d"] = {
    "output_size": "integer"
}
signatures["AdaptiveAvgPool2d"] = {
    "output_size": "integer"
}
signatures["AdaptiveAvgPool3d"] = {
    "output_size": "tuple"
}
signatures["ReflectionPad1d"] = {
    "padding": "integer"
}
signatures["ReflectionPad2d"] = {
    "padding": "integer"
}
signatures["ReflectionPad3d"] = {
    "padding": "integer"
}
signatures["ReplicationPad1d"] = {
    "padding": "integer"
}
signatures["ReplicationPad2d"] = {
    "padding": "integer"
}
signatures["ReplicationPad3d"] = {
    "padding": "integer"
}
signatures["CrossMapLRN2d"] = {
    "size": "integer",
    "alpha": "float",
    "beta": "float",
    "k": "float"
}
signatures["LocalResponseNorm"] = {
    "size": "integer",
    "alpha": "float",
    "beta": "float",
    "k": "float"
}
signatures["BatchNorm1d"] = {
    "num_features": "integer",
    "eps": "float",
    "momentum": "float",
    "affine": "boolean",
    "track_running_stats": "boolean"
}
signatures["BatchNorm2d"] = {
    "num_features": "integer",
    "eps": "float",
    "momentum": "float",
    "affine": "boolean",
    "track_running_stats": "boolean"
}
signatures["BatchNorm3d"] = {
    "num_features": "integer",
    "eps": "float",
    "momentum": "float",
    "affine": "boolean",
    "track_running_stats": "boolean"
}
signatures["GroupNorm"] = {
    "num_groups": "integer",
    "num_channels": "integer",
    "eps": "float",
    "affine": "boolean"
}
signatures["InstanceNorm1d"] = {
    "num_features": "integer",
    "eps": "float",
    "momentum": "float",
    "affine": "boolean",
    "track_running_stats": "boolean"
}
signatures["InstanceNorm2d"] = {
    "num_features": "integer",
    "eps": "float",
    "momentum": "float",
    "affine": "boolean",
    "track_running_stats": "boolean"
}
signatures["InstanceNorm3d"] = {
    "num_features": "integer",
    "eps": "float",
    "momentum": "float",
    "affine": "boolean",
    "track_running_stats": "boolean"
}
signatures["LayerNorm"] = {
    "normalized_shape": "tuple", # can be int
    "eps": "float",
    "elementwise_affine": "boolean"
}
signatures["torch.nn.functional.dropout"] = {
    "input": "tensor",
    "p": "float",
    "training": "boolean",
    "inplace": "boolean"
}
signatures["torch.nn.functional.dropout2d"] = {
    "input": "tensor",
    "p": "float",
    "training": "boolean",
    "inplace": "boolean"
}
signatures["torch.nn.functional.dropout3d"] = {
    "input": "tensor",
    "p": "float",
    "training": "boolean",
    "inplace": "boolean"
}
signatures["torch.nn.functional.max_pool1d"] = {
    "input": "tensor",
    "kernel_size": "integer",
    "stride": "integer",
    "padding": "integer",
    "dilation": "integer",
    "ceil_mode": "boolean",
    "return_indices": "boolean"
}
signatures["torch.nn.functional.max_pool2d"] = {
    "input": "tensor",
    "kernel_size": "integer",
    "stride": "integer",
    "padding": "integer",
    "dilation": "integer",
    "ceil_mode": "boolean",
    "return_indices": "boolean"
}
signatures["torch.nn.functional.max_pool3d"] = {
    "input": "tensor",
    "kernel_size": "integer",
    "stride": "integer",
    "padding": "integer",
    "dilation": "integer",
    "ceil_mode": "boolean",
    "return_indices": "boolean"
}
signatures["torch.nn.functional.avg_pool1d"] = {
    "input": "tensor",
    "kernel_size": "integer",
    "stride": "integer",
    "padding": "integer",
    "ceil_mode": "boolean",
    "count_include_pad": "boolean"
}
signatures["torch.nn.functional.avg_pool2d"] = {
    "input": "tensor",
    "kernel_size": "integer",
    "stride": "integer",
    "padding": "integer",
    "ceil_mode": "boolean",
    "count_include_pad": "boolean",
    "divisor_override": "integer"
}
signatures["torch.nn.functional.avg_pool3d"] = {
    "input": "tensor",
    "kernel_size": "integer",
    "stride": "integer",
    "padding": "integer",
    "ceil_mode": "boolean",
    "count_include_pad": "boolean",
    "divisor_override": "integer"
}
signatures["torch.nn.functional.fractional_max_pool2d"] = {
    "input": "tensor",
    "kernel_size": "integer",
    "output_size": "integer",
    "output_ratio": "tuple",
    "return_indices": "boolean"
}
signatures["torch.nn.functional.adaptive_max_pool1d"] = {
    "input": "tensor",
    "output_size": "integer",
    "return_indices": "boolean"
}
signatures["torch.nn.functional.adaptive_max_pool2d"] = {
    "input": "tensor",
    "output_size": "integer",
    "return_indices": "boolean"
}
signatures["torch.nn.functional.adaptive_max_pool3d"] = {
    "input": "tensor",
    "output_size": "tuple",
    "return_indices": "boolean"
}
signatures["torch.nn.functional.adaptive_avg_pool1d"] = {
    "input": "tensor",
    "output_size": "integer"
}
signatures["torch.nn.functional.adaptive_avg_pool2d"] = {
    "input": "tensor",
    "output_size": "integer"
}
signatures["torch.nn.functional.adaptive_avg_pool3d"] = {
    "input": "tensor",
    "output_size": "tuple"
}
signatures["torch.nn.functional.interpolate"] = {
    "input": "tensor",
    "size": "integer", # Can be a tuple too
    "scale_factor": "float", # Can be a tuple too
    "mode": "string",
    "align_corners": "boolean",
    "recompute_scale_factor": "boolean"
}
signatures["torch.nn.modules.rnn.RNN"] = {
    "input_size": "integer",
    "hidden_size": "integer",
    "num_layers": "integer",
    "nonlinearity": "string",
    "bias": "boolean",
    "batch_first": "boolean",
    "dropout": "float",
    "bidirectional": "boolean"
}
signatures["torch.nn.modules.rnn.LSTM"] = {
    "input_size": "integer",
    "hidden_size": "integer",
    "num_layers": "integer",
    "bias": "boolean",
    "batch_first": "boolean",
    "dropout": "float",
    "bidirectional": "boolean",
    "proj_size": "integer"
}
signatures["torch.nn.modules.rnn.GRU"] = {
    "input_size": "integer",
    "hidden_size": "integer",
    "num_layers": "integer",
    "bias": "boolean",
    "batch_first": "boolean",
    "dropout": "float",
    "bidirectional": "boolean"
}
signatures["torch.nn.modules.rnn.RNNBase.forward"] = {
    "input": "tensor",
    "hx": "tensor" # Could be a tuple of tensors for LSTM
}
signatures["torch.nn.Embedding"] = {
    "num_embeddings": "integer",
    "embedding_dim": "integer",
    "padding_idx": "integer",
    "max_norm": "float",
    "norm_type": "float",
    "scale_grad_by_freq": "boolean",
    "sparse": "boolean",
    "_weight": "tensor" #actually a parameter, but represents it with tensor
}
signatures["torch.nn.EmbeddingBag"] = {
    "num_embeddings": "integer",
    "embedding_dim": "integer",
    "max_norm": "float",
    "norm_type": "float",
    "scale_grad_by_freq": "boolean",
    "mode": "string",
    "sparse": "boolean",
    "padding_idx": "integer",
    "include_last_offset": "boolean",
    "_weight": "tensor" #actually a parameter, but represents it with tensor
}
signatures["torch.nn.modules.transformer.Transformer"] = {
    "d_model": "integer",
    "nhead": "integer",
    "num_encoder_layers": "integer",
    "num_decoder_layers": "integer",
    "dim_feedforward": "integer",
    "dropout": "float",
    "activation": "string",
    "custom_encoder": "None", # unsure of what the type should be here
    "custom_decoder": "None", # unsure of what the type should be here
    "layer_norm_eps": "float",
    "batch_first": "boolean",
    "norm_first": "boolean"
}
signatures["torch.nn.modules.upsampling.Upsample"] = {
    "size": "tuple", # or integer or None, but tuple is the most common
    "scale_factor": "tuple", # or float or None, but tuple is the most common
    "mode": "string",
    "align_corners": "boolean",
    "recompute_scale_factor": "boolean"
}
signatures["torch.nn.modules.utils._pair"] = {
    "x": "integer" # Could be a tuple but most often an integer
}
signatures["torch.nn.modules.utils._quadruple"] = {
    "x": "integer" # Could be a tuple but most often an integer
}
signatures["torch.nn.modules.utils._single"] = {
    "x": "integer" # Could be a tuple but most often an integer
}
signatures["torch.nn.parallel.DistributedDataParallel"] = {
    "module": "tensor", # Should ideally be nn.Module, but closest is tensor.
    "device_ids": "list", # Could be a list of integers, but list is broader.
    "output_device": "integer",
    "dim": "integer",
    "broadcast_buffers": "boolean",
    "process_group": "tensor", #torch.distributed.ProcessGroup, represents an established distributed process group
    "bucket_cap_mb": "float",
    "find_unused_parameters": "boolean",
    "check_reduction": "boolean",
    "gradient_as_bucket_view": "boolean",
    "static_graph": "boolean"
}
signatures["torch.nn.parallel.DataParallel"] = {
    "module": "tensor", # Should ideally be nn.Module, but closest is tensor.
    "device_ids": "list", # Could be a list of integers, but list is broader.
    "output_device": "integer",
    "dim": "integer"
}
signatures["torch.nn.parallel.DataParallel"] = {
    "module": "module", # Should ideally be nn.Module, but that's not an option
    "device_ids": "list",
    "output_device": "integer",
    "dim": "integer"
}
signatures["torch.nn.parallel.DistributedDataParallel"] = {
    "module": "module", # There isn't a good type here.
    "device_ids": "list",
    "output_device": "integer",
    "dim": "integer",
    "broadcast_buffers": "boolean",
    "init_sync": "boolean",
    "process_group": "process_group", # There isn't a good type here.
    "bucket_cap_mb": "float",
    "find_unused_parameters": "boolean",
    "check_reduction": "boolean",
    "gradient_as_bucket_view": "boolean",
    "static_graph": "boolean",
    "delay_all_reduce_named_params": "list",
    "param_to_hook_all_reduce": "tensor", # or module parameter
    "mixed_precision": "boolean", # Maybe? not explicitly mentioned, but related to mixed precision training
    "device_mesh": "tensor" # There isn't a good type here.
}
signatures["torch.nn.parallel.DistributedDataParallelCPU"] = {
    "module": "tensor", # I think this should be nn.Module, but choosing tensor for simplicity
    "process_group": "list" # Should be a ProcessGroup, but list seems closest
}
signatures["torch.nn.parallel.comm.broadcast"] = {
    "tensor": "tensor",
    "devices": "list" # Should it be "tuple"?
}
signatures["torch.nn.parallel.comm.gather"] = {
    "inputs": "tensor_list",
    "dim": "integer"
}
signatures["torch.nn.parallel.comm.scatter"] = {
    "inputs": "tensor_list",
    "dim": "integer"
}
signatures["torch.nn.parallel.data_parallel"] = {
    "module": "module", # Should ideally be a nn.Module type, but no direct equivalent
    "inputs": "tensor", # Can also be a tuple/list of tensors
    "device_ids": "list", # Can also be an integer, should ideally be a list/tuple of integers
    "output_device": "integer"
}
signatures["torch.nn.parallel.deprecated"] = {
}
signatures["torch.nn.parallel.distributed.all_gather"] = {
    "tensor_list": "list",
    "tensor": "tensor",
    "group": "object", #torch.distributed.ProcessGroup object, but no suitable type available
    "async_op": "boolean"
}
signatures["torch.nn.parallel.distributed.all_reduce"] = {
    "tensor": "tensor",
    "op": "object", #torch.distributed.ReduceOp object, but no suitable type available
    "group": "object", #torch.distributed.ProcessGroup object, but no suitable type available
    "async_op": "boolean"
}
signatures["torch.nn.parallel.distributed.broadcast"] = {
    "tensor": "tensor",
    "src": "integer",
    "group": "object", #torch.distributed.ProcessGroup object, but no suitable type available
    "async_op": "boolean"
}
signatures["torch.nn.parallel.distributed.gather"] = {
    "tensor": "tensor",
    "gather_list": "list",
    "dst": "integer",
    "group": "object", #torch.distributed.ProcessGroup object, but no suitable type available
    "async_op": "boolean"
}
signatures["torch.nn.parallel.distributed.reduce"] = {
    "tensor": "tensor",
    "op": "object", #torch.distributed.ReduceOp object, but no suitable type available
    "dst": "integer",
    "group": "object", #torch.distributed.ProcessGroup object, but no suitable type available
    "async_op": "boolean"
}
signatures["torch.nn.parallel.distributed.scatter"] = {
    "tensor": "tensor",
    "scatter_list": "list",
    "src": "integer",
    "group": "object", #torch.distributed.ProcessGroup object, but no suitable type available
    "async_op": "boolean"
}
signatures["torch.nn.parallel.distributed.send"] = {
    "tensor": "tensor",
    "dst": "integer",
    "group": "object", #torch.distributed.ProcessGroup object, but no suitable type available
    "tag": "integer"
}
signatures["torch.nn.parallel.distributed.recv"] = {
    "tensor": "tensor",
    "src": "integer",
    "group": "object", #torch.distributed.ProcessGroup object, but no suitable type available
    "tag": "integer"
}
signatures["torch.nn.parallel.distributed.barrier"] = {
    "group": "object", #torch.distributed.ProcessGroup object, but no suitable type available
    "async_op": "boolean",
    "timeout": "float"
}
signatures["torch.nn.parallel.distributed.isend"] = {
    "tensor": "tensor",
    "dst": "integer",
    "group": "object", #torch.distributed.ProcessGroup object, but no suitable type available
    "tag": "integer"
}
signatures["torch.nn.parallel.distributed.irecv"] = {
    "tensor": "tensor",
    "src": "integer",
    "group": "object", #torch.distributed.ProcessGroup object, but no suitable type available
    "tag": "integer"
}
signatures["torch.nn.parallel.distributed.new_group"] = {
    "ranks": "list",
    "backend": "string" #backend_type argument, ideally an enum
}
signatures["torch.nn.parallel.distributed.all_to_all"] = {
    "output_tensor_list": "list",
    "input_tensor_list": "list",
    "group": "object", #torch.distributed.ProcessGroup object, but no suitable type available
    "async_op": "boolean"
}
signatures["torch.nn.parallel.gather"] = {
    "target_device": "integer",
    "output_device": "integer",
    "dim": "integer"
}
signatures["torch.nn.parallel.parallel_apply"] = {
    "modules": "list",
    "inputs": "list",
    "kwargs_tup": "list" # Could also be tuple, but list seems more common given the inputs argument
}
signatures["torch.nn.parallel.replicate"] = {
    "module": "module", # Assuming 'module' refers to a nn.Module instance
    "device_ids": "list" # It could also be a tuple of integers
}
signatures["torch.nn.parallel.scatter"] = {
    "inputs": "tensor_list", # Could potentially also accept a tensor, but tensor_list seems more common from usage.
    "target_gpus": "list",
    "dim": "integer"
}
signatures["torch.nn.parallel.scatter_gather"] = {
    "scatter": "list",
    "gather": "list",
    "target_gpus": "list",
    "dim": "integer"
}
signatures["torch.nn.parameter.Parameter"] = {
    "data": "tensor",
    "requires_grad": "boolean"
}
signatures["torch.nn.qat.Linear"] = {
    "in_features": "integer",
    "out_features": "integer",
    "bias": "boolean"
}
signatures["torch.nn.qat.Conv2d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # Can also be integer, but tuple is more common
    "stride": "tuple", # Can also be integer, but tuple is more common
    "padding": "tuple", # Can also be integer, but tuple is more common
    "dilation": "tuple", # Can also be integer, but tuple is more common
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.qat.Conv3d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # Can also be integer, but tuple is more common
    "stride": "tuple", # Can also be integer, but tuple is more common
    "padding": "tuple", # Can also be integer, but tuple is more common
    "dilation": "tuple", # Can also be integer, but tuple is more common
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.qat.modules.conv.Conv2d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # Can also be integer, but tuple is more common
    "stride": "tuple", # Can also be integer, but tuple is more common
    "padding": "tuple", # Can also be integer, but tuple is more common
    "dilation": "tuple", # Can also be integer, but tuple is more common
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.qat.modules.linear.Linear"] = {
    "in_features": "integer",
    "out_features": "integer",
    "bias": "boolean"
}
signatures["torch.nn.qat.Conv1d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "integer", # Could also be tuple
    "stride": "integer", # Could also be tuple
    "padding": "integer", # Could also be tuple
    "dilation": "integer", # Could also be tuple
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.qat.Conv2d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # can be integer as well
    "stride": "tuple", # can be integer as well
    "padding": "tuple", # can be integer or string as well
    "dilation": "tuple", # can be integer as well
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.qat.Conv3d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # Can be integer or tuple, assuming tuple is more common
    "stride": "tuple", # Can be integer or tuple, assuming tuple is more common
    "padding": "tuple", # Can be string, integer, or tuple, assuming tuple is more common
    "dilation": "tuple", # Can be integer or tuple, assuming tuple is more common
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string",
    "qconfig": "object" #This should ideally be more specific. Maybe "QConfig" if that existed
}
signatures["torch.nn.qat.Embedding"] = {
    "num_embeddings": "integer",
    "embedding_dim": "integer",
    "padding_idx": "integer",
    "max_norm": "float",
    "norm_type": "float",
    "scale_grad_by_freq": "boolean",
    "sparse": "boolean",
    "_weight": "tensor" # Originally a parameter, but we represent that as a tensor
}
signatures["torch.nn.qat.EmbeddingBag"] = {
    "num_embeddings": "integer",
    "embedding_dim": "integer",
    "max_norm": "float",
    "norm_type": "float",
    "scale_grad_by_freq": "boolean",
    "mode": "string",
    "sparse": "boolean",
    "_weight": "tensor", # Should this be a Parameter?
    "include_last_offset": "boolean",
    "padding_idx": "integer" #Potentially Optional[int]
}
signatures["torch.nn.qat.Linear"] = {
    "in_features": "integer",
    "out_features": "integer",
    "bias": "boolean"
}
signatures["torch.nn.qat.dynamic.QuantLinear"] = {
    "in_features": "integer",
    "out_features": "integer",
    "bias": "boolean",
    "qconfig": "object" # unsure, likely an object
}
signatures["torch.nn.qat.dynamic.QuantConv2d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # can also be integer, using most common
    "stride": "tuple", # can also be integer, using most common
    "padding": "tuple", # can also be integer, using most common
    "dilation": "tuple", # can also be integer, using most common
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string",
    "qconfig": "object" # unsure, likely an object
}
signatures["torch.nn.qat.dynamic.Linear"] = {
    "in_features": "integer",
    "out_features": "integer",
    "bias": "boolean"
}
signatures["torch.nn.qat.dynamic.modules.Conv2d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # Could also be an integer, but tuple is more common
    "stride": "tuple", # Could also be an integer, but tuple is more common
    "padding": "tuple", # Could also be an integer, but tuple is more common
    "dilation": "tuple", # Could also be an integer, but tuple is more common
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.qat.dynamic.modules.Linear"] = {
    "in_features": "integer",
    "out_features": "integer",
    "bias": "boolean"
}
signatures["torch.nn.qat.dynamic.modules.Linear"] = {
    "in_features": "integer",
    "out_features": "integer",
    "bias": "boolean"
}
signatures["torch.nn.qat.dynamic.modules.linear"] = {
    "in_features": "integer",
    "out_features": "integer",
    "bias": "boolean"
}
signatures["torch.nn.qat.modules.Conv2d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # or integer
    "stride": "tuple", # or integer
    "padding": "tuple", # or string or integer
    "dilation": "tuple", # or integer
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.qat.modules.Linear"] = {
    "in_features": "integer",
    "out_features": "integer",
    "bias": "boolean"
}
signatures["torch.nn.qat.modules.Conv1d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "integer", # Could also be tuple
    "stride": "integer", # Could also be tuple
    "padding": "integer", # Could also be tuple
    "dilation": "integer", # Could also be tuple
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.qat.modules.Conv2d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # could be tuple or integer, chose tuple because it's likely more common
    "stride": "tuple", # could be tuple or integer, chose tuple because it's likely more common
    "padding": "tuple", # could be tuple or integer, chose tuple because it's likely more common
    "dilation": "tuple", # could be tuple or integer, chose tuple because it's likely more common
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string",
    "qconfig": "dtype" # unsure if this is the most accurate type, but represents the configuration
}
signatures["torch.nn.qat.modules.Conv3d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # could also be integer or list
    "stride": "tuple", # could also be integer or list
    "padding": "tuple", # could also be integer or list
    "dilation": "tuple", # could also be integer or list
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.qat.modules.Embedding"] = {
    "num_embeddings": "integer",
    "embedding_dim": "integer",
    "padding_idx": "integer",
    "max_norm": "float",
    "norm_type": "float",
    "scale_grad_by_freq": "boolean",
    "sparse": "boolean",
    "_weight": "tensor" # Could be None, but tensor is the more common case
}
signatures["torch.nn.qat.modules.EmbeddingBag"] = {
    "num_embeddings": "integer",
    "embedding_dim": "integer",
    "max_norm": "float",
    "norm_type": "float",
    "scale_grad_by_freq": "boolean",
    "mode": "string",
    "sparse": "boolean",
    "weight": "tensor",
    "include_last_offset": "boolean",
    "padding_idx": "integer", #Could also be None, but integer is more common
    "_weight_fake_quant": "string" # this should ideally be a type of quantizer, but we don't have that
}
signatures["torch.nn.qat.modules.Linear"] = {
    "in_features": "integer",
    "out_features": "integer",
    "bias": "boolean"
}
signatures["torch.nn.qat.modules.conv.Conv2d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # Can also be integer, but tuple is more common for multiple dimensions
    "stride": "tuple", # Can also be integer
    "padding": "tuple", # Can also be integer
    "dilation": "tuple", # Can also be integer
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.qat.modules.conv.Conv3d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # Can also be integer, but tuple is more common for multiple dimensions
    "stride": "tuple", # Can also be integer
    "padding": "tuple", # Can also be integer
    "dilation": "tuple", # Can also be integer
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.qat.modules.embedding_ops"] = {
    "num_embeddings": "integer",
    "embedding_dim": "integer",
    "padding_idx": "integer",
    "max_norm": "float",
    "norm_type": "float",
    "scale_grad_by_freq": "boolean",
    "sparse": "boolean",
    "weight": "tensor"
}
signatures["torch.nn.qat.modules.linear"] = {
    'in_features': 'integer',
    'out_features': 'integer',
    'bias': 'boolean'
}
signatures["torch.nn.quantizable.QuantStub"] = {}
signatures["torch.nn.quantizable.DeQuantStub"] = {}
signatures["torch.nn.quantizable.Quantize"] = {
    "scale": "float",
    "zero_point": "integer",
    "quant_min": "integer",
    "quant_max": "integer",
    "dtype": "dtype"
}
signatures["torch.nn.quantizable.DeQuantize"] = {}
signatures["torch.nn.quantizable.LSTM"] = {
    "input_size": "integer",
    "hidden_size": "integer",
    "num_layers": "integer",
    "bias": "boolean",
    "batch_first": "boolean",
    "dropout": "float",
    "bidirectional": "boolean",
    "proj_size": "integer"
}
signatures["torch.nn.quantizable.LSTMCell"] = {
    "input_size": "integer",
    "hidden_size": "integer",
    "bias": "boolean",
    "dtype": "dtype" # could be more specific, maybe list of dtypes if that was an option
}
signatures["torch.nn.quantizable.MultiheadAttention"] = {
    "embed_dim": "integer",
    "num_heads": "integer",
    "dropout": "float",
    "bias": "boolean",
    "add_bias_kv": "boolean",
    "add_zero_attn": "boolean",
    "kdim": "integer",
    "vdim": "integer",
    "batch_first": "boolean"
}
signatures["torch.nn.quantizable.modules.FloatFunctional"] = {}
signatures["torch.nn.quantizable.modules.FloatFunctional.add"] = {
    "other": "tensor"
}
signatures["torch.nn.quantizable.modules.FloatFunctional.mul"] = {
    "other": "tensor"
}
signatures["torch.nn.quantizable.modules.FloatFunctional.cat"] = {
    "tensors": "tensor_list",
    "dim": "integer"
}
signatures["torch.nn.quantizable.modules.FloatFunctional.add_scalar"] = {
    "scalar": "float"
}
signatures["torch.nn.quantizable.modules.FloatFunctional.mul_scalar"] = {
    "scalar": "float"
}
signatures["torch.nn.quantizable.modules.FloatFunctional.scalar_add"] = {
    "scalar": "float"
}
signatures["torch.nn.quantizable.modules.FloatFunctional.hardtanh"] = {
    "min_val": "float",
    "max_val": "float",
    "inplace": "boolean"
}
signatures["torch.nn.quantizable.modules.FloatFunctional.relu"] = {
    "inplace": "boolean"
}
signatures["torch.nn.quantizable.modules.FloatFunctional.adaptive_avg_pool2d"] = {
    "output_size": "tuple" # Could also be integer, but tuple is more common.
}
signatures["torch.nn.quantizable.modules.FloatFunctional.avg_pool2d"] = {
    "kernel_size": "tuple", # Could also be integer
    "stride": "tuple", # Could also be integer
    "padding": "tuple", # Could also be integer
    "ceil_mode": "boolean",
    "count_include_pad": "boolean",
    "divisor_override": "integer"
}
signatures["torch.nn.quantizable.modules.FloatFunctional.interpolate"] = {
    "size": "tuple", #Could be list or None
    "scale_factor": "float", #Could be tuple, list or None
    "mode": "string",
    "align_corners": "boolean",
    "recompute_scale_factor": "boolean"
}
signatures["torch.nn.quantizable.modules.QFunctional"] = {}
signatures["torch.nn.quantizable.modules.QFunctional.add"] = {
    "x": "tensor",
    "y": "tensor",
    "scale": "float",
    "zero_point": "integer"
}
signatures["torch.nn.quantizable.modules.QFunctional.mul"] = {
    "x": "tensor",
    "y": "tensor",
    "scale": "float",
    "zero_point": "integer"
}
signatures["torch.nn.quantizable.modules.Conv2d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple",  # Can also be an int
    "stride": "tuple", # Can also be an int
    "padding": "tuple", # Can also be an int
    "dilation": "tuple", # Can also be an int
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.quantizable.modules.Linear"] = {
    "in_features": "integer",
    "out_features": "integer",
    "bias": "boolean"
}
signatures["torch.nn.quantizable.modules.LSTM"] = {
    "input_size": "integer",
    "hidden_size": "integer",
    "num_layers": "integer",
    "bias": "boolean",
    "batch_first": "boolean",
    "dropout": "float",
    "bidirectional": "boolean"
}
signatures["torch.nn.quantizable.modules.LSTM"] = {
    "input_size": "integer",
    "hidden_size": "integer",
    "num_layers": "integer",
    "bias": "boolean",
    "batch_first": "boolean",
    "dropout": "float",
    "bidirectional": "boolean",
    "proj_size": "integer"
}
signatures["torch.nn.quantizable.modules.LSTMCell"] = {
    "input": "tensor",
    "hx": "tuple", # Could be a tuple of tensors, but tuple seems more appropriate as a single type
    "cx": "tensor",
    "w_ih": "tensor",
    "w_hh": "tensor",
    "b_ih": "tensor", # Optional, defaults to None
    "b_hh": "tensor"  # Optional, defaults to None
}
signatures["torch.nn.quantizable.modules.MultiheadAttention"] = {
    "embed_dim_to_check": "integer",
    "num_heads": "integer",
    "dropout": "float",
    "bias": "boolean",
    "add_bias_kv": "boolean",
    "add_zero_attn": "boolean",
    "kdim": "integer",
    "vdim": "integer",
    "batch_first": "boolean",
    "device": "string", # Should be removed according to prompt, but it's in the init
    "dtype": "dtype" # Should be removed according to prompt, but it's in the init
}
signatures["torch.nn.quantized.FloatFunctional"] = {} # This class doesn't have a commonly used signature in the constructor.
signatures["torch.nn.quantized.QFunctional"] = {} # This class doesn't have a commonly used signature in the constructor.
signatures["torch.nn.quantized.Quantize"] = {
    "scale": "float",
    "zero_point": "integer",
    "dtype": "dtype"
}
signatures["torch.nn.quantized.DeQuantize"] = {} # This class doesn't have a commonly used signature in the constructor.
signatures["torch.nn.quantized.Linear"] = {
    "in_features": "integer",
    "out_features": "integer",
    "bias": "boolean"
}
signatures["torch.nn.quantized.Conv2d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # could also be an integer
    "stride": "tuple", # could also be an integer
    "padding": "tuple", # could also be an integer
    "dilation": "tuple", # could also be an integer
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.quantized.BatchNorm2d"] = {
    "num_features": "integer",
    "eps": "float",
    "momentum": "float",
    "affine": "boolean",
    "track_running_stats": "boolean",
    "quantize": "boolean" # I'm not 100% sure about this one, but I think it makes sense to be boolean
}
signatures["torch.nn.quantized.BatchNorm3d"] = {
    "num_features": "integer",
    "eps": "float",
    "momentum": "float",
    "affine": "boolean",
    "track_running_stats": "boolean"
}
signatures["torch.nn.quantized.Conv1d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "integer", # Could also be a tuple, but integer is more common
    "stride": "integer", # Could also be a tuple, but integer is more common
    "padding": "integer", # Could also be a tuple, but integer is more common
    "dilation": "integer", # Could also be a tuple, but integer is more common
    "groups": "integer",
    "padding_mode": "string",
    "qconfig": "object" # There isn't a perfect match for qconfig, which is a QuantizationConfig object
}
signatures["torch.nn.quantized.Conv2d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # Could also be integer
    "stride": "tuple", # Could also be integer
    "padding": "tuple", # Could also be integer
    "dilation": "tuple", # Could also be integer
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.quantized.Conv3d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # Can also be integer or tuple of integers
    "stride": "tuple", # Can also be integer or tuple of integers
    "padding": "tuple", # Can also be integer or tuple of integers
    "dilation": "tuple", # Can also be integer or tuple of integers
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string",
    "qconfig": "object" # This should ideally be a class type, but we don't have that option, so leaving it as object
}
signatures["torch.nn.quantized.ConvTranspose1d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "integer", # Could be tuple, but integer is more common
    "stride": "integer", # Could be tuple, but integer is more common
    "padding": "integer", # Could be tuple, but integer is more common
    "output_padding": "integer", # Could be tuple, but integer is more common
    "groups": "integer",
    "dilation": "integer",
    "bias": "boolean",
    "padding_mode": "string",
    "qconfig": "string" # Ideally this would be a specific object type, but string is the closest
}
signatures["torch.nn.quantized.ConvTranspose2d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple",  # Can also be an integer, but tuple is more common for 2D conv
    "stride": "tuple",  # Can also be an integer, but tuple is more common for 2D conv
    "padding": "tuple",  # Can also be an integer, but tuple is more common for 2D conv
    "output_padding": "tuple",  # Can also be an integer, but tuple is more common for 2D conv
    "dilation": "integer",
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string",
    "qconfig": "string" # I am unsure about the exact type here. It is a quantization configuration, but string is the closest.
}
signatures["torch.nn.quantized.ConvTranspose3d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # Could also be integer. Chose the most common type
    "stride": "tuple", # Could also be integer. Chose the most common type
    "padding": "tuple", # Could also be integer. Chose the most common type
    "output_padding": "tuple", # Could also be integer. Chose the most common type
    "dilation": "tuple", # Could also be integer. Chose the most common type
    "groups": "integer",
    "padding_mode": "string",
    "weight": "tensor",
    "bias": "tensor",
    "scale": "float",
    "zero_point": "integer"
}
signatures["torch.nn.quantized.DeQuantize"] = {
    "": "tensor" # The input is a tensor, and the output is also a tensor. There are no explicit arguments to the constructor other than the input tensor itself. I am not sure if I should name this argument "input".
}
signatures["torch.nn.quantized.Dropout"] = {
    "p": "float", # Could be argued as "float" or "integer", float seems more common
    "inplace": "boolean"
}
signatures["torch.nn.quantized.ELU"] = {
    "alpha": "float",
    "scale": "float",
    "zero_point": "integer",
    "q_dtype": "dtype" # Could also be string if referring to the name of the dtype
}
signatures["torch.nn.quantized.Embedding"] = {
    "num_embeddings": "integer",
    "embedding_dim": "integer",
    "padding_idx": "integer",
    "scale": "float", # could potentially be tensor, but float seems more common
    "zero_point": "integer" # could potentially be tensor, but integer seems more common
}
signatures["torch.nn.quantized.EmbeddingBag"] = {
    "num_embeddings": "integer",
    "embedding_dim": "integer",
    "padding_idx": "integer",
    "scale_grad_by_freq": "boolean",
    "mode": "string",
    "sparse": "boolean",
    "_weight": "tensor" # Should this be "tensor"? It's a parameter, which is a tensor.
}
signatures["torch.nn.quantized.FXFloatFunctional.add"] = {
    "input": "tensor",
    "other": "tensor" # or "float", depending on the common usage. Assuming tensor for generality
}
signatures["torch.nn.quantized.FXFloatFunctional.mul"] = {
    "input": "tensor",
    "other": "tensor" # or "float", depending on the common usage. Assuming tensor for generality
}
signatures["torch.nn.quantized.FXFloatFunctional.cat"] = {
    "tensors": "tensor_list",
    "dim": "integer"
}
signatures["torch.nn.quantized.FXFloatFunctional.linear"] = {
    "input": "tensor",
    "weight": "tensor",
    "bias": "tensor"
}
signatures["torch.nn.quantized.FXFloatFunctional.relu"] = {
    "input": "tensor"
}
signatures["torch.nn.quantized.FXFloatFunctional.hardtanh"] = {
    "input": "tensor",
    "min_val": "float",
    "max_val": "float"
}
signatures["torch.nn.quantized.FXFloatFunctional.adaptive_avg_pool2d"] = {
    "input": "tensor",
    "output_size": "tuple" # or "integer", depending on the common usage
}
signatures["torch.nn.quantized.FXFloatFunctional.avg_pool2d"] = {
    "input": "tensor",
    "kernel_size": "tuple", # or "integer", depending on the common usage
    "stride": "tuple", # or "integer", depending on the common usage
    "padding": "tuple", # or "integer", depending on the common usage
    "ceil_mode": "boolean",
    "count_include_pad": "boolean",
    "divisor_override": "float"
}
signatures["torch.nn.quantized.FXFloatFunctional.max_pool2d"] = {
    "input": "tensor",
    "kernel_size": "tuple", # or "integer", depending on the common usage
    "stride": "tuple", # or "integer", depending on the common usage
    "padding": "tuple", # or "integer", depending on the common usage
    "dilation": "tuple", # or "integer", depending on the common usage
    "ceil_mode": "boolean"
}
signatures["torch.nn.quantized.FXFloatFunctional.interpolate"] = {
    "input": "tensor",
    "size": "tuple", # Can also accept "int" or None, but tuple seems like a more common use case
    "scale_factor": "float", # Can also accept tuple
    "mode": "string",
    "align_corners": "boolean",
    "recompute_scale_factor": "boolean"
}
signatures["torch.nn.quantized.FXFloatFunctional.batch_norm"] = {
    "input": "tensor",
    "weight": "tensor",
    "bias": "tensor",
    "running_mean": "tensor",
    "running_var": "tensor",
    "training": "boolean",
    "momentum": "float",
    "eps": "float"
}
signatures["torch.nn.quantized.FXFloatFunctional.layer_norm"] = {
    "input": "tensor",
    "normalized_shape": "list",
    "weight": "tensor",
    "bias": "tensor",
    "eps": "float"
}
signatures["torch.nn.quantized.FXFloatFunctional.group_norm"] = {
    "input": "tensor",
    "num_groups": "integer",
    "weight": "tensor",
    "bias": "tensor",
    "eps": "float"
}
signatures["torch.nn.quantized.FloatFunctional.add_scalar"] = {
    "input": "tensor",
    "scalar": "float"
}
signatures["torch.nn.quantized.FloatFunctional.add"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.nn.quantized.FloatFunctional.mul_scalar"] = {
    "input": "tensor",
    "scalar": "float"
}
signatures["torch.nn.quantized.FloatFunctional.mul"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.nn.quantized.FloatFunctional.cat"] = {
    "tensors": "tensor_list",
    "dim": "integer"
}
signatures["torch.nn.quantized.FloatFunctional.relu"] = {
    "input": "tensor"
}
signatures["torch.nn.quantized.FloatFunctional.hardtanh"] = {
    "input": "tensor",
    "min_val": "float",
    "max_val": "float"
}
signatures["torch.nn.quantized.FloatFunctional.adaptive_avg_pool2d"] = {
    "input": "tensor",
    "output_size": "tuple" # Could be tuple or integer, tuple is more common for output_size
}
signatures["torch.nn.quantized.FloatFunctional.avg_pool2d"] = {
    "input": "tensor",
    "kernel_size": "tuple", #Could be tuple or integer, tuple is more common
    "stride": "tuple", #Could be tuple or integer, tuple is more common
    "padding": "tuple", #Could be tuple or integer, tuple is more common
    "ceil_mode": "boolean",
    "count_include_pad": "boolean",
    "divisor_override": "integer" #Could be None, but integer is more common
}
signatures["torch.nn.quantized.FloatFunctional.interpolate"] = {
    "input": "tensor",
    "size": "list", # size, scale_factor, mode, align_corners, recompute_scale_factor - size is most common
    "scale_factor": "float", # this value will not be used as we chose size
    "mode": "string",
    "align_corners": "boolean",
    "recompute_scale_factor": "boolean"
}
signatures["torch.nn.quantized.GroupNorm"] = {
    "num_groups": "integer",
    "num_channels": "integer",
    "eps": "float",
    "affine": "boolean",
}
signatures["torch.nn.quantized.Hardswish"] = {
}
signatures["torch.nn.quantized.InstanceNorm1d"] = {
    "num_features": "integer",
    "eps": "float",
    "momentum": "float",
    "affine": "boolean",
    "track_running_stats": "boolean"
}
signatures["torch.nn.quantized.InstanceNorm2d"] = {
    "num_features": "integer",
    "eps": "float",
    "momentum": "float",
    "affine": "boolean",
    "track_running_stats": "boolean"
}
signatures["torch.nn.quantized.InstanceNorm3d"] = {
    "num_features": "integer",
    "eps": "float",
    "momentum": "float",
    "affine": "boolean",
    "track_running_stats": "boolean"
}
signatures["torch.nn.quantized.LSTM"] = {
    "input_size": "integer",
    "hidden_size": "integer",
    "num_layers": "integer",
    "bias": "boolean",
    "batch_first": "boolean",
    "dropout": "float",
    "bidirectional": "boolean",
    "proj_size": "integer"
}
signatures["torch.nn.quantized.LayerNorm"] = {
    "normalized_shape": "list", # Could be tuple too, but list seems more common in examples
    "eps": "float",
    "elementwise_affine": "boolean"
}
signatures["torch.nn.quantized.LeakyReLU"] = {
    "negative_slope": "float",
    "inplace": "boolean"
}
signatures["torch.nn.quantized.Linear"] = {
    "in_features": "integer",
    "out_features": "integer",
    "bias": "boolean"
}
signatures["torch.nn.quantized.MaxPool2d"] = {
    "kernel_size": "integer", # Can also be tuple, choosing integer as it's likely more common
    "stride": "integer", # Can also be tuple, choosing integer as it's likely more common
    "padding": "integer", # Can also be tuple, choosing integer as it's likely more common
    "dilation": "integer", # Can also be tuple, choosing integer as it's likely more common
    "ceil_mode": "boolean"
}
signatures["torch.nn.quantized.MultiheadAttention"] = {
    "embed_dim": "integer",
    "num_heads": "integer",
    "dropout": "float",
    "bias": "boolean",
    "add_bias_kv": "boolean",
    "add_zero_attn": "boolean",
    "kdim": "integer",
    "vdim": "integer"
}
signatures["torch.nn.quantized.PReLU"] = {
    "num_parameters": "integer",
    "init": "float"
}
signatures["torch.nn.quantized.QFunctional"] = {} # QFunctional doesn't have arguments in the constructor
signatures["torch.ops.quantized.add_relu"] = {
    "x": "tensor",
    "y": "tensor",
    "scale": "float",
    "zero_point": "integer"
}
signatures["torch.ops.quantized.add"] = {
    "x": "tensor",
    "y": "tensor",
    "scale": "float",
    "zero_point": "integer"
}
signatures["torch.ops.quantized.mul"] = {
    "x": "tensor",
    "y": "tensor",
    "scale": "float",
    "zero_point": "integer"
}
signatures["torch.ops.quantized.conv2d"] = {
    "input": "tensor",
    "weight": "tensor",
    "bias": "tensor", # bias can be None, but it's a tensor in the common case
    "stride": "list", # Could also be tuple or integer, but list is more general
    "padding": "list",
    "dilation": "list",
    "groups": "integer",
    "scale": "float",
    "zero_point": "integer"
}
signatures["torch.ops.quantized.linear"] = {
    "input": "tensor",
    "weight": "tensor",
    "bias": "tensor",
    "scale": "float",
    "zero_point": "integer"
}
signatures["torch.nn.quantized.Quantize"] = {
    "module": "dtype" # Should this be a torch.nn.Module type?
}
signatures["torch.nn.quantized.ReLU6"] = {
    "inplace": "boolean"
}
signatures["torch.nn.quantized.Sigmoid"] = {
}
signatures["torch.nn.quantized.Softmax"] = {
    "dim": "integer",
    "dtype": "dtype" # could also be None but specifying dtype as most common
}
signatures["torch.nn.quantized.dynamic.Linear"] = {
    "in_features": "integer",
    "out_features": "integer",
    "bias": "boolean"
}
signatures["torch.nn.quantized.dynamic.LSTM"] = {
    "input_size": "integer",
    "hidden_size": "integer",
    "num_layers": "integer",
    "bias": "boolean",
    "batch_first": "boolean",
    "dropout": "float",
    "bidirectional": "boolean"
}
signatures["torch.nn.quantized.dynamic.GRU"] = {
    "input_size": "integer",
    "hidden_size": "integer",
    "num_layers": "integer",
    "bias": "boolean",
    "batch_first": "boolean",
    "dropout": "float",
    "bidirectional": "boolean"
}
signatures["torch.nn.quantized.dynamic.RNN"] = {
    "input_size": "integer",
    "hidden_size": "integer",
    "num_layers": "integer",
    "nonlinearity": "string",
    "bias": "boolean",
    "batch_first": "boolean",
    "dropout": "float",
    "bidirectional": "boolean"
}
signatures["torch.nn.quantized.dynamic.Conv1d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "integer", # Could also be tuple, but integer seems more common
    "stride": "integer", # Could also be tuple, but integer seems more common
    "padding": "integer", # Could also be tuple, but integer seems more common
    "dilation": "integer", # Could also be tuple, but integer seems more common
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.quantized.dynamic.Conv2d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # Could also be integer but tuple is more common for kernel size
    "stride": "tuple", # Could also be integer but tuple is more common
    "padding": "tuple", # Could also be integer but tuple is more common
    "dilation": "tuple", # Could also be integer but tuple is more common
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.quantized.dynamic.Conv3d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # Could be integer as well
    "stride": "tuple", # Could be integer as well
    "padding": "tuple", # Could be integer or string as well
    "dilation": "tuple", # Could be integer as well
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.quantized.dynamic.ConvTranspose1d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "integer", # Could also be tuple
    "stride": "integer", # Could also be tuple
    "padding": "integer", # Could also be tuple
    "output_padding": "integer", # Could also be tuple
    "groups": "integer",
    "bias": "boolean",
    "dilation": "integer", # Could also be tuple
    "padding_mode": "string"
}
signatures["torch.nn.quantized.dynamic.ConvTranspose2d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # Can also be integer
    "stride": "tuple", # Can also be integer
    "padding": "tuple", # Can also be integer
    "output_padding": "tuple", # Can also be integer
    "groups": "integer",
    "bias": "boolean",
    "dilation": "tuple", # Can also be integer
    "padding_mode": "string"
}
signatures["torch.nn.quantized.dynamic.ConvTranspose3d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # or integer
    "stride": "tuple", # or integer
    "padding": "tuple", # or integer
    "output_padding": "tuple", # or integer
    "dilation": "tuple", # or integer
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.quantized.dynamic.GRU"] = {
    "input_size": "integer",
    "hidden_size": "integer",
    "num_layers": "integer",
    "bias": "boolean",
    "batch_first": "boolean",
    "dropout": "float",
    "bidirectional": "boolean"
}
signatures["torch.nn.quantized.dynamic.GRUCell"] = {
    "input": "tensor",
    "hx": "tensor"
}
signatures["torch.nn.quantized.dynamic.LSTM"] = {
    "input_size": "integer",
    "hidden_size": "integer",
    "num_layers": "integer",
    "bias": "boolean",
    "batch_first": "boolean",
    "dropout": "float",
    "bidirectional": "boolean"
}
signatures["torch.nn.quantized.dynamic.LSTMCell"] = {
    "input_size": "integer",
    "hidden_size": "integer",
    "bias": "boolean",
    "dtype": "dtype" # Could also be string depending on how it's used
}
signatures["torch.nn.quantized.dynamic.Linear"] = {
    "in_features": "integer",
    "out_features": "integer",
    "bias": "boolean" # Could be a tensor but most commonly a boolean
}
signatures["torch.nn.quantized.dynamic.RNNCell"] = {
    "input_size": "integer",
    "hidden_size": "integer",
    "bias": "boolean",
    "nonlinearity": "string" # Could be an enum of available non-linearities. String seems like the closest match.
}
signatures["torch.nn.quantized.dynamic.Linear"] = {
    "in_features": "integer",
    "out_features": "integer",
    "bias": "boolean"
}
signatures["torch.nn.quantized.dynamic.LSTM"] = {
    "input_size": "integer",
    "hidden_size": "integer",
    "num_layers": "integer",
    "bias": "boolean",
    "batch_first": "boolean",
    "dropout": "float",
    "bidirectional": "boolean"
}
signatures["torch.nn.quantized.dynamic.GRU"] = {
    "input_size": "integer",
    "hidden_size": "integer",
    "num_layers": "integer",
    "bias": "boolean",
    "batch_first": "boolean",
    "dropout": "float",
    "bidirectional": "boolean"
}
signatures["torch.nn.quantized.dynamic.modules.Conv1d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "integer", # Could also be tuple
    "stride": "integer", # Could also be tuple
    "padding": "integer", # Could also be tuple
    "dilation": "integer", # Could also be tuple
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.quantized.dynamic.modules.Conv2d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # Could also be integer, but tuple is more common
    "stride": "tuple", # Could also be integer
    "padding": "tuple", # Could also be integer
    "dilation": "tuple", # Could also be integer
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.quantized.dynamic.modules.Conv3d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # Could also be integer, but tuple is more common
    "stride": "tuple", # Could also be integer, but tuple is more common
    "padding": "tuple", # Could also be integer, but tuple is more common
    "dilation": "tuple", # Could also be integer, but tuple is more common
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.quantized.dynamic.modules.ConvTranspose1d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "integer", # Could also be tuple
    "stride": "integer", # Could also be tuple
    "padding": "integer", # Could also be tuple
    "output_padding": "integer", # Could also be tuple
    "groups": "integer",
    "bias": "boolean",
    "dilation": "integer", # Could also be tuple
    "padding_mode": "string"
}
signatures["torch.nn.quantized.dynamic.modules.ConvTranspose2d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # Could also be integer
    "stride": "tuple", # Could also be integer
    "padding": "tuple", # Could also be integer
    "output_padding": "tuple", # Could also be integer
    "groups": "integer",
    "bias": "boolean",
    "dilation": "tuple", # Could also be integer
    "padding_mode": "string"
}
signatures["torch.nn.quantized.dynamic.modules.ConvTranspose3d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # Could also be integer, but tuple is more common
    "stride": "tuple", # Could also be integer, but tuple is more common
    "padding": "tuple", # Could also be integer, but tuple is more common
    "output_padding": "tuple", # Could also be integer, but tuple is more common
    "groups": "integer",
    "bias": "boolean",
    "dilation": "tuple", # Could also be integer, but tuple is more common
    "padding_mode": "string"
}
signatures["torch.nn.quantized.dynamic.modules.GRU"] = {
    "input_size": "integer",
    "hidden_size": "integer",
    "num_layers": "integer",
    "bias": "boolean",
    "batch_first": "boolean",
    "dropout": "float",
    "bidirectional": "boolean",
    "dtype": "dtype" # Could also be none
}
signatures["torch.nn.quantized.dynamic.modules.GRUCell"] = {
    "input": "tensor",
    "hx": "tensor",
    "weight_ih": "tensor",
    "weight_hh": "tensor",
    "bias_ih": "tensor",
    "bias_hh": "tensor"
}
signatures["torch.nn.quantized.dynamic.modules.LSTM"] = {
    "input_size": "integer",
    "hidden_size": "integer",
    "num_layers": "integer",
    "bias": "boolean",
    "batch_first": "boolean",
    "dropout": "float",
    "bidirectional": "boolean",
    "proj_size": "integer" # Could be None, but int is more common if not none
}
signatures["torch.nn.quantized.dynamic.modules.LSTMCell"] = {
    "input": "tensor",
    "hx": "tuple",
    "weight_ih": "tensor",
    "weight_hh": "tensor",
    "bias_ih": "tensor",
    "bias_hh": "tensor"
}
signatures["torch.nn.quantized.dynamic.modules.Linear"] = {
    "in_features": "integer",
    "out_features": "integer",
    "bias": "boolean"
}
signatures["torch.nn.quantized.dynamic.modules.RNNCell"] = {
    "input_size": "integer",
    "hidden_size": "integer",
    "bias": "boolean",
    "dtype": "dtype" # Could also be a more specific dtype like torch.qint8 or torch.quint8 but "dtype" seems more general
}
signatures["torch.nn.quantized.dynamic.modules.conv"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # Can also be integer, but tuple is more common
    "stride": "tuple", # Can also be integer, but tuple is more common
    "padding": "tuple", # Can also be integer, but tuple is more common
    "dilation": "tuple", # Can also be integer, but tuple is more common
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.quantized.dynamic.modules.linear"] = {
    "in_features": "integer",
    "out_features": "integer",
    "bias": "boolean"
}
signatures["torch.nn.quantized.dynamic.modules.rnn"] = {
    "input_size": "integer",
    "hidden_size": "integer",
    "num_layers": "integer",
    "nonlinearity": "string",
    "bias": "boolean",
    "batch_first": "boolean",
    "dropout": "float",
    "bidirectional": "boolean",
    "proj_size": "integer",
    "quantize": "boolean" # I'm not sure if this should be boolean or not. It depends on the context of whether this parameter controls quantization or is a quantization config.
}
signatures["torch.nn.quantized.functional.conv2d"] = {
    "input": "tensor",
    "weight": "tensor",
    "bias": "tensor",
    "stride": "tuple", # could be integer too
    "padding": "tuple", # could be integer too
    "dilation": "tuple", # could be integer too
    "groups": "integer",
    "scale": "float",
    "zero_point": "integer"
}
signatures["torch.nn.quantized.functional.linear"] = {
    "input": "tensor",
    "weight": "tensor",
    "bias": "tensor",
    "scale": "float",
    "zero_point": "integer"
}
signatures["torch.nn.quantized.functional.relu"] = {
    "input": "tensor",
    "scale": "float",
    "zero_point": "integer"
}
signatures["torch.nn.quantized.modules.Conv2d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # could also be int, but tuple more common
    "stride": "tuple", # could also be int, but tuple more common
    "padding": "tuple", # could also be int, but tuple more common
    "dilation": "tuple", # could also be int, but tuple more common
    "groups": "integer",
    "padding_mode": "string"
}
signatures["torch.nn.quantized.modules.Linear"] = {
    "in_features": "integer",
    "out_features": "integer"
}
signatures["torch.nn.quantized.modules.ReLU"] = {}
signatures["torch.nn.quantized.modules.ReLU6"] = {}
signatures["torch.nn.quantized.modules.MaxPool2d"] = {
    "kernel_size": "tuple", # could also be int, but tuple more common
    "stride": "tuple", # could also be int, but tuple more common
    "padding": "tuple", # could also be int, but tuple more common
    "dilation": "tuple", # could also be int, but tuple more common
    "return_indices": "boolean",
    "ceil_mode": "boolean"
}
signatures["torch.nn.quantized.modules.AvgPool2d"] = {
    "kernel_size": "tuple", # could also be int, but tuple more common
    "stride": "tuple", # could also be int, but tuple more common
    "padding": "tuple", # could also be int, but tuple more common
    "ceil_mode": "boolean",
    "count_include_pad": "boolean",
    "divisor_override": "integer" # Could also be NoneType, but int seems more common
}
signatures["torch.nn.quantized.modules.functional.conv2d"] = {
    "input": "tensor",
    "weight": "tensor",
    "bias": "tensor",
    "stride": "tuple",
    "padding": "tuple",
    "dilation": "tuple",
    "groups": "integer"
}
signatures["torch.nn.quantized.modules.BatchNorm2d"] = {
    "num_features": "integer"
}
signatures["torch.nn.quantized.modules.BatchNorm2d"] = {
    "num_features": "integer",
    "eps": "float",
    "momentum": "float",
    "affine": "boolean",
    "track_running_stats": "boolean"
}
signatures["torch.nn.quantized.modules.BatchNorm3d"] = {
    "num_features": "integer",
    "eps": "float",
    "momentum": "float",
    "affine": "boolean",
    "track_running_stats": "boolean"
}
signatures["torch.nn.quantized.modules.Conv1d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "integer", # Could also be tuple
    "stride": "integer", # Could also be tuple
    "padding": "integer", # Could also be tuple
    "dilation": "integer", # Could also be tuple
    "groups": "integer",
    "padding_mode": "string",
    "qconfig": "dtype" # Ideally a quantization config object, representing as dtype for simplicity.
}
signatures["torch.nn.quantized.modules.Conv2d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # Could also be integer, but tuple is more common
    "stride": "tuple", # Could also be integer, but tuple is more common
    "padding": "tuple", # Could also be integer, but tuple is more common
    "dilation": "tuple", # Could also be integer, but tuple is more common
    "groups": "integer",
    "padding_mode": "string",
    "qconfig": "string", # Ideally this should be a specific quantization config object, but string is the closest type.
    "bias": "boolean"
}
signatures["torch.nn.quantized.modules.Conv3d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # Could also be an integer
    "stride": "tuple", # Could also be an integer
    "padding": "tuple", # Could also be an integer
    "dilation": "tuple", # Could also be an integer
    "groups": "integer",
    "padding_mode": "string",
    "qconfig": "object", # Seems to be an object, can't represent that
    "bias": "boolean"
}
signatures["torch.nn.quantized.modules.ConvTranspose1d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "integer", # or tuple
    "stride": "integer", # or tuple
    "padding": "integer", # or tuple
    "output_padding": "integer", # or tuple
    "groups": "integer",
    "bias": "boolean",
    "dilation": "integer", # or tuple
    "padding_mode": "string", # Should be enum but approximating with string
    "qconfig": "string" # should be QConfig, but approximating with string
}
signatures["torch.nn.quantized.modules.ConvTranspose2d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # Can also be integer
    "stride": "tuple", # Can also be integer
    "padding": "tuple", # Can also be integer
    "output_padding": "tuple", # Can also be integer
    "dilation": "tuple", # Can also be integer
    "groups": "integer",
    "bias": "boolean",
    "padding_mode": "string"
}
signatures["torch.nn.quantized.modules.ConvTranspose3d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # could also be integer or tuple, but tuple is more common
    "stride": "tuple", # could also be integer or tuple, but tuple is more common
    "padding": "tuple", # could also be integer or tuple, but tuple is more common
    "output_padding": "tuple", # could also be integer or tuple, but tuple is more common
    "groups": "integer",
    "bias": "boolean",
    "dilation": "tuple", # could also be integer or tuple, but tuple is more common
    "padding_mode": "string", # should be enum but string is closest
    "qconfig": "string" # I think this can also be an object, but string is closest and most common case
}
signatures["torch.nn.quantized.modules.DeQuantize"] = {
    "": "None" # This module takes no arguments in the constructor, the forward pass takes a tensor.
}
signatures["torch.nn.quantized.modules.Dropout"] = {
    "p": "float",
    "inplace": "boolean"
}
signatures["torch.nn.quantized.modules.ELU"] = {
    "alpha": "float",
    "scale": "float",
    "zero_point": "integer"
}
signatures["torch.nn.quantized.modules.Embedding"] = {
    "num_embeddings": "integer",
    "embedding_dim": "integer",
    "padding_idx": "integer", # Could potentially be None, but integer is most common
    "scale": "float",
    "zero_point": "integer"
}
signatures["torch.nn.quantized.modules.EmbeddingBag"] = {
    "num_embeddings": "integer",
    "embedding_dim": "integer",
    "padding_idx": "integer", # Could also be None but int is more common
    "scale_grad_by_freq": "boolean",
    "mode": "string",
    "sparse": "boolean",
    "_weight": "tensor" # This is not exposed in the constructor, but it's part of the state. I am including it as _weight is a crucial part of the module.
}
signatures["torch.nn.quantized.modules.FXFloatFunctional.add"] = {
    "input": "tensor",
    "other": "tensor" # Could also be float, but tensor is more common
}
signatures["torch.nn.quantized.modules.FXFloatFunctional.mul"] = {
    "input": "tensor",
    "other": "tensor" # Could also be float, but tensor is more common
}
signatures["torch.nn.quantized.modules.FXFloatFunctional.cat"] = {
    "tensors": "tensor_list",
    "dim": "integer"
}
signatures["torch.nn.quantized.modules.FXFloatFunctional.relu"] = {
    "input": "tensor"
}
signatures["torch.nn.quantized.modules.FXFloatFunctional.hardtanh"] = {
    "input": "tensor",
    "min_val": "float",
    "max_val": "float"
}
signatures["torch.nn.quantized.modules.FXFloatFunctional.adaptive_avg_pool2d"] = {
    "input": "tensor",
    "output_size": "list" # Could also be tuple, but list seems slightly more common in examples
}
signatures["torch.nn.quantized.modules.FXFloatFunctional.avg_pool2d"] = {
    "input": "tensor",
    "kernel_size": "integer", #Could also be a tuple
    "stride": "integer", #Could also be a tuple
    "padding": "integer", #Could also be a tuple
    "ceil_mode": "boolean",
    "count_include_pad": "boolean",
    "divisor_override": "float"
}
signatures["torch.nn.quantized.modules.FXFloatFunctional.interpolate"] = {
    "input": "tensor",
    "size": "list", # Could also be tuple or None
    "scale_factor": "float", #Could also be a list/tuple
    "mode": "string",
    "align_corners": "boolean",
    "recompute_scale_factor": "boolean"
}
signatures["torch.nn.quantized.modules.FXFloatFunctional.linear"] = {
    "input": "tensor",
    "weight": "tensor",
    "bias": "tensor"
}
signatures["torch.nn.quantized.modules.FloatFunctional.add"] = {
    "other": "tensor"
}
signatures["torch.nn.quantized.modules.FloatFunctional.mul"] = {
    "other": "tensor"
}
signatures["torch.nn.quantized.modules.FloatFunctional.cat"] = {
    "tensors": "tensor_list",
    "dim": "integer"
}
signatures["torch.nn.quantized.modules.FloatFunctional.add_scalar"] = {
    "other": "float"
}
signatures["torch.nn.quantized.modules.FloatFunctional.mul_scalar"] = {
    "other": "float"
}
signatures["torch.nn.quantized.modules.FloatFunctional.relu"] = {}
signatures["torch.nn.quantized.modules.FloatFunctional.hardtanh"] = {
    "min_val": "float",
    "max_val": "float"
}
signatures["torch.nn.quantized.modules.FloatFunctional.rrelu"] = {
    "lower": "float",
    "upper": "float",
    "training": "boolean"
}
signatures["torch.nn.quantized.modules.FloatFunctional.leaky_relu"] = {
    "negative_slope": "float"
}
signatures["torch.nn.quantized.modules.FloatFunctional.elu"] = {
    "alpha": "float"
}
signatures["torch.nn.quantized.modules.FloatFunctional.threshold"] = {
    "threshold": "float",
    "value": "float"
}
signatures["torch.nn.quantized.modules.FloatFunctional.hardtanh_"] = {
    "min_val": "float",
    "max_val": "float"
}
signatures["torch.nn.quantized.modules.FloatFunctional.relu_"] = {}
signatures["torch.nn.quantized.modules.FloatFunctional.sigmoid"] = {}
signatures["torch.nn.quantized.modules.FloatFunctional.sigmoid_"] = {}
signatures["torch.nn.quantized.modules.GroupNorm"] = {
    "num_groups": "integer",
    "num_channels": "integer",
    "eps": "float",
    "affine": "boolean",
    "scale": "tensor",
    "zero_point": "tensor"
}
signatures["torch.nn.quantized.modules.Hardswish"] = {
}
signatures["torch.nn.quantized.modules.InstanceNorm1d"] = {
    "num_features": "integer",
    "eps": "float",
    "momentum": "float",
    "affine": "boolean",
    "track_running_stats": "boolean"
    # running_mean and running_var are not arguments to the constructor
}
signatures["torch.nn.quantized.modules.InstanceNorm2d"] = {
    "num_features": "integer",
    "eps": "float",
    "momentum": "float",
    "affine": "boolean",
    "track_running_stats": "boolean"
}
signatures["torch.nn.quantized.modules.InstanceNorm3d"] = {
    "num_features": "integer",
    "eps": "float",
    "momentum": "float",
    "affine": "boolean",
    "track_running_stats": "boolean"
}
signatures["torch.nn.quantized.modules.LSTM"] = {
    "input_size": "integer",
    "hidden_size": "integer",
    "num_layers": "integer",
    "bias": "boolean",
    "proj_size": "integer" # could potentially also be None
}
signatures["torch.nn.quantized.modules.LayerNorm"] = {
    "normalized_shape": "tuple", # Could also be list or integer
    "eps": "float",
    "elementwise_affine": "boolean"
}
signatures["torch.nn.quantized.modules.LeakyReLU"] = {
    "negative_slope": "float",
    "inplace": "boolean"
}
signatures["torch.nn.quantized.modules.Linear"] = {
    "in_features": "integer",
    "out_features": "integer",
    "bias": "boolean"
}
signatures["torch.nn.quantized.modules.MaxPool2d"] = {
    "kernel_size": "integer", # Could be tuple as well, but integer seems more common
    "stride": "integer", # Could be tuple as well, but integer seems more common
    "padding": "integer", # Could be tuple as well, but integer seems more common
    "dilation": "integer", # Could be tuple as well, but integer seems more common
    "return_indices": "boolean",
    "ceil_mode": "boolean"
}
signatures["torch.nn.quantized.modules.MultiheadAttention"] = {
    "embed_dim": "integer",
    "num_heads": "integer",
    "dropout": "float",
    "bias": "boolean",
    "add_bias_kv": "boolean",
    "add_zero_attn": "boolean",
    "kdim": "integer",
    "vdim": "integer"
}
signatures["torch.nn.quantized.modules.PReLU"] = {
    "num_parameters": "integer",
    "init": "float"
}
signatures["torch.nn.quantized.modules.QFunctional"] = {
}
signatures["torch.nn.quantized.modules.Quantize"] = {
    "scale": "float",
    "zero_point": "integer",
    "quant_min": "integer",
    "quant_max": "integer",
    "dtype": "dtype"
}
signatures["torch.nn.quantized.modules.ReLU6"] = {
}
signatures["torch.nn.quantized.modules.Sigmoid"] = {}
signatures["torch.nn.quantized.modules.Softmax"] = {
    "dim": "integer" # could be a tuple of integers, but integer is more common
}
signatures["torch.nn.quantized.modules.activation.ReLU"] = {}
signatures["torch.nn.quantized.modules.activation.ReLU6"] = {}
signatures["torch.nn.quantized.modules.activation.Sigmoid"] = {}
signatures["torch.nn.quantized.modules.activation.Tanh"] = {}
signatures["torch.nn.quantized.modules.batchnorm"] = {
    "num_features": "integer",
    "eps": "float",
    "momentum": "float",
    "affine": "boolean",
    "track_running_stats": "boolean", # Could be boolean or None
    "qconfig": "dtype" # Assuming this is a type argument. Could also be string.
}
signatures["torch.nn.quantized.modules.conv.Conv2d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # Could also be integer
    "stride": "tuple", # Could also be integer
    "padding": "tuple", # Could also be integer
    "dilation": "tuple", # Could also be integer
    "groups": "integer",
    "padding_mode": "string",
    "qconfig": "object" # Assuming qconfig is a configuration object, not a primitive type
}
signatures["torch.nn.quantized.modules.conv.Conv3d"] = {
    "in_channels": "integer",
    "out_channels": "integer",
    "kernel_size": "tuple", # Could also be integer
    "stride": "tuple", # Could also be integer
    "padding": "tuple", # Could also be integer
    "dilation": "tuple", # Could also be integer
    "groups": "integer",
    "padding_mode": "string",
    "qconfig": "object" # Assuming qconfig is a configuration object, not a primitive type
}
signatures["torch.nn.quantized.modules.dropout"] = {
    "p": "float", # Could also be interpreted as a tensor with a single element
    "inplace": "boolean"
}
signatures["torch.nn.quantized.modules.embedding_ops"] = {
    "num_embeddings": "integer",
    "embedding_dim": "integer",
    "padding_idx": "integer", # Could also be None, but integer is more common
    "scale": "tensor",
    "zero_point": "tensor",
    "_weight": "tensor",
    "freeze_qweights": "boolean"
}
signatures["torch.nn.quantized.modules.functional_modules.BatchNorm2d"] = {
    "input": "tensor",
    "weight": "tensor",
    "bias": "tensor",
    "running_mean": "tensor",
    "running_var": "tensor",
    "training": "boolean",
    "momentum": "float",
    "eps": "float",
}
signatures["torch.nn.quantized.modules.functional_modules.Conv2d"] = {
    "input": "tensor",
    "weight": "tensor",
    "bias": "tensor",
    "stride": "tuple",
    "padding": "tuple",
    "dilation": "tuple",
    "groups": "integer",
}
signatures["torch.nn.quantized.modules.linear"] = {
    "in_features": "integer",
    "out_features": "integer",
    "bias": "boolean"
}
signatures["torch.nn.quantized.modules.normalization.LayerNorm"] = {
    "normalized_shape": "integer", # Could also be tuple
    "eps": "float",
    "elementwise_affine": "boolean"
}
signatures["torch.nn.quantized.modules.normalization.BatchNorm2d"] = {
    "num_features": "integer",
    "eps": "float",
    "momentum": "float",
}
signatures["torch.nn.quantized.modules.rnn.LSTM"] = {
    "input_size": "integer",
    "hidden_size": "integer",
    "num_layers": "integer",
    "bias": "boolean",
    "batch_first": "boolean",
    "dropout": "float",
    "bidirectional": "boolean",
    "proj_size": "integer"
}
signatures["torch.nn.quantized.modules.rnn.GRU"] = {
    "input_size": "integer",
    "hidden_size": "integer",
    "num_layers": "integer",
    "bias": "boolean",
    "batch_first": "boolean",
    "dropout": "float",
    "bidirectional": "boolean"
}
signatures["torch.nn.quantized.modules.rnn.RNN"] = {
    "input_size": "integer",
    "hidden_size": "integer",
    "num_layers": "integer",
    "nonlinearity": "string",
    "bias": "boolean",
    "batch_first": "boolean",
    "dropout": "float",
    "bidirectional": "boolean"
}
signatures["torch.nn.quantized.modules.utils._quantize"] = {
    "mod": "tensor", # Should probably be Module type, but approximating with tensor
    "inplace": "boolean"
}
signatures["torch.nn.quantized.modules.utils._dequantize"] = {
    "mod": "tensor", # Should probably be Module type, but approximating with tensor
    "inplace": "boolean"
}
signatures["torch.nn.quantized.modules.utils.load_qconfig"] = {
    "mod": "tensor", # Should probably be Module type, but approximating with tensor
    "qconfig_summary": "string" # Not exactly string, but a dictionary-like representation of QConfigSummary
}
signatures["torch.nn.quantized.modules.utils.get_fqn_dict"] = {
    "mod": "tensor" # Should probably be Module type, but approximating with tensor
}
signatures["torch.nn.utils.clip_grad_norm_"] = {
    "parameters": "tensor_list",
    "max_norm": "float",
    "norm_type": "float" # Could also be inf, but float seems more common
}
signatures["torch.nn.utils.clip_grad_value_"] = {
    "parameters": "tensor_list",
    "clip_value": "float"
}
signatures["torch.nn.utils.rnn.pack_padded_sequence"] = {
    "input": "tensor",
    "lengths": "tensor",
    "batch_first": "boolean",
    "enforce_sorted": "boolean"
}
signatures["torch.nn.utils.rnn.pad_packed_sequence"] = {
    "sequence": "tensor",
    "batch_first": "boolean",
    "padding_value": "float",
    "total_length": "integer"
}
signatures["torch.nn.utils.rnn.pack_sequence"] = {
    "sequences": "tensor_list",
    "enforce_sorted": "boolean"
}
signatures["torch.nn.utils.rnn.pad_sequence"] = {
    "sequences": "tensor_list",
    "batch_first": "boolean",
    "padding_value": "float",
    "total_length": "integer"
}
signatures["torch.nn.utils.vector_to_parameters"] = {
    "vec": "tensor",
    "parameters": "tensor_list"
}
signatures["torch.nn.utils.parameters_to_vector"] = {
    "parameters": "tensor_list"
}
signatures["torch.nn.utils.clip_grad"] = {
    "parameters": "tensor_list",
    "max_norm": "float",
    "norm_type": "float" # Could also be integer, but float is more common
}
signatures["torch.nn.utils.clip_grad_norm"] = {
    "parameters": "tensor_list",
    "max_norm": "float",
    "norm_type": "float" #could be integer too, but float seems more common
}
signatures["torch.nn.utils.clip_grad_value_"] = {
    "parameters": "tensor_list",
    "clip_value": "float" # Could potentially be a tensor if different clip values are needed for each parameter, but float is the most common case.
}
signatures["torch.nn.utils.clip_grads_with_norm_"] = {
    "grads": "tensor_list",
    "max_norm": "float",
    "norm_type": "float" # could be integer, but float is more common and acceptable
}
signatures["torch.nn.utils.convert_conv2d_weight_memory_format"] = {
    "weight": "tensor",
    "memory_format": "string"
}
signatures["torch.nn.utils.convert_conv3d_weight_memory_format"] = {
    "weight": "tensor",
    "memory_format": "string" # Could potentially be more specific, but string is the closest.
}
signatures["torch.nn.utils.convert_parameters"] = {
    "parameters": "tensor_list",
}
signatures["torch.nn.utils.fuse_conv_bn_eval"] = {
    "conv": "tensor", # Should be a Module, but assuming tensor for simplicity
    "bn": "tensor" # Should be a Module, but assuming tensor for simplicity
}
signatures["torch.nn.utils.fuse_conv_bn_weights"] = {
    "conv": "tensor", # Should this be a module (nn.Conv2d)?
    "bn": "tensor" # Should this be a module (nn.BatchNorm2d)?
}
signatures["torch.nn.utils.fuse_linear_bn_eval"] = {
    "linear": "tensor", # Should this be nn.Linear?
    "bn": "tensor" # Should this be nn.BatchNorm1d or nn.BatchNorm2d?
}
signatures["torch.nn.utils.fuse_linear_bn_weights"] = {
    "linear": "tensor",
    "bn": "tensor" # Should probably be a nn.BatchNorm1d or nn.BatchNorm2d object
}
signatures["torch.nn.utils.fusion.fuse_conv_bn_eval"] = {
    "conv": "tensor", # Actually a module
    "bn": "tensor" # Actually a module
}
signatures["torch.nn.utils.fusion.fuse_deconv_bn_eval"] = {
    "deconv": "tensor", # Actually a module
    "bn": "tensor" # Actually a module
}
signatures["torch.nn.utils.get_total_norm"] = {
    "parameters": "tensor_list",
    "norm_type": "float" # Could also be integer, but float is more common
}
signatures["torch.nn.utils.init.calculate_gain"] = {
    "nonlinearity": "string",
    "param": "float" # could be None as well
}
signatures["torch.nn.utils.init.uniform_"] = {
    "tensor": "tensor",
    "a": "float",
    "b": "float"
}
signatures["torch.nn.utils.init.normal_"] = {
    "tensor": "tensor",
    "mean": "float",
    "std": "float"
}
signatures["torch.nn.utils.init.trunc_normal_"] = {
    "tensor": "tensor",
    "mean": "float",
    "std": "float",
    "a": "float",
    "b": "float"
}
signatures["torch.nn.utils.init.constant_"] = {
    "tensor": "tensor",
    "val": "float"
}
signatures["torch.nn.utils.init.ones_"] = {
    "tensor": "tensor"
}
signatures["torch.nn.utils.init.zeros_"] = {
    "tensor": "tensor"
}
signatures["torch.nn.utils.init.eye_"] = {
    "tensor": "tensor"
}
signatures["torch.nn.utils.init.dirac_"] = {
    "tensor": "tensor",
    "offset": "integer"
}
signatures["torch.nn.utils.init.xavier_uniform_"] = {
    "tensor": "tensor",
    "gain": "float"
}
signatures["torch.nn.utils.init.xavier_normal_"] = {
    "tensor": "tensor",
    "gain": "float"
}
signatures["torch.nn.utils.init.kaiming_uniform_"] = {
    "tensor": "tensor",
    "a": "float",
    "mode": "string",
    "nonlinearity": "string"
}
signatures["torch.nn.utils.init.kaiming_normal_"] = {
    "tensor": "tensor",
    "a": "float",
    "mode": "string",
    "nonlinearity": "string"
}
signatures["torch.nn.utils.init.orthogonal_"] = {
    "tensor": "tensor",
    "gain": "float"
}
signatures["torch.nn.utils.init.sparse_"] = {
    "tensor": "tensor",
    "sparsity": "float",
    "std": "float"
}
signatures["torch.nn.utils.memory_format"] = {
    "input": "tensor",
    "memory_format": "string" # Could also be MemoryFormat, but string is more general
}
signatures["torch.nn.utils.parameters_to_vector"] = {
    "parameters": "tensor_list"
}
signatures["torch.nn.utils.parametrizations.spectral_norm"] = {
    "module": "tensor", # Can be a nn.Module as well
    "name": "string",
    "n_power_iterations": "integer",
    "eps": "float",
    "dim": "integer"
}
signatures["torch.nn.utils.parametrizations.remove_parametrizations"] = {
    "module": "tensor", # Can be a nn.Module as well
    "name": "string",
    "leave_parametrized": "boolean"
}
signatures["torch.nn.utils.parametrizations.orthogonal"] = {
    "module": "tensor", # Can be a nn.Module as well
    "name": "string",
    "use_householder": "boolean"
}
signatures["torch.nn.utils.parametrize"] = {
    "module": "module", # could also be string (name of the module), but module seems more appropriate
    "parameter_name": "string",
    "parametrization": "module", # or Callable? but module is also reasonable
    "unsafe": "boolean"
}
signatures["torch.nn.utils.remove_spectral_norm"] = {
    "module": "module", # Should be "module", nn.Module
    "name": "string"
}
signatures["torch.nn.utils.remove_weight_norm"] = {
    "module": "module", # Could potentially be more specific, like "nn.Module", but sticking to the limited types
    "name": "string"
}
signatures["torch.nn.utils.rnn.pack_padded_sequence"] = {
    "input": "tensor",
    "lengths": "tensor", # could be integer list but tensor is more common
    "batch_first": "boolean",
    "enforce_sorted": "boolean"
}
signatures["torch.nn.utils.rnn.pad_packed_sequence"] = {
    "sequence": "tensor",
    "batch_first": "boolean",
    "padding_value": "float",
    "total_length": "integer"
}
signatures["torch.nn.utils.rnn.pack_sequence"] = {
    "sequences": "tensor_list",
    "enforce_sorted": "boolean"
}
signatures["torch.nn.utils.rnn.pad_sequence"] = {
    "sequences": "tensor_list",
    "batch_first": "boolean",
    "padding_value": "float"
}
signatures["torch.nn.utils.skip_init"] = {
    "module": "module" # This should probably be a subclass of nn.Module but it does not fall into the strictly defined allowed types
}
signatures["torch.nn.utils.spectral_norm"] = {
    "module": "tensor", # Should be a nn.Module but closest type is tensor
    "name": "string",
    "n_power_iterations": "integer",
    "eps": "float",
    "dim": "integer"
}
signatures["torch.nn.utils.stateless.functional_call"] = {
    "module": "torch.nn.Module", # Should ideally be a "module" type, but that is not one of the options
    "weights": "tuple",
    "inputs": "tuple",
    "kwargs": "dictionary" # This argument type is not in the choices, but it's implicitly included in the specifications example
}
signatures["torch.nn.utils.stateless.flatten_params"] = {
    "module": "torch.nn.Module" # Should ideally be a "module" type, but that is not one of the options
}
signatures["torch.nn.utils.stateless.unflatten_params"] = {
    "params": "tensor",
    "module": "torch.nn.Module" # Should ideally be a "module" type, but that is not one of the options
}
signatures["torch.nn.utils.vector_to_parameters"] = {
    "vec": "tensor",
    "parameters": "tensor_list" # Could also be list, but tensor_list seems more appropriate.
}
signatures["torch.nn.utils.weight_norm"] = {
    "module": "tensor", # Actually it is a nn.Module, but tensor seems closer
    "name": "string",
    "dim": "integer"
}
signatures["torch.no_grad"] = {
    "orig_func": "None" # Can also be a function. Defaulting to None for simplicity.
}
signatures["torch.nonzero"] = {
    "input": "tensor",
    "as_tuple": "boolean"
}
signatures["torch.nonzero_static"] = {
    "input": "tensor"
}
signatures["torch.norm"] = {
    "input": "tensor",
    "p": "float", # Could also be string like 'fro' or 'nuc', but float is more common for numerical order.
    "dim": "tuple", # Can also be int or list, but tuple is more general for multiple dimensions
    "keepdim": "boolean",
    "dtype": "dtype"
}
signatures["torch.norm_except_dim"] = {
    "input": "tensor",
    "dim": "integer" , # can also be tuple of ints
    "keepdim": "boolean"
}
signatures["torch.normal"] = {
    "mean": "tensor",
    "std": "tensor",
    "generator": "torch.Generator", # unclear, but best match
    "size": "tuple" # it could also be a list of integers
}
signatures["torch.not_equal"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.nuclear_norm"] = {
    "input": "tensor",
    "p": "float", # Could also be a string like 'fro', unclear if this is the common case
    "keepdim": "boolean"
}
signatures["torch.numel"] = {
    "input": "tensor"
}
signatures["torch.ones"] = {
    "size": "tuple", # Could also be a variable number of integers, but tuple is more common as a collection
    "dtype": "dtype",
    "layout": "string",
    "requires_grad": "boolean"
}
signatures["torch.ones_like"] = {
    "input": "tensor",
    "dtype": "dtype",
    "layout": "string", # Could also be something else, not really sure.
    "requires_grad": "boolean",
    "memory_format": "string" # Could also be something else, not really sure.
}
signatures["torch.abs"] = {
    "input": "tensor"
}
signatures["torch.acos"] = {
    "input": "tensor"
}
signatures["torch.add"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.addmm"] = {
    "input": "tensor",
    "mat1": "tensor",
    "mat2": "tensor",
    "beta": "float",
    "alpha": "float"
}
signatures["torch.all"] = {
    "input": "tensor",
    "dim": "integer",
    "keepdim": "boolean"
}
signatures["torch.any"] = {
    "input": "tensor",
    "dim": "integer",
    "keepdim": "boolean"
}
signatures["torch.arange"] = {
    "start": "float",
    "end": "float",
    "step": "float"
}
signatures["torch.argmax"] = {
    "input": "tensor",
    "dim": "integer",
    "keepdim": "boolean" # It could be boolean, but integer is more used
}
signatures["torch.argmin"] = {
    "input": "tensor",
    "dim": "integer",
    "keepdim": "boolean" # It could be boolean, but integer is more used
}
signatures["torch.argsort"] = {
    "input": "tensor",
    "dim": "integer",
    "descending": "boolean"
}
signatures["torch.asin"] = {
    "input": "tensor"
}
signatures["torch.atan"] = {
    "input": "tensor"
}
signatures["torch.atan2"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.baddbmm"] = {
    "input": "tensor",
    "batch1": "tensor",
    "batch2": "tensor",
    "beta": "float",
    "alpha": "float"
}
signatures["torch.bernoulli"] = {
    "input": "tensor"
}
signatures["torch.bmm"] = {
    "input": "tensor",
    "mat2": "tensor"
}
signatures["torch.broadcast_to"] = {
    "input": "tensor",
    "size": "tuple"
}
signatures["torch.cat"] = {
    "tensors": "tensor_list",
    "dim": "integer"
}
signatures["torch.ceil"] = {
    "input": "tensor"
}
signatures["torch.chunk"] = {
    "input": "tensor",
    "chunks": "integer",
    "dim": "integer"
}
signatures["torch.clamp"] = {
    "input": "tensor",
    "min": "float",
    "max": "float"
}
signatures["torch.clone"] = {
    "input": "tensor"
}
signatures["torch.cos"] = {
    "input": "tensor"
}
signatures["torch.cosh"] = {
    "input": "tensor"
}
signatures["torch.cross"] = {
    "input": "tensor",
    "other": "tensor",
    "dim": "integer"
}
signatures["torch.cumprod"] = {
    "input": "tensor",
    "dim": "integer"
}
signatures["torch.cumsum"] = {
    "input": "tensor",
    "dim": "integer"
}
signatures["torch.det"] = {
    "input": "tensor"
}
signatures["torch.diag"] = {
    "input": "tensor",
    "diagonal": "integer"
}
signatures["torch.diag_embed"] = {
    "input": "tensor",
    "offset": "integer",
    "dim1": "integer",
    "dim2": "integer"
}
signatures["torch.diagflat"] = {
    "input": "tensor",
    "offset": "integer"
}
signatures["torch.diagonal"] = {
    "input": "tensor",
    "offset": "integer",
    "dim1": "integer",
    "dim2": "integer"
}
signatures["torch.diff"] = {
    "input": "tensor",
    "n": "integer",
    "dim": "integer"
}
signatures["torch.digamma"] = {
    "input": "tensor"
}
signatures["torch.div"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.dot"] = {
    "input": "tensor",
    "tensor2": "tensor"
}
signatures["torch.empty"] = {
    "size": "tuple"
}
signatures["torch.eq"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.exp"] = {
    "input": "tensor"
}
signatures["torch.expm1"] = {
    "input": "tensor"
}
signatures["torch.fft.fft"] = {
    "input": "tensor",
    "n": "integer",
    "dim": "tuple", # Could be integer
    "norm": "string"
}
signatures["torch.fill"] = {
    "input": "tensor",
    "value": "float"
}
signatures["torch.flatten"] = {
    "input": "tensor",
    "start_dim": "integer",
    "end_dim": "integer"
}
signatures["torch.flip"] = {
    "input": "tensor",
    "dims": "tuple"
}
signatures["torch.floor"] = {
    "input": "tensor"
}
signatures["torch.fmod"] = {
    "input": "tensor",
    "divisor": "float"
}
signatures["torch.frac"] = {
    "input": "tensor"
}
signatures["torch.gather"] = {
    "input": "tensor",
    "dim": "integer",
    "index": "tensor"
}
signatures["torch.gcd"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.geqrf"] = {
    "input": "tensor"
}
signatures["torch.gradient"] = {
    "input": "tensor",
    "dim": "tuple", # Could be integer
    "spacing": "tuple", # Could be float
    "edge_order": "integer"
}
signatures["torch.gt"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.hstack"] = {
    "tensors": "tensor_list"
}
signatures["torch.hypot"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.i0"] = {
    "input": "tensor"
}
signatures["torch.igamma"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.igammac"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.imag"] = {
    "input": "tensor"
}
signatures["torch.index_select"] = {
    "input": "tensor",
    "dim": "integer",
    "index": "tensor"
}
signatures["torch.inner"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.int_repr"] = {
    "input": "tensor"
}
signatures["torch.inverse"] = {
    "input": "tensor"
}
signatures["torch.isclose"] = {
    "input": "tensor",
    "other": "tensor",
    "rtol": "float",
    "atol": "float",
    "equal_nan": "boolean"
}
signatures["torch.isfinite"] = {
    "input": "tensor"
}
signatures["torch.isinf"] = {
    "input": "tensor"
}
signatures["torch.isnan"] = {
    "input": "tensor"
}
signatures["torch.isreal"] = {
    "input": "tensor"
}
signatures["torch.kron"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.lcm"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.ldexp"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.lerp"] = {
    "input": "tensor",
    "end": "tensor",
    "weight": "float"
}
signatures["torch.lgamma"] = {
    "input": "tensor"
}
signatures["torch.log"] = {
    "input": "tensor"
}
signatures["torch.log10"] = {
    "input": "tensor"
}
signatures["torch.log1p"] = {
    "input": "tensor"
}
signatures["torch.log2"] = {
    "input": "tensor"
}
signatures["torch.logaddexp"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.logaddexp2"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.logdet"] = {
    "input": "tensor"
}
signatures["torch.logical_and"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.logical_not"] = {
    "input": "tensor"
}
signatures["torch.logical_or"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.logical_xor"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.logsumexp"] = {
    "input": "tensor",
    "dim": "tuple", # Could be integer
    "keepdim": "boolean"
}
signatures["torch.lstsq"] = {
    "input": "tensor",
    "A": "tensor"
}
signatures["torch.lt"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.matmul"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.max"] = {
    "input": "tensor",
    "dim": "integer",
    "keepdim": "boolean"
}
signatures["torch.maximum"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.mean"] = {
    "input": "tensor",
    "dim": "tuple",  # Could be integer
    "keepdim": "boolean"
}
signatures["torch.median"] = {
    "input": "tensor"
}
signatures["torch.min"] = {
    "input": "tensor",
    "dim": "integer",
    "keepdim": "boolean"
}
signatures["torch.minimum"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.mm"] = {
    "input": "tensor",
    "mat2": "tensor"
}
signatures["torch.movedim"] = {
    "input": "tensor",
    "source": "tuple", # Could be integer
    "destination": "tuple" # Could be integer
}
signatures["torch.moveaxis"] = {
    "input": "tensor",
    "source": "integer",
    "destination": "integer"
}
signatures["torch.mul"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.multinomial"] = {
    "input": "tensor",
    "num_samples": "integer",
    "replacement": "boolean"
}
signatures["torch.mv"] = {
    "input": "tensor",
    "vec": "tensor"
}
signatures["torch.mvlgamma"] = {
    "input": "tensor",
    "p": "integer"
}
signatures["torch.nan_to_num"] = {
    "input": "tensor",
    "nan": "float",
    "posinf": "float",
    "neginf": "float"
}
signatures["torch.nanmean"] = {
    "input": "tensor",
    "dim": "tuple", # Could be integer
    "keepdim": "boolean"
}
signatures["torch.nanmedian"] = {
    "input": "tensor"
}
signatures["torch.nansum"] = {
    "input": "tensor",
    "dim": "tuple", # Could be integer
    "keepdim": "boolean"
}
signatures["torch.narrow"] = {
    "input": "tensor",
    "dim": "integer",
    "start": "integer",
    "length": "integer"
}
signatures["torch.ndimension"] = {
    "input": "tensor"
}
signatures["torch.ne"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.negative"] = {
    "input": "tensor"
}
signatures["torch.nextafter"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.nonzero"] = {
    "input": "tensor"
}
signatures["torch.norm"] = {
    "input": "tensor",
    "p": "float",
    "dim": "tuple", # Could be integer
    "keepdim": "boolean"
}
signatures["torch.normal"] = {
    "mean": "float",
    "std": "float",
    "size": "tuple"
}
signatures["torch.numel"] = {
    "input": "tensor"
}
signatures["torch.orgqr"] = {
    "input": "tensor",
    "input2": "tensor" # unsure of the name
}
signatures["torch.outer"] = {
    "input": "tensor",
    "vec2": "tensor"
}
signatures["torch.permute"] = {
    "input": "tensor",
    "dims": "tuple"
}
signatures["torch.polygamma"] = {
    "n": "integer",
    "input": "tensor"
}
signatures["torch.positive"] = {
    "input": "tensor"
}
signatures["torch.pow"] = {
    "input": "tensor",
    "exponent": "float"
}
signatures["torch.prod"] = {
    "input": "tensor",
    "dim": "integer",
    "keepdim": "boolean"
}
signatures["torch.qr"] = {
    "input": "tensor"
}
signatures["torch.rand"] = {
    "size": "tuple"
}
signatures["torch.randint"] = {
    "low": "integer",
    "high": "integer",
    "size": "tuple"
}
signatures["torch.randn"] = {
    "size": "tuple"
}
signatures["torch.ravel"] = {
    "input": "tensor"
}
signatures["torch.real"] = {
    "input": "tensor"
}
signatures["torch.reciprocal"] = {
    "input": "tensor"
}
signatures["torch.repeat"] = {
    "input": "tensor",
    "repeats": "tuple" # unsure of the name
}
signatures["torch.reshape"] = {
    "input": "tensor",
    "shape": "tuple"
}
signatures["torch.round"] = {
    "input": "tensor"
}
signatures["torch.rsqrt"] = {
    "input": "tensor"
}
signatures["torch.scatter"] = {
    "input": "tensor",
    "dim": "integer",
    "index": "tensor",
    "src": "tensor"
}
signatures["torch.segment_reduce"] = {
    "data": "tensor",
    "segment_ids": "tensor",
    "reduce": "string",
    "init": "float" # could be tensor too
}
signatures["torch.select"] = {
    "input": "tensor",
    "dim": "integer",
    "index": "integer"
}
signatures["torch.sgn"] = {
    "input": "tensor"
}
signatures["torch.sigmoid"] = {
    "input": "tensor"
}
signatures["torch.sign"] = {
    "input": "tensor"
}
signatures["torch.sin"] = {
    "input": "tensor"
}
signatures["torch.sinc"] = {
    "input": "tensor"
}
signatures["torch.sinh"] = {
    "input": "tensor"
}
signatures["torch.size"] = {
    "input": "tensor"
}
signatures["torch.slice"] = {
    "input": "tensor",
    "dim": "integer",
    "start": "integer",
    "end": "integer",
    "step": "integer"
}
signatures["torch.sort"] = {
    "input": "tensor",
    "dim": "integer",
    "descending": "boolean"
}
signatures["torch.sqrt"] = {
    "input": "tensor"
}
signatures["torch.square"] = {
    "input": "tensor"
}
signatures["torch.squeeze"] = {
    "input": "tensor",
    "dim": "integer"
}
signatures["torch.stack"] = {
    "tensors": "tensor_list",
    "dim": "integer"
}
signatures["torch.std"] = {
    "input": "tensor",
    "dim": "tuple", # Could be integer
    "unbiased": "boolean",
    "keepdim": "boolean"
}
signatures["torch.sub"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.sum"] = {
    "input": "tensor",
    "dim": "tuple", # Could be integer
    "keepdim": "boolean"
}
signatures["torch.swapaxes"] = {
    "input": "tensor",
    "dim0": "integer",
    "dim1": "integer"
}
signatures["torch.swapdims"] = {
    "input": "tensor",
    "dim0": "integer",
    "dim1": "integer"
}
signatures["torch.tan"] = {
    "input": "tensor"
}
signatures["torch.tanh"] = {
    "input": "tensor"
}
signatures["torch.tensor_split"] = {
    "input": "tensor",
    "indices_or_sections": "integer", # Could also be a list
    "dim": "integer"
}
signatures["torch.tile"] = {
    "input": "tensor",
    "dims": "tuple"
}
signatures["torch.topk"] = {
    "input": "tensor",
    "k": "integer",
    "dim": "integer",
    "largest": "boolean",
    "sorted": "boolean"
}
signatures["torch.trace"] = {
    "input": "tensor"
}
signatures["torch.transpose"] = {
    "input": "tensor",
    "dim0": "integer",
    "dim1": "integer"
}
signatures["torch.trapz"] = {
    "y": "tensor",
    "x": "tensor", # Could be float
    "dim": "integer"
}
signatures["torch.triangular_solve"] = {
    "input": "tensor",
    "A": "tensor",
    "upper": "boolean",
    "transpose": "boolean",
    "unitriangular": "boolean"
}
signatures["torch.tril"] = {
    "input": "tensor",
    "diagonal": "integer"
}
signatures["torch.triu"] = {
    "input": "tensor",
    "diagonal": "integer"
}
signatures["torch.trunc"] = {
    "input": "tensor"
}
signatures["torch.unique"] = {
    "input": "tensor",
    "sorted": "boolean",
    "return_inverse": "boolean",
    "return_counts": "boolean",
    "dim": "integer"
}
signatures["torch.unsqueeze"] = {
    "input": "tensor",
    "dim": "integer"
}
signatures["torch.var"] = {
    "input": "tensor",
    "dim": "tuple", # Could be integer
    "unbiased": "boolean",
    "keepdim": "boolean"
}
signatures["torch.vstack"] = {
    "tensors": "tensor_list"
}
signatures["torch.where"] = {
    "condition": "tensor",
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.zeros"] = {
    "size": "tuple"
}
signatures["torch.optim.Optimizer"] = {
    "params": "list", # Can also be a generator or dict of parameters
    "defaults": "dict"
}
signatures["torch.optim.SGD"] = {
    "params": "list", #iterable of parameters to optimize or dicts defining parameter groups
    "lr": "float",
    "momentum": "float",
    "dampening": "float",
    "weight_decay": "float",
    "nesterov": "boolean"
}
signatures["torch.optim.ASGD"] = {
    "params": "list",
    "lr": "float",
    "lambd": "float",
    "alpha": "float",
    "t0": "float",
    "weight_decay": "float"
}
signatures["torch.optim.Rprop"] = {
    "params": "list",
    "lr": "float",
    "eta": "tuple",
    "step_sizes": "tuple",
    "weight_decay": "float"
}
signatures["torch.optim.Adagrad"] = {
    "params": "list",
    "lr": "float",
    "lr_decay": "float",
    "weight_decay": "float",
    "initial_accumulator_value": "float",
    "eps": "float"
}
signatures["torch.optim.Adam"] = {
    "params": "list",
    "lr": "float",
    "betas": "tuple",
    "eps": "float",
    "weight_decay": "float",
    "amsgrad": "boolean"
}
signatures["torch.optim.AdamW"] = {
    "params": "list",
    "lr": "float",
    "betas": "tuple",
    "eps": "float",
    "weight_decay": "float",
    "amsgrad": "boolean"
}
signatures["torch.optim.Adamax"] = {
    "params": "list",
    "lr": "float",
    "betas": "tuple",
    "eps": "float",
    "weight_decay": "float"
}
signatures["torch.optim.RMSprop"] = {
    "params": "list",
    "lr": "float",
    "alpha": "float",
    "eps": "float",
    "weight_decay": "float",
    "momentum": "float",
    "centered": "boolean"
}
signatures["torch.optim.LBFGS"] = {
    "params": "list",
    "lr": "float",
    "max_iter": "integer",
    "max_eval": "integer",
    "tolerance_grad": "float",
    "tolerance_change": "float",
    "history_size": "integer",
    "line_search_fn": "string" #or None
}
signatures["torch.optim.Adadelta"] = {
    "params": "list",
    "lr": "float",
    "rho": "float",
    "eps": "float",
    "weight_decay": "float"
}
signatures["torch.optim.SparseAdam"] = {
    "params": "list",
    "lr": "float",
    "betas": "tuple",
    "eps": "float"
}
signatures["torch.orgqr"] = {
    "input": "tensor",
    "tau": "tensor"
}
signatures["torch.ormqr"] = {
    "input": "tensor",
    "tau": "tensor",
    "other": "tensor",
    "left": "boolean",
    "transpose": "boolean"
}
signatures["torch.os.makedirs"] = {
    "name": "string",
    "mode": "integer", # Can also be a string representing octal mode
    "exist_ok": "boolean"
}
signatures["torch.os.remove"] = {
    "path": "string"
}
signatures["torch.os.rmdir"] = {
    "path": "string"
}
signatures["torch.os.rename"] = {
    "src": "string",
    "dst": "string"
}
signatures["torch.os.replace"] = {
    "src": "string",
    "dst": "string"
}
signatures["torch.os.renames"] = {
    "old": "string",
    "new": "string"
}
signatures["torch.os.removedirs"] = {
    "name": "string"
}
signatures["torch.os.mkdir"] = {
    "path": "string",
    "mode": "integer" # Can also be a string representing octal mode
}
signatures["torch.os.path.exists"] = {
    "path": "string"
}
signatures["torch.os.path.isfile"] = {
    "path": "string"
}
signatures["torch.os.path.isdir"] = {
    "path": "string"
}
signatures["torch.overrides.has_torch_function"] = {
    "relevant_args": "tuple" # Could also be a list, but tuple is slightly more common
}
signatures["torch.overrides.is_tensor_method"] = {
    "frame": "string" # Assuming frame refers to a string representation of the function call
}
signatures["torch.overrides.get_overridable_functions"] = {
}
signatures["torch.package"] = {
    "package_dir": "string",
    "importer": "string",
    "extern_modules": "list", # Could be a list of strings, or modules. Assuming string for simplicity
    "extra_files": "list" # Could be a list of strings, or tuples. Assuming list for simplicity
}
signatures["torch.pairwise_distance"] = {
    "x1": "tensor",
    "x2": "tensor",
    "p": "float", # could also be integer
    "eps": "float",
    "keepdim": "boolean"
}
signatures["torch.parse_ir"] = {
    "code": "string",
    "namespace": "string" # Could potentially be a dictionary or something more specific
}
signatures["torch.parse_schema"] = {
    "schema_string": "string"
}
signatures["torch.parse_type_comment"] = {
    "comment": "string",
    "module": "string" # Could also be a module type, not sure.
}
signatures["torch.pca_lowrank"] = {
    "A": "tensor",
    "q": "integer",
    "center": "boolean",
    "niter": "integer"
}
signatures["torch.pdist"] = {
    "input": "tensor",
    "p": "float" # Could also be integer, but float seems more common for distance metrics
}
signatures["torch.permute"] = {
    "input": "tensor",
    "dims": "tuple"
}
signatures["torch.permute_copy"] = {
    "input": "tensor",
    "dims": "tuple" # Could also accept "list", but "tuple" seems more common for specifying dimensions.
}
signatures["torch.pixel_shuffle"] = {
    "input": "tensor",
    "upscale_factor": "integer"
}
signatures["torch.pixel_unshuffle"] = {
    "input": "tensor",
    "downscale_factor": "integer"
}
signatures["torch.platform"] = {
}
signatures["torch.poisson"] = {
    "input": "tensor",
    "generator": "torch.Generator" # Could also be None, but torch.Generator seems more appropriate when it is not None
}
signatures["torch.poisson_nll_loss"] = {
    "input": "tensor",
    "target": "tensor",
    "log_input": "boolean",
    "full": "boolean",
    "eps": "float",
    "reduction": "string"
}
signatures["torch.polygamma"] = {
    "n": "integer",
    "input": "tensor"
}
signatures["torch.positive"] = {
    "input": "tensor"
}
signatures["torch.pow"] = {
    "input": "tensor",
    "exponent": "float" # Can be float or tensor, choosing float as the scalar exponent seems more common
}
signatures["torch.prelu"] = {
    "input": "tensor",
    "weight": "tensor" # Could potentially also be a single float, but tensor is more common
}
signatures["torch.prepare_multiprocessing_environment"] = {
    "rank": "integer"
}
signatures["torch.prod"] = {
    "input": "tensor",
    "dim": "integer",
    "keepdim": "boolean",
    "dtype": "dtype"
}
signatures["torch.profiler.profile"] = {
    "activities": "list", # list of profiler activity
    "schedule": "callable",
    "on_trace_ready": "callable",
    "record_shapes": "boolean",
    "profile_memory": "boolean",
    "with_stack": "boolean",
    "with_flops": "boolean",
    "experimental_config": "tuple" # unclear what this should be but tuple sounds most fitting
}
signatures["torch.profiler.record_function"] = {
    "name": "string"
}
signatures["torch.profiler.tensorboard_trace_handler"] = {
    "dir_name": "string",
    "worker_name": "string",
    "use_gzip": "boolean"
}
signatures["torch.profiler.schedule"] = {
    "wait": "integer",
    "warmup": "integer",
    "active": "integer",
    "repeat": "integer"
}
signatures["torch.put"] = {
    "input": "tensor",
    "index": "tensor",
    "source": "tensor",
    "accumulate": "boolean"
}
signatures["torch.q_per_channel_axis"] = {
    "input": "tensor",
    "scales": "tensor",
    "zero_points": "tensor",
    "axis": "integer"
}
signatures["torch.q_per_channel_scales"] = {
    "input": "tensor"
}
signatures["torch.q_per_channel_zero_points"] = {
    "scales": "tensor",
    "qmin": "integer",
    "qmax": "integer",
    "dtype": "dtype"
}
signatures["torch.q_scale"] = {
    "qtensor": "tensor"
}
signatures["torch.q_zero_point"] = {
    "input": "tensor"
}
signatures["torch.qr"] = {
    "input": "tensor",
    "some": "boolean"
}
signatures["torch.qscheme"] = {
    "qscheme": "string" # Could also be a custom enum type
}
signatures["torch.quantile"] = {
    "input": "tensor",
    "q": "tensor", # Could also be a float but tensor seems more general
    "dim": "integer",
    "keepdim": "boolean",
    "interpolation": "string"
}
signatures["torch.quantization.QuantStub"] = {
    "dtype": "dtype" # should this be optional?
}
signatures["torch.quantization.DeQuantStub"] = {
    "dtype": "dtype" # should this be optional?
}
signatures["torch.quantization.convert"] = {
    "module": "tensor" # Module is also accepted but Tensor seems more correct
}
signatures["torch.quantization.prepare"] = {
    "model": "tensor", # Module is also accepted but Tensor seems more correct
    "qconfig_dict": "dict"
}
signatures["torch.quantization.propagate_qconfig_"] = {
    "module": "tensor" # Module is also accepted but Tensor seems more correct
}
signatures["torch.quantization.get_default_qconfig"] = {
    "backend": "string"
}
signatures["torch.quantization.get_default_qconfig_propagation_list"] = {
    "backend": "string"
}
signatures["torch.quantization.QConfig"] = {
    "activation": "object", # A quantization scheme for activations
    "weight": "object" # A quantization scheme for weights
}
signatures["torch.quantization.default_qconfig"] = {
    "backend": "string"
}
signatures["torch.quantize_per_channel"] = {
    "input": "tensor",
    "scales": "tensor",
    "zero_points": "tensor",
    "axis": "integer",
    "dtype": "dtype"
}
signatures["torch.quantize_per_tensor"] = {
    "input": "tensor",
    "scale": "float",
    "zero_point": "integer",
    "dtype": "dtype"
}
signatures["torch.quantize_per_tensor_dynamic"] = {
    "input": "tensor",
    "qconfig": "tuple", # Could be a QConfig object, but tuple is a more general representation
    "dtype": "dtype"
}
signatures["torch.quantized_batch_norm"] = {
    "input": "tensor",
    "weight": "tensor",
    "bias": "tensor",
    "mean": "tensor",
    "var": "tensor",
    "eps": "float",
    "output_scale": "float",
    "output_zero_point": "integer"
}
signatures["torch.quantized_gru"] = {
    "input": "tensor",
    "hx": "tensor",
    "params": "tensor_list",
    "has_biases": "boolean",
    "num_layers": "integer",
    "dropout": "float",
    "train": "boolean",
    "bidirectional": "boolean",
    "batch_first": "boolean"
}
signatures["torch.quantized_gru_cell"] = {
    "input_tensor": "tensor",
    "hx": "tensor",
    "w_ih": "tensor",
    "w_hh": "tensor",
    "b_ih": "tensor",
    "b_hh": "tensor",
    "scale_ih": "float",
    "zero_point_ih": "integer",
    "scale_hh": "float",
    "zero_point_hh": "integer"
}
signatures["torch.quantized_lstm"] = {
    "input": "tensor",
    "hx": "tuple", # could be a tuple or a tensor
    "params_list": "list",
    "has_biases": "boolean",
    "num_layers": "integer",
    "dropout": "float",
    "train": "boolean",
    "bidirectional": "boolean",
    "batch_first": "boolean",
    "use_relu": "boolean"
}
signatures["torch.quantized_lstm_cell"] = {
    "input": "tensor",
    "hx": "tuple",
    "w_ih": "tensor",
    "w_hh": "tensor",
    "b_ih": "tensor",
    "b_hh": "tensor",
    "scale_ih": "float",
    "scale_hh": "float",
    "zero_point_ih": "integer",
    "zero_point_hh": "integer"
}
signatures["torch.quantized_rnn_relu_cell"] = {
    "input_size": "integer",
    "hidden_size": "integer",
    "input": "tensor",
    "hx": "tensor",
    "w_ih": "tensor",
    "w_hh": "tensor",
    "b_ih": "tensor",
    "b_hh": "tensor",
    "scale_ih": "float",
    "scale_hh": "float",
    "zero_point_ih": "integer",
    "zero_point_hh": "integer"
}
signatures["torch.quantized_rnn_tanh_cell"] = {
    "input": "tensor",
    "hx": "tensor",
    "w_ih": "tensor",
    "w_hh": "tensor",
    "b_ih": "tensor",
    "b_hh": "tensor",
    "packed_params": "tensor" # Assuming packed_params is a tensor
}
signatures["torch.quasirandom.SobolEngine"] = {
    "dimension": "integer",
    "scramble": "boolean",
    "seed": "integer"
}
signatures["torch.quasirandom.SobolEngine.reset"] = {
    "seed": "integer"
}
signatures["torch.quasirandom.SobolEngine.forward"] = {
    "n": "integer"
}
signatures["torch.quasirandom.SobolEngine.draw"] = {
    "n": "integer",
    "dtype": "dtype" # might also be a tensor but dtype seems more accurate
}
signatures["torch.quasirandom.SobolEngine.i4_sobol_generate"] = {
    "n": "integer",
    "dim_num": "integer",
    "skip": "integer"
}
signatures["torch.rand"] = {
    "size": "tuple", # Could also be integer, but tuple/list is more general for shape
    "generator": "torch.Generator", # Should probably be a more general type, like "object"
    "dtype": "dtype",
    "layout": "torch.layout", # Should probably be a more general type, like "object"
    "requires_grad": "boolean",
    "pin_memory": "boolean"
}
signatures["torch.rand_like"] = {
    "input": "tensor",
    "dtype": "dtype",
    "layout": "string", # Should ideally be torch.layout, but string is the closest
    "requires_grad": "boolean",
    "memory_format": "string" # Should ideally be torch.memory_format, but string is the closest
}
signatures["torch.randint"] = {
    "low": "integer",
    "high": "integer",
    "size": "tuple",
    "generator": "tensor", #torch.Generator is actually a class, but can be thought of as a tensor
    "dtype": "dtype",
    "layout": "string",
    "requires_grad": "boolean"
}
signatures["torch.randint_like"] = {
    "input": "tensor",
    "low": "integer",
    "high": "integer",
    "dtype": "dtype",
    "layout": "string", # could also be layout but most users will just use string for the layout name.
    "generator": "tensor",
}
signatures["torch.randn"] = {
    "size": "tuple", # Could also be a list of integers, but tuple seems more appropriate given the description
    "generator": "torch.Generator", # This is not one of the options, but torch.Generator is the closest
    "dtype": "dtype",
    "layout": "torch.layout", # This is not one of the options, but torch.layout is the closest
    "requires_grad": "boolean",
    "pin_memory": "boolean"
}
signatures["torch.randn_like"] = {
    "input": "tensor",
    "dtype": "dtype",
    "layout": "string", # Could be torch.layout but string seems more appropriate based on examples.
    "requires_grad": "boolean",
    "memory_format": "string" # Could be torch.memory_format but string seems more appropriate based on examples.
}
signatures["torch.random.default_generator"] = {} # This function has no arguments
signatures["torch.random.get_rng_state"] = {} # This function has no arguments
signatures["torch.random.manual_seed"] = {
    "seed": "integer"
}
signatures["torch.random.seed"] = {} # This function has no arguments
signatures["torch.random.set_rng_state"] = {
    "new_state": "tensor"
}
signatures["torch.random.initial_seed"] = {} # This function has no arguments
signatures["torch.randperm"] = {
    "n": "integer",
    "generator": "tensor", # Could also be torch.Generator, but "tensor" is the closest allowed type
    "dtype": "dtype",
    "layout": "string",
    "requires_grad": "boolean",
    "pin_memory": "boolean"
}
signatures["torch.ravel"] = {
    "input": "tensor"
}
signatures["torch.read_vitals"] = {
    "path": "string",
    "devices": "list", #Could be a list of strings, unsure
    "timeout": "float",
    "retry_count": "integer"
}
signatures["torch.real"] = {
    "input": "tensor"
}
signatures["torch.reciprocal"] = {
    "input": "tensor"
}
signatures["torch.reciprocal_"] = {
    "input": "tensor"
}
signatures["torch.relu"] = {
    "input": "tensor"
}
signatures["torch.relu_"] = {
    "input": "tensor"
}
signatures["torch.renorm"] = {
    "input": "tensor",
    "p": "float",
    "dim": "integer",
    "maxnorm": "float"
}
signatures["torch.repeat_interleave"] = {
    "input": "tensor",
    "repeats": "tensor", # Could be list or integer but commonly tensor
    "dim": "integer"
}
signatures["torch.reshape"] = {
    "input": "tensor",
    "shape": "tuple"
}
signatures["torch.resize_as_"] = {
    "input": "tensor",
    "target": "tensor"
}
signatures["torch.resize_as_sparse_"] = {
    "input": "tensor",
    "tensor": "tensor"
}
signatures["torch.resolve_conj"] = {
    "x": "tensor"
}
signatures["torch.resolve_neg"] = {
    "input": "tensor"
}
signatures["torch.result_type"] = {
    "tensor1": "tensor",
    "tensor2": "tensor"
}
signatures["torch.return_types.namedtuple"] = { # Most common use is with tensors
    "input": "tensor" # Generic input
}
signatures["torch.rms_norm"] = {
    "input": "tensor",
    "weight": "tensor",
    "normalized_shape": "list", # Could also be tuple, but list is more common
    "eps": "float",
    "bias": "tensor"
}
signatures["torch.rnn_relu"] = {
    "input": "tensor",
    "h_0": "tensor",
    "num_layers": "integer",
    "dropout": "float",
    "train": "boolean",
    "bidirectional": "boolean",
    "batch_first": "boolean"
}
signatures["torch.rnn_relu_cell"] = {
    "input": "tensor",
    "hx": "tensor",
    "w_ih": "tensor",
    "w_hh": "tensor",
    "b_ih": "tensor",
    "b_hh": "tensor"
}
signatures["torch.rnn_tanh"] = {
    "input": "tensor",
    "hx": "tensor",
    "cx": "tensor" # There is no cx argument in rnn_tanh according to torch documentation. Assuming it's a typo for hx
}
signatures["torch.rnn_tanh_cell"] = {
    "input": "tensor",
    "hx": "tensor",
    "weight_ih": "tensor",
    "weight_hh": "tensor",
    "bias_ih": "tensor",
    "bias_hh": "tensor"
}
signatures["torch.roll"] = {
    "input": "tensor",
    "shifts": "tuple", # Could also be an integer but tuple seems more common
    "dims": "tuple" # Could also be an integer but tuple seems more common
}
signatures["torch.round"] = {
    "input": "tensor",
    "decimals": "integer"
}
signatures["torch.round_"] = {
    "input": "tensor"
}
signatures["torch.row_indices_copy"] = {
    "input": "tensor",
    "row_indices": "tensor",
    "source": "tensor"
}
signatures["torch.row_stack"] = {
    "tensors": "tensor_list"
}
signatures["torch.rrelu"] = {
    "input": "tensor",
    "lower": "float",
    "upper": "float",
    "training": "boolean",
    "inplace": "boolean"
}
signatures["torch.rrelu_"] = {
    "input": "tensor",
    "lower": "float",
    "upper": "float",
    "training": "boolean"
}
signatures["torch.rsqrt"] = {
    "input": "tensor"
}
signatures["torch.rsqrt_"] = {
    "input": "tensor"
}
signatures["torch.rsub"] = {
    "input": "tensor",
    "other": "tensor", # could also be a float or integer, but tensor is more general and common
    "alpha": "float" # could also be an integer, but float is more general and common
}
signatures["torch.saddmm"] = {
    "input": "tensor",
    "mat1": "tensor",
    "mat2": "tensor",
    "alpha": "float",
    "beta": "float"
}
signatures["torch.save"] = {
    "obj": "object", # Could potentially be "tensor" or a more specific object type depending on common usage
    "f": "string", # Could also be a file-like object, but string is more commonly used as the path
    "pickle_module": "object",
    "pickle_protocol": "integer",
    "_use_new_zipfile_serialization": "boolean"
}
signatures["torch.scalar_tensor"] = {
    "s": "float", # could also accept integer, but float is more general
    "dtype": "dtype",
    "layout": "string", #best match to torch.layout
    "requires_grad": "boolean"
}
signatures["torch.scatter"] = {
    "input": "tensor",
    "dim": "integer",
    "index": "tensor",
    "src": "tensor"
}
signatures["torch.scatter_reduce"] = {
    "input": "tensor",
    "dim": "integer",
    "index": "tensor",
    "src": "tensor",
    "reduce": "string",
    "include_self": "boolean"
}
signatures["torch.seed"] = {}
signatures["torch.segment_reduce"] = {
    "data": "tensor",
    "segment_ids": "tensor",
    "reduce": "string",
    "init": "float" # Could also be tensor, but float seems more common
}
signatures["torch.select"] = {
    "input": "tensor",
    "dim": "integer",
    "index": "integer"
}
signatures["torch.select_copy"] = {
    "input": "tensor",
    "dim": "integer",
    "index": "integer"
}
signatures["torch.select_scatter"] = {
    "input": "tensor",
    "src": "tensor",
    "dim": "integer",
    "index": "tensor"
}
signatures["torch.selu"] = {
    "input": "tensor"
}
signatures["torch.selu_"] = {
    "input": "tensor"
}
signatures["torch.serialization.save"] = {
    "obj": "tensor", # Can be also a module or dict, but tensor is most common
    "f": "string", # Can also be a file-like object, but string (filepath) is most common
    "pickle_module": "string",
    "pickle_protocol": "integer",
    "_use_new_zipfile_serialization": "boolean"
}
signatures["torch.serialization.load"] = {
    "f": "string", # Can also be a file-like object, but string (filepath) is most common
    "map_location": "string", # Can also be a function or a dict, but string is most common
    "pickle_module": "string",
    "pickle_protocol": "integer",
    "weights_only": "boolean",
    "mmap": "boolean"
}
signatures["torch.set_anomaly_enabled"] = {
    "mode": "boolean"
}
signatures["torch.set_autocast_cache_enabled"] = {
    "enabled": "boolean"
}
signatures["torch.set_autocast_cpu_dtype"] = {
    "dtype": "dtype"
}
signatures["torch.set_autocast_cpu_enabled"] = {
    "enabled": "boolean"
}
signatures["torch.set_autocast_dtype"] = {
    "dtype": "dtype"
}
signatures["torch.set_autocast_enabled"] = {
    "enabled": "boolean"
}
signatures["torch.set_autocast_gpu_dtype"] = {
    "dtype": "dtype"
}
signatures["torch.set_autocast_ipu_dtype"] = {
    "dtype": "dtype"
}
signatures["torch.set_autocast_ipu_enabled"] = {
    "enabled": "boolean"
}
signatures["torch.set_autocast_xla_dtype"] = {
    "dtype": "dtype"
}
signatures["torch.set_autocast_xla_enabled"] = {
    "enabled": "boolean"
}
signatures["torch.set_default_device"] = {
    "device": "string" # Could also be "device" type but "string" is more commonly used.
}
signatures["torch.set_default_dtype"] = {
    "d": "dtype"
}
signatures["torch.set_default_tensor_type"] = {
    "t": "string" # should ideally be a dtype, but the documentation mentions the string representation of the type.
}
signatures["torch.set_deterministic_debug_mode"] = {
    "mode": "string" # Could also be an enum-like integer, but string is more descriptive
}
signatures["torch.set_flush_denormal"] = {
    "mode": "boolean"
}
signatures["torch.set_grad_enabled"] = {
    "mode": "boolean"
}
signatures["torch.set_num_interop_threads"] = {
    "num_threads": "integer"
}
signatures["torch.set_num_threads"] = {
    "threads": "integer"
}
signatures["torch.set_printoptions"] = {
    "precision": "integer",
    "threshold": "integer",
    "edgeitems": "integer",
    "linewidth": "integer",
    "line_length": "integer", # line_length is deprecated, linewidth is used instead, keeping integer type as linewidth
    "profile": "string",
    "sci_mode": "boolean"
}
signatures["torch.set_rng_state"] = {
    "new_state": "tensor"
}
signatures["torch.set_vital"] = {
    "input": "tensor",
    "vital": "boolean"
}
signatures["torch.set_warn_always"] = {
    "warn_always": "boolean"
}
signatures["torch.sigmoid_"] = {
    "input": "tensor"
}
signatures["torch.sign"] = {
    "input": "tensor"
}
signatures["torch.signal.stft"] = {
    "input": "tensor",
    "n_fft": "integer",
    "hop_length": "integer",
    "win_length": "integer",
    "window": "tensor",
    "center": "boolean",
    "pad_mode": "string",
    "normalized": "boolean",
    "onesided": "boolean",
    "return_complex": "boolean"
}
signatures["torch.signal.istft"] = {
    "input": "tensor",
    "n_fft": "integer",
    "hop_length": "integer",
    "win_length": "integer",
    "window": "tensor",
    "center": "boolean",
    "normalized": "boolean",
    "onesided": "boolean",
    "length": "integer",
    "return_complex": "boolean"
}
signatures["torch.signal.windows.hann"] = {
    "window_length": "integer",
    "periodic": "boolean",
    "dtype": "dtype",
    "requires_grad": "boolean"
}
signatures["torch.signal.windows.hamming"] = {
    "window_length": "integer",
    "periodic": "boolean",
    "alpha": "float",
    "dtype": "dtype",
    "requires_grad": "boolean"
}
signatures["torch.signal.windows.bartlett"] = {
    "window_length": "integer",
    "periodic": "boolean",
    "dtype": "dtype",
    "requires_grad": "boolean"
}
signatures["torch.signal.windows.blackman"] = {
    "window_length": "integer",
    "periodic": "boolean",
    "dtype": "dtype",
    "requires_grad": "boolean"
}
signatures["torch.sin"] = {
    "input": "tensor"
}
signatures["torch.sin_"] = {
    "input": "tensor"
}
signatures["torch.sinc"] = {
    "input": "tensor"
}
signatures["torch.sinc_"] = {
    "input": "tensor"
}
signatures["torch.sinh"] = {
    "input": "tensor"
}
signatures["torch.sinh_"] = {
    "input": "tensor"
}
signatures["torch.slice_copy"] = {
    "input": "tensor",
    "indices": "list", # Could also be tuple, but list seems more appropriate for indices
    "value": "tensor"
}
signatures["torch.slice_inverse"] = {
    "input": "tensor",
    "indices": "list", # Could potentially be tuple as well, but list seems more common
    "dim": "integer",
    "slice_size": "integer"
}
signatures["torch.slice_scatter"] = {
    "input": "tensor",
    "src": "tensor",
    "dims": "list", # Could be a tuple instead of list
    "starts": "list", # Could be a tuple instead of list
    "ends": "list" # Could be a tuple instead of list
}
signatures["torch.smm"] = {
    "input": "tensor",
    "mat": "tensor"
}
signatures["torch.softmax"] = {
    "input": "tensor",
    "dim": "integer",
    "dtype": "dtype"
}
signatures["torch.sort"] = {
    "input": "tensor",
    "dim": "integer",
    "descending": "boolean",
    "stable": "boolean"
}
signatures["torch.sparse.add"] = {
    "input": "tensor",
    "other": "tensor",
    "alpha": "float"
}
signatures["torch.sparse.softmax"] = {
    "input": "tensor",
    "dim": "integer",
    "dtype": "dtype"
}
signatures["torch.sparse.sum"] = {
    "input": "tensor",
    "dim": "list", # Could be integer or tuple as well, but list seems most general
    "keepdim": "boolean",
    "dtype": "dtype"
}
signatures["torch.sparse.mm"] = {
    "input": "tensor",
    "mat2": "tensor"
}
signatures["torch.sparse.sampled_addmm"] = {
    "input": "tensor",
    "mat1": "tensor",
    "mat2": "tensor",
    "beta": "float",
    "alpha": "float"
}
signatures["torch.sparse.mul"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.sparse.div"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.sparse.fill_"] = {
    "input": "tensor",
    "value": "float"
}
signatures["torch.sparse.threshold"] = {
    "input": "tensor",
    "threshold": "float",
    "value": "float"
}
signatures["torch.sparse.maximum"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.sparse.minimum"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.sparse_bsc_tensor"] = {
    "crow_indices": "tensor",
    "col_indices": "tensor",
    "values": "tensor",
    "size": "tuple",
    "dtype": "dtype",
    "requires_grad": "boolean" # Could also be None, boolean seems like the closest type
}
signatures["torch.sparse_bsr_tensor"] = {
    "crow_indices": "tensor",
    "col_indices": "tensor",
    "values": "tensor",
    "size": "tuple",
    "blocksize": "tuple",
    "dtype": "dtype",
    "requires_grad": "boolean"
}
signatures["torch.sparse_compressed_tensor"] = {
    "compressed_indices": "tensor",
    "plain_indices": "tensor",
    "values": "tensor",
    "size": "tuple",
    "sparse_layout": "string", # could be an enum but string is safer
    "dtype": "dtype",
    "requires_grad": "boolean"
}
signatures["torch.sparse_csc_tensor"] = {
    "crow_indices": "tensor",
    "col_indices": "tensor",
    "values": "tensor",
    "size": "tuple", # could also be integer, but tuple is more common for size
    "dtype": "dtype",
    "requires_grad": "boolean"
}
signatures["torch.sparse_csr_tensor"] = {
    "crow_indices": "tensor",
    "col_indices": "tensor",
    "values": "tensor",
    "size": "tuple",
    "dtype": "dtype",
    "requires_grad": "boolean"
}
signatures["torch.special.airy_ai"] = {
    "x": "tensor"
}
signatures["torch.special.airy_bi"] = {
    "x": "tensor"
}
signatures["torch.special.airy_ai_zeros"] = {
    "n": "integer"
}
signatures["torch.special.airy_bi_zeros"] = {
    "n": "integer"
}
signatures["torch.special.bessel_j0"] = {
    "x": "tensor"
}
signatures["torch.special.bessel_j1"] = {
    "x": "tensor"
}
signatures["torch.special.bessel_y0"] = {
    "x": "tensor"
}
signatures["torch.special.bessel_y1"] = {
    "x": "tensor"
}
signatures["torch.special.bessel_jn"] = {
    "n": "integer",
    "x": "tensor"
}
signatures["torch.special.bessel_yn"] = {
    "n": "integer",
    "x": "tensor"
}
signatures["torch.special.beta"] = {
    "x": "tensor",
    "y": "tensor"
}
signatures["torch.special.binominal"] = { # Should this be binomial?
    "n": "tensor",
    "count": "tensor"
}
signatures["torch.special.digamma"] = {
    "x": "tensor"
}
signatures["torch.special.erf"] = {
    "x": "tensor"
}
signatures["torch.special.erfc"] = {
    "x": "tensor"
}
signatures["torch.special.erfinv"] = {
    "y": "tensor"
}
signatures["torch.special.expi"] = {
    "x": "tensor"
}
signatures["torch.special.expm1"] = {
    "x": "tensor"
}
signatures["torch.special.gammainc"] = {
    "a": "tensor",
    "x": "tensor"
}
signatures["torch.special.igamma"] = {
    "a": "tensor",
    "x": "tensor"
}
signatures["torch.special.gammaincc"] = {
    "a": "tensor",
    "x": "tensor"
}
signatures["torch.special.igammac"] = {
    "a": "tensor",
    "x": "tensor"
}
signatures["torch.special.gammaln"] = {
    "x": "tensor"
}
signatures["torch.special.i0"] = {
    "x": "tensor"
}
signatures["torch.special.i0e"] = {
    "x": "tensor"
}
signatures["torch.special.i1"] = {
    "x": "tensor"
}
signatures["torch.special.i1e"] = {
    "x": "tensor"
}
signatures["torch.special.log_ndtr"] = {
    "x": "tensor"
}
signatures["torch.special.logit"] = {
    "x": "tensor",
    "eps": "float"
}
signatures["torch.special.ndtr"] = {
    "x": "tensor"
}
signatures["torch.special.ndtri"] = {
    "p": "tensor"
}
signatures["torch.special.polygamma"] = {
    "n": "integer",
    "x": "tensor"
}
signatures["torch.special.psi"] = {
    "x": "tensor"
}
signatures["torch.special.round"] = { # Is this the correct signature? There's also 'decimals'
    "input": "tensor"
}
signatures["torch.special.sinc"] = {
    "x": "tensor"
}
signatures["torch.special.spherical_bessel_j0"] = {
    "x": "tensor"
}
signatures["torch.special.zeta"] = {
    "x": "tensor",
    "q": "tensor"
}
signatures["torch.special.airy_ai"] = {
    "x": "tensor"
}
signatures["torch.special.airy_bi"] = {
    "x": "tensor"
}
signatures["torch.special.airy_ai_zeros"] = {
    "n": "integer"
}
signatures["torch.special.airy_bi_zeros"] = {
    "n": "integer"
}
signatures["torch.special.bessel_j0"] = {
    "x": "tensor"
}
signatures["torch.special.bessel_j1"] = {
    "x": "tensor"
}
signatures["torch.special.bessel_y0"] = {
    "x": "tensor"
}
signatures["torch.special.bessel_y1"] = {
    "x": "tensor"
}
signatures["torch.special.beta"] = {
    "x": "tensor",
    "y": "tensor"
}
signatures["torch.special.digamma"] = {
    "x": "tensor"
}
signatures["torch.special.erf"] = {
    "x": "tensor"
}
signatures["torch.special.erfc"] = {
    "x": "tensor"
}
signatures["torch.special.erfinv"] = {
    "x": "tensor"
}
signatures["torch.special.expit"] = {
    "x": "tensor"
}
signatures["torch.special.expm1"] = {
    "x": "tensor"
}
signatures["torch.special.gammainc"] = {
    "a": "tensor",
    "x": "tensor"
}
signatures["torch.special.gammaincc"] = {
    "a": "tensor",
    "x": "tensor"
}
signatures["torch.special.gammaln"] = {
    "x": "tensor"
}
signatures["torch.special.i0"] = {
    "x": "tensor"
}
signatures["torch.special.i0e"] = {
    "x": "tensor"
}
signatures["torch.special.i1"] = {
    "x": "tensor"
}
signatures["torch.special.i1e"] = {
    "x": "tensor"
}
signatures["torch.special.log_ndtr"] = {
    "x": "tensor"
}
signatures["torch.special.logsumexp"] = {
    "input": "tensor",
    "dim": "list", # Could be int too. Choosing list as it can accept list of ints
    "keepdim": "boolean"
}
signatures["torch.special.ndtr"] = {
    "x": "tensor"
}
signatures["torch.special.ndtri"] = {
    "x": "tensor"
}
signatures["torch.special.polygamma"] = {
    "n": "integer",
    "x": "tensor"
}
signatures["torch.special.psi"] = {
    "x": "tensor"
}
signatures["torch.special.round"] = {
    "input": "tensor",
    "decimals": "integer"
}
signatures["torch.special.sinc"] = {
    "x": "tensor"
}
signatures["torch.special.softmax"] = {
    "input": "tensor",
    "dim": "integer",
    "dtype": "dtype"
}
signatures["torch.special.airy_ai"] = {
    "x": "tensor"
}
signatures["torch.special.bessel_j0"] = {
    "input": "tensor"
}
signatures["torch.special.bessel_j1"] = {
    "input": "tensor"
}
signatures["torch.special.bessel_y0"] = {
    "input": "tensor"
}
signatures["torch.special.bessel_y1"] = {
    "input": "tensor"
}
signatures["torch.special.chebyshev_polynomial_t"] = {
    "n": "integer",
    "x": "tensor"
}
signatures["torch.special.chebyshev_polynomial_u"] = {
    "n": "integer",
    "x": "tensor"
}
signatures["torch.special.chebyshev_polynomial_v"] = {
    "n": "integer",
    "x": "tensor"
}
signatures["torch.special.chebyshev_polynomial_w"] = {
    "n": "integer",
    "x": "tensor"
}
signatures["torch.special.digamma"] = {
    "input": "tensor"
}
signatures["torch.special.entr"] = {
    "input": "tensor"
}
signatures["torch.special.erf"] = {
    "input": "tensor"
}
signatures["torch.special.erfc"] = {
    "input": "tensor"
}
signatures["torch.special.erfcx"] = {
    "input": "tensor"
}
signatures["torch.special.erfinv"] = {
    "input": "tensor"
}
signatures["torch.special.exp2"] = {
    "input": "tensor"
}
signatures["torch.special.expit"] = {
    "input": "tensor"
}
signatures["torch.special.expm1"] = {
    "input": "tensor"
}
signatures["torch.special.gammainc"] = {
    "a": "tensor",
    "x": "tensor"
}
signatures["torch.special.gammaincc"] = {
    "a": "tensor",
    "x": "tensor"
}
signatures["torch.special.gammaln"] = {
    "input": "tensor"
}
signatures["torch.special.hermite_polynomial_h"] = {
    "n": "integer",
    "x": "tensor"
}
signatures["torch.special.hermite_polynomial_he"] = {
    "x": "tensor",
    "n": "integer" # Could be tensor as well, but integer is the most common.
}
signatures["torch.special.i0"] = {
    "input": "tensor"
}
signatures["torch.special.i0e"] = {
    "input": "tensor"
}
signatures["torch.special.i1"] = {
    "input": "tensor"
}
signatures["torch.special.i1e"] = {
    "input": "tensor"
}
signatures["torch.special.laguerre_polynomial_l"] = {
    "n": "integer",
    "x": "tensor"
}
signatures["torch.special.legendre_polynomial_p"] = {
    "n": "integer",
    "x": "tensor"
}
signatures["torch.special.log1p"] = {
    "input": "tensor"
}
signatures["torch.special.log_ndtr"] = {
    "input": "tensor"
}
signatures["torch.special.log_softmax"] = {
    "input": "tensor",
    "dim": "integer",
    "dtype": "dtype" # could also be None, but dtype is more appropriate
}
signatures["torch.special.logit"] = {
    "input": "tensor",
    "eps": "float" # or None, but float is more common
}
signatures["torch.special.logsumexp"] = {
    "input": "tensor",
    "dim": "list", # Can also accept integer, but list is more general as tuple is a list
    "keepdim": "boolean"
}
signatures["torch.special.modified_bessel_i0"] = {
    "input": "tensor"
}
signatures["torch.special.modified_bessel_i1"] = {
    "input": "tensor"
}
signatures["torch.special.modified_bessel_k0"] = {
    "input": "tensor"
}
signatures["torch.special.modified_bessel_k1"] = {
    "input": "tensor"
}
signatures["torch.special.multigammaln"] = {
    "input": "tensor",
    "p": "integer",
    "safe": "boolean"
}
signatures["torch.special.ndtr"] = {
    "input": "tensor"
}
signatures["torch.special.ndtri"] = {
    "input": "tensor"
}
signatures["torch.special.polygamma"] = {
    "n": "integer",
    "input": "tensor"
}
signatures["torch.special.psi"] = {
    "input": "tensor"
}
signatures["torch.special.round"] = {
    "input": "tensor"
}
signatures["torch.special.scaled_modified_bessel_k0"] = {
    "x": "tensor"
}
signatures["torch.special.scaled_modified_bessel_k1"] = {
    "x": "tensor"
}
signatures["torch.special.shifted_chebyshev_polynomial_t"] = {
    "n": "integer",
    "x": "tensor"
}
signatures["torch.special.shifted_chebyshev_polynomial_u"] = {
    "n": "integer",
    "x": "tensor"
}
signatures["torch.special.shifted_chebyshev_polynomial_v"] = {
    "n": "integer",
    "x": "tensor"
}
signatures["torch.special.shifted_chebyshev_polynomial_w"] = {
    "n": "integer",
    "x": "tensor"
}
signatures["torch.special.sinc"] = {
    "input": "tensor"
}
signatures["torch.special.softmax"] = {
    "input": "tensor",
    "dim": "integer",
    "dtype": "dtype"
}
signatures["torch.special.spherical_bessel_j0"] = {
    "input": "tensor"
}
signatures["torch.special.airy_ai"] = {
    "x": "tensor"
}
signatures["torch.special.airy_bi"] = {
    "x": "tensor"
}
signatures["torch.special.airy_aiprime"] = {
    "x": "tensor"
}
signatures["torch.special.airy_biprime"] = {
    "x": "tensor"
}
signatures["torch.special.bessel_j0"] = {
    "x": "tensor"
}
signatures["torch.special.bessel_j1"] = {
    "x": "tensor"
}
signatures["torch.special.bessel_y0"] = {
    "x": "tensor"
}
signatures["torch.special.bessel_y1"] = {
    "x": "tensor"
}
signatures["torch.special.betainc"] = {
    "a": "tensor",
    "b": "tensor",
    "x": "tensor"
}
signatures["torch.special.expi"] = {
    "x": "tensor"
}
signatures["torch.special.expm1"] = {
    "x": "tensor"
}
signatures["torch.special.digamma"] = {
    "x": "tensor"
}
signatures["torch.special.erf"] = {
    "x": "tensor"
}
signatures["torch.special.erfc"] = {
    "x": "tensor"
}
signatures["torch.special.erfinv"] = {
    "x": "tensor"
}
signatures["torch.special.exp2"] = {
    "x": "tensor"
}
signatures["torch.special.gammainc"] = {
    "a": "tensor",
    "x": "tensor"
}
signatures["torch.special.gammaincc"] = {
    "a": "tensor",
    "x": "tensor"
}
signatures["torch.special.gammaln"] = {
    "x": "tensor"
}
signatures["torch.special.i0"] = {
    "x": "tensor"
}
signatures["torch.special.i0e"] = {
    "x": "tensor"
}
signatures["torch.special.i1"] = {
    "x": "tensor"
}
signatures["torch.special.i1e"] = {
    "x": "tensor"
}
signatures["torch.special.log_ndtr"] = {
    "x": "tensor"
}
signatures["torch.special.logsumexp"] = {
    "input": "tensor",
    "dim": "integer",
    "keepdim": "boolean"
}
signatures["torch.special.ndtr"] = {
    "x": "tensor"
}
signatures["torch.special.ndtri"] = {
    "x": "tensor"
}
signatures["torch.special.polygamma"] = {
    "n": "integer",
    "x": "tensor"
}
signatures["torch.special.psi"] = {
    "x": "tensor"
}
signatures["torch.special.round"] = {
    "input": "tensor",
    "decimals": "integer"
}
signatures["torch.special.sinc"] = {
    "x": "tensor"
}
signatures["torch.special.softmax"] = {
    "input": "tensor",
    "dim": "integer",
    "dtype": "dtype"
}
signatures["torch.special.logit"] = {
    "x": "tensor",
    "eps": "float"
}
signatures["torch.special.zeta"] = {
    "x": "tensor",
    "q": "tensor"
}
signatures["torch.special.modified_bessel_i0"] = {
    "x": "tensor"
}
signatures["torch.special.modified_bessel_i1"] = {
    "x": "tensor"
}
signatures["torch.special.modified_bessel_k0"] = {
    "x": "tensor"
}
signatures["torch.special.modified_bessel_k1"] = {
    "x": "tensor"
}
signatures["torch.special.shifted_modified_bessel_k0"] = {
    "x": "tensor"
}
signatures["torch.special.shifted_modified_bessel_k1"] = {
    "x": "tensor"
}
signatures["torch.special.scaled_modified_bessel_k0"] = {
    "x": "tensor"
}
signatures["torch.special.scaled_modified_bessel_k1"] = {
    "x": "tensor"
}
signatures["torch.special.xlog1py"] = {
    "x": "tensor",
    "y": "tensor"
}
signatures["torch.special.xlogy"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.special.zeta"] = {
    "x": "tensor",
    "q": "tensor"
}
signatures["torch.split_copy"] = {
    "input": "tensor",
    "split_size": "integer", # could be list or integer, integer is probably more common
    "dim": "integer"
}
signatures["torch.split_with_sizes"] = {
    "tensor": "tensor",
    "split_size_or_sections": "list", # Could also be an integer in some cases, but list is more common
    "dim": "integer"
}
signatures["torch.split_with_sizes_copy"] = {
    "input": "tensor",
    "split_sizes": "list", # Could also be a tuple, but list is more common.
    "dim": "integer"
}
signatures["torch.spmm"] = {
    "input": "tensor",
    "mat2": "tensor"
}
signatures["torch.sqrt"] = {
    "input": "tensor"
}
signatures["torch.sqrt_"] = {
    "input": "tensor"
}
signatures["torch.square"] = {
    "input": "tensor"
}
signatures["torch.square_"] = {
    "input": "tensor"
}
signatures["torch.squeeze_copy"] = {
    "input": "tensor",
    "dim": "integer" # Could also accept tuple of integers, but integer is more common
}
signatures["torch.sspaddmm"] = {
    "input": "tensor",
    "mat1": "tensor",
    "mat2": "tensor",
    "beta": "float", # Could also be integer, but float seems more general
    "alpha": "float" # Could also be integer, but float seems more general
}
signatures["torch.stack"] = {
    "tensors": "tensor_list",
    "dim": "integer"
}
signatures["torch.stft"] = {
    "input": "tensor",
    "n_fft": "integer",
    "hop_length": "integer",
    "win_length": "integer",
    "window": "tensor",
    "center": "boolean",
    "pad_mode": "string",
    "normalized": "boolean",
    "onesided": "boolean",
    "return_complex": "boolean",
    "align_to_window": "boolean" # Could also be None, but boolean is more common.
}
signatures["torch.storage"] = {
    "size": "integer",
    "dtype": "dtype",
    "allocator": "string" #Could be a function but string is closest
}
signatures["torch.subtract"] = {
    "input": "tensor",
    "other": "tensor",
    "alpha": "float" # Could also be integer, but float seems more common given the context
}
signatures["torch.sum"] = {
    "input": "tensor",
    "dim": "tuple", # Could also be an integer, but tuple is more general
    "keepdim": "boolean",
    "dtype": "dtype"
}
signatures["torch.svd"] = {
    "input": "tensor",
    "some": "boolean",
    "compute_uv": "boolean"
}
signatures["torch.svd_lowrank"] = {
    "input": "tensor",
    "q": "integer",
    "center": "boolean",
    "niter": "integer"
}
signatures["torch.swapaxes"] = {
    "input": "tensor",
    "axis0": "integer",
    "axis1": "integer"
}
signatures["torch.swapdims"] = {
    "input": "tensor",
    "dim0": "integer",
    "dim1": "integer"
}
signatures["torch.sym_constrain_range"] = {
    "sym_sizes": "list", # I'm not sure if this is a list of strings or integers, so choosing list
    "min_val": "integer",
    "max_val": "integer"
}
signatures["torch.sym_constrain_range_for_size"] = {
    "size": "integer", # Could also be a SymInt or integer
    "min": "integer",
    "max": "integer"
}
signatures["torch.sym_float"] = {
    "a": "tensor"
}
signatures["torch.sym_fresh_size"] = {
    "name": "string"
}
signatures["torch.sym_int"] = {
    "a": "integer" # Could also be string, representing symbolic int but integer seems to be the base case.
}
signatures["torch.sym_ite"] = {
    "cond": "tensor",
    "true_val": "tensor",
    "false_val": "tensor"
}
signatures["torch.sym_max"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.sym_min"] = {
    "input": "tensor",
    "other": "tensor" # Could also be a float or integer, but tensor seems more common
}
signatures["torch.sym_not"] = {
    "input": "tensor"
}
signatures["torch.sym_sqrt"] = {
    "input": "tensor"
}
signatures["torch.sym_sum"] = {
    "tensors": "tensor_list"
}
signatures["torch.symeig"] = {
    "input": "tensor",
    "eigenvectors": "boolean",
    "upper": "boolean"
}
signatures["torch.sys.float_info"] = {} # This function takes no arguments
signatures["torch.sys.int_info"] = {} # This function takes no arguments
signatures["torch.sys.getwindowsversion"] = {} # This function takes no arguments
signatures["torch.sys.available"] = {} # This function takes no arguments
signatures["torch.sys.cpu_count"] = {} # This function takes no arguments
signatures["torch.sys.flags"] = {} # This function takes no arguments
signatures["torch.sys.version"] = {} # This function takes no arguments
signatures["torch.sys.platform"] = {} # This function takes no arguments
signatures["torch.sys.executable"] = {} # This function takes no arguments
signatures["torch.sys.prefix"] = {} # This function takes no arguments
signatures["torch.sys.getdlopenflags"] = {} # This function takes no arguments
signatures["torch.sys.setdlopenflags"] = {
    "flags": "integer"
}
signatures["torch.sys.stderr"] = {} # This function returns a value, takes no arguments.
signatures["torch.sys.stdin"] = {} # This function returns a value, takes no arguments.
signatures["torch.sys.stdout"] = {} # This function returns a value, takes no arguments.
signatures["torch.t"] = {
    "input": "tensor"
}
signatures["torch.t_copy"] = {
    "self": "tensor"
}
signatures["torch.take"] = {
    "input": "tensor",
    "index": "tensor" # index is LongTensor, but "tensor" is the closest type
}
signatures["torch.take_along_dim"] = {
    "input": "tensor",
    "indices": "tensor",
    "dim": "integer"
}
signatures["torch.tan_"] = {
    "input": "tensor"
}
signatures["torch.tanh"] = {
    "input": "tensor"
}
signatures["torch.tanh_"] = {
    "input": "tensor"
}
signatures["torch.tensor_split"] = {
    "tensor": "tensor",
    "indices_or_sections": "integer", # Could also be a list or tuple
    "dim": "integer"
}
signatures["torch.testing.assert_close"] = {
    "actual": "tensor",
    "expected": "tensor",
    "rtol": "float",
    "atol": "float",
    "equal_nan": "boolean",
    "check_device": "boolean",
    "check_dtype": "boolean"
}
signatures["torch.textwrap.wrap"] = {
    "text": "string",
    "width": "integer",
    "initial_indent": "string",
    "subsequent_indent": "string",
    "expand_tabs": "boolean",
    "replace_whitespace": "boolean",
    "drop_whitespace": "boolean",
    "break_long_words": "boolean",
    "break_on_hyphens": "boolean",
    "max_lines": "integer",
    "placeholder": "string"
}
signatures["torch.textwrap.fill"] = {
    "text": "string",
    "width": "integer",
    "initial_indent": "string",
    "subsequent_indent": "string",
    "expand_tabs": "boolean",
    "replace_whitespace": "boolean",
    "drop_whitespace": "boolean",
    "break_long_words": "boolean",
    "break_on_hyphens": "boolean",
    "max_lines": "integer",
    "placeholder": "string"
}
signatures["torch.textwrap.shorten"] = {
    "text": "string",
    "width": "integer",
    "placeholder": "string",
    "break_long_words": "boolean",
    "drop_whitespace": "boolean",
    "replace_whitespace": "boolean"
}
signatures["torch.threading.get_num_threads"] = {}
signatures["torch.threading.set_num_threads"] = {
    "threads": "integer"
}
signatures["torch.threshold"] = {
    "input": "tensor",
    "threshold": "float",
    "value": "float"
}
signatures["torch.threshold_"] = {
    "input": "tensor",
    "threshold": "float",
    "value": "float"
}
signatures["torch.tile"] = {
    "input": "tensor",
    "dims": "tuple"
}
signatures["torch.to_dlpack"] = {
    "input": "tensor"
}
signatures["torch.topk"] = {
    "input": "tensor",
    "k": "integer",
    "dim": "integer",
    "largest": "boolean",
    "sorted": "boolean"
}
signatures["torch.torch"] = {} # This API is not directly invokable and doesn't have direct arguments. It's the top-level torch namespace.
signatures["torch.torch_version"] = {} # No arguments
signatures["torch.trace"] = {
    "input": "tensor"
}
signatures["torch.transpose"] = {
    "input": "tensor",
    "dim0": "integer",
    "dim1": "integer"
}
signatures["torch.transpose_copy"] = {
    "input": "tensor",
    "dim0": "integer",
    "dim1": "integer"
}
signatures["torch.trapezoid"] = {
    "y": "tensor",
    "x": "tensor", # Could be None, but assuming tensor is more common when x is present
    "dx": "float", # Could be None
    "dim": "integer"
}
signatures["torch.tril"] = {
    "input": "tensor",
    "diagonal": "integer"
}
signatures["torch.triplet_margin_loss"] = {
    "anchor": "tensor",
    "positive": "tensor",
    "negative": "tensor",
    "margin": "float",
    "p": "float",
    "eps": "float",
    "swap": "boolean",
    "reduction": "string"
}
signatures["torch.true_divide"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.trunc_"] = {
    "input": "tensor"
}
signatures["torch.typename"] = {
    "input": "tensor"
}
signatures["torch.unbind"] = {
    "input": "tensor",
    "dim": "integer"
}
signatures["torch.unbind_copy"] = {
    "input": "tensor",
    "dim": "integer"
}
signatures["torch.unflatten"] = {
    "input": "tensor",
    "dim": "integer",
    "sizes": "tuple"
}
signatures["torch.unfold_copy"] = {
    "input": "tensor",
    "kernel_size": "tuple", # Could also be integer, but usually tuple
    "dilation": "tuple", # Could also be integer, but usually tuple
    "padding": "tuple", # Could also be integer, but usually tuple
    "stride": "tuple" # Could also be integer, but usually tuple
}
signatures["torch.unify_type_list"] = {
    "tensor_list": "tensor_list" # Could be list, but tensor_list seems more specific based on the context.
}
signatures["torch.unravel_index"] = {
    "indices": "tensor",
    "shape": "tuple"
}
signatures["torch.unsafe_chunk"] = {
    "input": "tensor",
    "chunks": "integer",
    "dim": "integer"
}
signatures["torch.unsafe_split"] = {
    "tensor": "tensor",
    "split_size_or_sections": "list", # Could also be integer or tuple, but list seems most general
    "dim": "integer"
}
signatures["torch.unsafe_split_with_sizes"] = {
    "tensor": "tensor",
    "split_sizes": "list", # Could also be tuple, but list is more common
    "dim": "integer"
}
signatures["torch.unsqueeze"] = {
    "input": "tensor",
    "dim": "integer"
}
signatures["torch.unsqueeze_copy"] = {
    "input": "tensor",
    "dim": "integer"
}
signatures["torch.use_deterministic_algorithms"] = {
    "mode": "boolean"
}
signatures["torch.utils.data.DataLoader"] = {
    "dataset": "tensor", # or "tensor_list" if it supports multiple datasets
    "batch_size": "integer",
    "shuffle": "boolean",
    "sampler": "tensor", # or other custom sampler type
    "batch_sampler": "tensor", # or other custom batch sampler type
    "num_workers": "integer",
    "collate_fn": "tensor", # or custom function, unclear how to represent
    "pin_memory": "boolean",
    "drop_last": "boolean",
    "timeout": "float",
    "worker_init_fn": "tensor", # or custom function, unclear how to represent
    "prefetch_factor": "integer",
    "persistent_workers": "boolean"
}
signatures["torch.utils.data.Dataset"] = {} # This is an abstract class so adding any parameters does not make sense
signatures["torch.utils.data.IterableDataset"] = {} # This is an abstract class so adding any parameters does not make sense
signatures["torch.utils.data.TensorDataset"] = {
    "tensors": "tensor_list"
}
signatures["torch.utils.data.random_split"] = {
    "dataset": "tensor", # dataset object, but using "tensor" as best match
    "lengths": "list",  # A list of integers
    "generator": "tensor" # Generator object
}
signatures["torch.utils.tensorboard.SummaryWriter"] = {
    "log_dir": "string",
    "comment": "string",
    "purge_step": "integer",
    "max_queue": "integer",
    "flush_secs": "integer",
    "filename_suffix": "string"
}
signatures["torch.values_copy"] = {
    "input": "tensor",
    "src": "tensor"
}
signatures["torch.vander"] = {
    "x": "tensor",
    "N": "integer", # Could also be None, but integer is more common
    "increasing": "boolean"
}
signatures["torch.vdot"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.version"] = {}
signatures["torch.view_as_complex"] = {
    "input": "tensor"
}
signatures["torch.view_as_complex_copy"] = {
    "input": "tensor"
}
signatures["torch.view_as_real"] = {
    "input": "tensor"
}
signatures["torch.view_as_real_copy"] = {
    "input": "tensor"
}
signatures["torch.view_copy"] = {
    "input": "tensor",
    "size": "tuple" # can also be an integer, but tuple is more general for representing shape
}
signatures["torch.vitals_enabled"] = {
}
signatures["torch.vmap"] = {
    "func": "string", # Should be a function, but string is the closest available type
    "in_dims": "integer", # Could also be a nested structure (tuple/list/dict), but integer is the most basic/common
    "out_dims": "integer", # Could also be a tuple
    "randomness": "string",
    "chunk_size": "integer" # Could also be None, but integer is closer
}
signatures["torch.vsplit"] = {
    "input": "tensor",
    "indices_or_sections": "list" # Could also be integer or tuple, but list seems most general
}
signatures["torch.vstack"] = {
    "tensors": "tensor_list"
}
signatures["torch.wait"] = {
    "events": "list" # Could also be a single event, but list seems more common.
}
signatures["torch.while_loop"] = {
    "cond": "list", # callable
    "body": "list", # callable
    "inputs": "tuple",
    "max_iterations": "integer"
}
signatures["torch.windows"] = {
    "input": "tensor",
    "size": "integer", # Can be a tuple too, but integer is more common
    "step": "integer" # Can be a tuple too, but integer is more common
}
signatures["torch.xlogy"] = {
    "input": "tensor",
    "other": "tensor"
}
signatures["torch.xlogy_"] = {
    "input": "tensor",
    "other": "tensor" # other can also be a float, but tensor is the most common.
}
signatures["torch.xpu.is_available"] = {}
signatures["torch.xpu.device_count"] = {}
signatures["torch.xpu.current_device"] = {}
signatures["torch.xpu.set_device"] = {
    "device": "integer" # Could be a Device object as well
}
signatures["torch.xpu.get_device_name"] = {
    "device": "integer" # Could be a Device object as well
}
signatures["torch.xpu.empty_cache"] = {}
signatures["torch.xpu.memory_summary"] = {
    "device": "integer", # Could be a Device object as well
    "abbreviated": "boolean"
}
signatures["torch.xpu.synchronize"] = {
    "device": "integer" # Could be a Device object as well
}
signatures["torch.xpu.stream"] = {}
signatures["torch.xpu.Stream"] = {}
signatures["torch.xpu.stream.current_stream"] = {
    "device": "integer" # Could be a Device object as well
}
signatures["torch.xpu.stream.synchronize"] = {
    "stream": "stream"
}
signatures["torch.xpu.Event"] = {}
signatures["torch.xpu.event.record"] = {
    "stream": "stream"
}
signatures["torch.xpu.event.wait"] = {
    "stream": "stream"
}
signatures["torch.xpu.event.elapsed_time"] = {
    "end_event": "stream",
    "start_event": "stream"
}
signatures["torch.xpu.init"] = {}
signatures["torch.xpu.is_initialized"] = {}
signatures["torch.zero_"] = {
    "input": "tensor"
}
signatures["torch.zeros"] = {
    "size": "tuple", # Could also be integer, but tuple seems more representative of the general case
    "dtype": "dtype",
    "layout": "string", # Though it's specifically torch.layout, string is the closest available type
    "requires_grad": "boolean"
}
signatures["torch.zeros_like"] = {
    "input": "tensor",
    "dtype": "dtype",
    "layout": "string", # Should ideally be Layout, but no matching type available
    "requires_grad": "boolean",
    "memory_format": "string" # Should ideally be torch.memory_format, but no matching type available
}

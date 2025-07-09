signatures = {}
signatures["tf.image.extract_patches"] = {
    "args": {
        "images": "tensor",
        "sizes": "list",
        "strides": "list",
        "rates": "list"
    },
    "kwargs": {
        "padding": "string",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.signal.rfft"] = {
    "args": {
        "input_tensor": "tensor"
    },
    "kwargs": {
        "fft_length": "integer", # Tensor of type int32 and shape [1] is an integer
        "name": "string"
    },
    "inner": {}
}
signatures["tf.linalg.expm"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.acosh"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.linalg.LinearOperatorBlockLowerTriangular"] = {
    "args": {
        "operators": "list" # Nested list of LinearOperators
    },
    "kwargs": {
        "is_non_singular": "boolean",
        "is_self_adjoint": "boolean",
        "is_positive_definite": "boolean",
        "is_square": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.io.parse_single_sequence_example"] = {
    "args": {
        "serialized": "string"
    },
    "kwargs": {
        "context_features": "dict", # A mapping of feature keys to `FixedLenFeature` or `VarLenFeature` or `RaggedFeature` values.
        "sequence_features": "dict", # A mapping of feature keys to `FixedLenSequenceFeature` or `VarLenFeature` or `RaggedFeature` values.
        "example_name": "string", # A scalar (0-D Tensor) of strings (optional), the name of the serialized proto.
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.real"] = {
    "args": {
        "val": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.sets.difference"] = {
    "args": {
        "a": "tensor",
        "b": "tensor"
    },
    "kwargs": {
        "aminusb": "boolean",
        "validate_indices": "boolean"
    },
    "inner": {}
}
signatures["tf.expand_dims"] = {
    "args": {
        "input": "tensor",
        "axis": "integer"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.random_uniform_initializer"] = {
    "args": {},
    "kwargs": {
        "minval": "float",
        "maxval": "float",
        "seed": "integer"
    },
    "inner": {}
}
signatures["tf.nn.embedding_lookup"] = {
    "args": {
        "params": "tensor_list", # Could also be a tensor
        "ids": "tensor"
    },
    "kwargs": {
        "max_norm": "float", # Could be None, but float is the closest type
        "name": "string"
    },
    "inner": {}
}
signatures["tf.linalg.LinearOperatorPermutation"] = {
    "args": {
        "perm": "tensor"
    },
    "kwargs": {
        "dtype": "dtype",
        "is_non_singular": "boolean",
        "is_self_adjoint": "boolean",
        "is_positive_definite": "boolean",
        "is_square": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.cumprod"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "exclusive": "boolean",
        "reverse": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.linalg.slogdet"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.edit_distance"] = {
    "args": {
        "hypothesis": "tensor",
        "truth": "tensor"
    },
    "kwargs": {
        "normalize": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.dtypes.as_dtype"] = {
    "args": {
        "type_value": "dtype" # Could also be string or integer, but dtype seems most appropriate
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.io.parse_single_example"] = {
    "args": {
        "serialized": "string", # A scalar string Tensor, a single serialized Example.
        "features": "dict" # A mapping of feature keys to `FixedLenFeature` or `VarLenFeature` values.
    },
    "kwargs": {
        "example_names": "string", # A scalar string Tensor, the associated name.
        "name": "string"
    },
    "inner": {}
}
signatures["tf.broadcast_to"] = {
    "args": {
        "input": "tensor",
        "shape": "tensor" # A Tensor of type int32 or int64 representing shape. Could also be list/tuple of ints?
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.linalg.logdet"] = {
    "args": {
        "matrix": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.strings.unicode_encode"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "output_encoding": "string",
        "errors": "string",
        "replacement_char": "integer", # Could also be a tensor representing the codepoint
        "name": "string"
    },
    "inner": {}
}
signatures["tf.nn.sufficient_statistics"] = {
    "args": {
        "x": "tensor",
        "axes": "list"
    },
    "kwargs": {
        "shift": "tensor",
        "keepdims": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.queue.RandomShuffleQueue"] = {
    "args": {
        "capacity": "integer",
        "min_after_dequeue": "integer",
        "dtypes": "list"
    },
    "kwargs": {
        "shapes": "list",  # Could also be tuple
        "names": "list",
        "seed": "integer",
        "shared_name": "string",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.feature_column.shared_embeddings"] = {
    "args": {
        "categorical_columns": "list",
        "dimension": "integer"
    },
    "kwargs": {
        "combiner": "string",
        "initializer": "tensor",  # Could be a callable, but representing it as a tensor for simplicity
        "shared_embedding_collection_name": "string",
        "ckpt_to_load_from": "string",
        "tensor_name_in_ckpt": "string",
        "max_norm": "float",
        "trainable": "boolean",
        "use_safe_embedding_lookup": "boolean"
    },
    "inner": {}
}
signatures["tf.math.xlog1py"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.nn.fractional_max_pool"] = {
    "args": {
        "value": "tensor",
        "pooling_ratio": "list" # could also be int, but list seems more general from the description
    },
    "kwargs": {
        "pseudo_random": "boolean",
        "overlapping": "boolean",
        "seed": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.special.fresnel_cos"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.arcsinh"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.histogram_fixed_width"] = {
    "args": {
        "values": "tensor",
        "value_range": "tensor"
    },
    "kwargs": {
        "nbins": "integer",
        "dtype": "dtype",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.fliplr"] = {
    "args": {
        "m": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.sysconfig.get_compile_flags"] = {
    "args": {},
    "kwargs": {},
    "inner": {}
}
signatures["tf.parallel_stack"] = {
    "args": {
        "values": "tensor_list"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.ragged.constant"] = {
    "args": {
        "pylist": "list" # Could also be tuple or np.ndarray, but list is the most general
    },
    "kwargs": {
        "dtype": "dtype",
        "ragged_rank": "integer",
        "inner_shape": "tuple",
        "name": "string",
        "row_splits_dtype": "dtype"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.empty_like"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["tf.math.bessel_i1e"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.random.get_global_generator"] = {
    "args": {},
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.atleast_3d"] = {
    "args": {
        "arys": "tensor_list" # Should be a list of tensors
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.strings.upper"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "encoding": "string",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.round"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "decimals": "integer"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.less"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.image.random_hue"] = {
    "args": {
        "image": "tensor",
        "max_delta": "float"
    },
    "kwargs": {
        "seed": "integer" # Could also be a tensor, but integer is closest.
    },
    "inner": {}
}
signatures["tf.nn.conv1d_transpose"] = {
    "args": {
        "input": "tensor",
        "filters": "tensor",
        "output_shape": "tensor" # Should probably be a list or tuple of integers
    },
    "kwargs": {
        "strides": "integer",
        "padding": "string",
        "data_format": "string",
        "dilations": "integer",
        "name": "string"
    },
    "inner": {}
}

signatures["tf.nn.conv1d_transpose_2"] = {
    "args": {
        "input": "tensor",
        "filters": "tensor",
        "output_shape": "tensor" # Should probably be a list or tuple of integers
    },
    "kwargs": {
        "strides": "list",
        "padding": "string",
        "data_format": "string",
        "dilations": "list",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.data.experimental.group_by_window"] = {
    "args": {
        "key_func": "list",  # Assuming this is a function, representing it as a list (no function type available)
        "reduce_func": "list"   # Assuming this is a function, representing it as a list (no function type available)
    },
    "kwargs": {
        "window_size": "tensor",
        "window_size_func": "list"  # Assuming this is a function, representing it as a list (no function type available)
    },
    "inner": {}
}
signatures["tf.experimental.numpy.greater"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.math.special.bessel_k1e"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.data.experimental.prefetch_to_device"] = {
    "args": {
        "device": "string"
    },
    "kwargs": {
        "buffer_size": "integer" # Could be tf.compat.v1.data.AUTOTUNE which would be an integer
    },
    "inner": {}
}
signatures["tf.nn.avg_pool3d"] = {
    "args": {
        "input": "tensor",
        "ksize": "list", # can also be an int, but defaulting to list
        "strides": "list", # can also be an int, but defaulting to list
        "padding": "string"
    },
    "kwargs": {
        "data_format": "string",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.sets.intersection"] = {
    "args": {
        "a": "tensor",
        "b": "tensor"
    },
    "kwargs": {
        "validate_indices": "boolean"
    },
    "inner": {}
}
signatures["tf.strings.unicode_decode"] = {
    "args": {
        "input": "tensor",
        "input_encoding": "string"
    },
    "kwargs": {
        "errors": "string",
        "replacement_char": "integer",
        "replace_control_characters": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.argsort"] = {
    "args": {
        "values": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "direction": "string",
        "stable": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.is_non_decreasing"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.truncatemod"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.clip"] = {
    "args": {
        "a": "tensor",
        "a_min": "tensor",
        "a_max": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.data.experimental.make_saveable_from_iterator"] = {
    "args": {
        "iterator": "tensor" # Iterator is essentially a tensor
    },
    "kwargs": {
        "external_state_policy": "string"
    },
    "inner": {}
}
signatures["tf.sparse.add"] = {
    "args": {
        "a": "tensor",
        "b": "tensor"
    },
    "kwargs": {
        "threshold": "tensor" # Although the doc says 0-D Tensor, "tensor" seems like the closest match.
    },
    "inner": {}
}
signatures["tf.repeat"] = {
    "args": {
        "input": "tensor",
        "repeats": "tensor" # It can be an integer but also an 1-D tensor of integers so it falls under tensor type
    },
    "kwargs": {
        "axis": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.one_hot"] = {
    "args": {
        "indices": "tensor",
        "depth": "integer"
    },
    "kwargs": {
        "on_value": "tensor", # Should probably be float or integer but picking tensor for now
        "off_value": "tensor", # Should probably be float or integer but picking tensor for now
        "axis": "integer",
        "dtype": "dtype",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.floor_divide"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.image.sobel_edges"] = {
    "args": {
        "image": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.math.atan2"] = {
    "args": {
        "y": "tensor",
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.io.serialize_tensor"] = {
    "args": {
        "tensor": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.sparse.sparse_dense_matmul"] = {
    "args": {
        "sp_a": "tensor",
        "b": "tensor"
    },
    "kwargs": {
        "adjoint_a": "boolean",
        "adjoint_b": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.image.resize_with_pad"] = {
    "args": {
        "image": "tensor",
        "target_height": "integer",
        "target_width": "integer"
    },
    "kwargs": {
        "method": "string",
        "antialias": "boolean"
    },
    "inner": {}
}
signatures["tf.feature_column.embedding_column"] = {
    "args": {
        "categorical_column": "list", # A `CategoricalColumn` created by a `categorical_column_with_*` function. Using "list" since CategoricalColumn isn't a primitive type.
        "dimension": "integer"
    },
    "kwargs": {
        "combiner": "string",
        "initializer": "tensor", # A variable initializer function, so assuming it's a tensor that represents the initialized variable.
        "ckpt_to_load_from": "string",
        "tensor_name_in_ckpt": "string",
        "max_norm": "float",
        "trainable": "boolean",
        "use_safe_embedding_lookup": "boolean"
    },
    "inner": {}
}
signatures["tf.lookup.TextFileInitializer"] = {
    "args": {
        "filename": "string",
        "key_dtype": "dtype",
        "key_index": "integer", # Could also be TextFileIndex.WHOLE_LINE or TextFileIndex.LINE_NUMBER, but these are internally represented as integers.
        "value_dtype": "dtype",
        "value_index": "integer" # Same as key_index.
    },
    "kwargs": {
        "vocab_size": "integer",
        "delimiter": "string",
        "name": "string",
        "value_index_offset": "integer"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.isinf"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.feature_column.categorical_column_with_hash_bucket"] = {
    "args": {
        "key": "string",
        "hash_bucket_size": "integer"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["tf.einsum"] = {
    "args": {
        "equation": "string",
        "inputs": "tensor_list"
    },
    "kwargs": {
        "optimize": "string",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.nn.safe_embedding_lookup_sparse"] = {
    "args": {
        "embedding_weights": "tensor_list", # Can be a single tensor or a list of tensors
        "sparse_ids": "tensor" # Could also be a RaggedTensor but representing as tensor
    },
    "kwargs": {
        "sparse_weights": "tensor", # Could also be a RaggedTensor or None, representing as tensor
        "combiner": "string",
        "default_id": "integer", # Assuming default_id can be represented as integer
        "max_norm": "float", # Can be None, representing as float
        "name": "string",
        "allow_fast_lookup": "boolean"
    },
    "inner": {}
}
signatures["tf.nn.local_response_normalization"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "depth_radius": "integer",
        "bias": "float",
        "alpha": "float",
        "beta": "float",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.signal.kaiser_window"] = {
    "args": {
        "window_length": "tensor"
    },
    "kwargs": {
        "beta": "float",
        "dtype": "dtype",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.lbeta"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.random.stateless_normal"] = {
    "args": {
        "shape": "tensor", # Could also be list or tuple of integers, but tensor is more general
        "seed": "tensor"
    },
    "kwargs": {
        "mean": "float",
        "stddev": "float",
        "dtype": "dtype",
        "name": "string",
        "alg": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.argmax"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer" # Could also be None, but integer is the closest type.
    },
    "inner": {}
}
signatures["tf.linalg.LinearOperatorTridiag"] = {
    "args": {
        "diagonals": "tensor"
    },
    "kwargs": {
        "diagonals_format": "string",
        "is_non_singular": "boolean",
        "is_self_adjoint": "boolean",
        "is_positive_definite": "boolean",
        "is_square": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.queue.PaddingFIFOQueue"] = {
    "args": {
        "capacity": "integer",
        "dtypes": "list",  # List of dtypes
        "shapes": "list"   # List of shapes; could also be tuple, but list seems more appropriate here.
    },
    "kwargs": {
        "names": "list",    # List of strings
        "shared_name": "string",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.nn.leaky_relu"] = {
    "args": {
        "features": "tensor"
    },
    "kwargs": {
        "alpha": "float",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.signal.mfccs_from_log_mel_spectrograms"] = {
    "args": {
        "log_mel_spectrograms": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.sysconfig.get_include"] = {
    "args": {},
    "kwargs": {},
    "inner": {}
}
signatures["tf.nn.pool"] = {
    "args": {
        "input": "tensor",
        "window_shape": "list", # Could also be tuple
        "pooling_type": "string"
    },
    "kwargs": {
        "strides": "list", # Could also be tuple or None
        "padding": "string",
        "data_format": "string",
        "dilations": "list", # Could also be tuple or None
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.abs"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.io.gfile.copy"] = {
    "args": {
        "src": "string",
        "dst": "string"
    },
    "kwargs": {
        "overwrite": "boolean"
    },
    "inner": {}
}
signatures["tf.slice"] = {
    "args": {
        "input_": "tensor",
        "begin": "tensor", # int32 or int64 Tensor
        "size": "tensor"  # int32 or int64 Tensor
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.compat.dimension_value"] = {
    "args": {
        "dimension": "integer" # Could also be None, but integer is the closest type available
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.math.logical_or"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.image.adjust_brightness"] = {
    "args": {
        "image": "tensor",
        "delta": "float"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.extract_volume_patches"] = {
    "args": {
        "input": "tensor",
        "ksizes": "list",
        "strides": "list",
        "padding": "string"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.absolute"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.math.special.bessel_y0"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.promote_types"] = {
    "args": {
        "type1": "dtype",
        "type2": "dtype"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.io.gfile.listdir"] = {
    "args": {
        "path": "string"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.nn.normalize_moments"] = {
    "args": {
        "counts": "tensor",
        "mean_ss": "tensor",
        "variance_ss": "tensor",
        "shift": "tensor" # It can also be None, but "tensor" seems like the closest match
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.sparse.reset_shape"] = {
    "args": {
        "sp_input": "tensor"
    },
    "kwargs": {
        "new_shape": "tensor" # Could also be a list or tuple of integers, but representing it as a tensor for simplicity
    },
    "inner": {}
}
signatures["tf.io.decode_csv"] = {
    "args": {
        "records": "tensor",
        "record_defaults": "tensor_list"
    },
    "kwargs": {
        "field_delim": "string",
        "use_quote_delim": "boolean",
        "na_value": "string",
        "select_cols": "list", # or tuple of integers
        "name": "string"
    },
    "inner": {}
}
signatures["tf.strings.reduce_join"] = {
    "args": {
        "inputs": "tensor"
    },
    "kwargs": {
        "axis": "integer", # Could also be a tensor?
        "keepdims": "boolean",
        "separator": "string",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.unravel_index"] = {
    "args": {
        "indices": "tensor",
        "dims": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.random.experimental.stateless_fold_in"] = {
    "args": {
        "seed": "tensor",
        "data": "integer"
    },
    "kwargs": {
        "alg": "string"
    },
    "inner": {}
}
signatures["tf.math.conj"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.ones"] = {
    "args": {
        "shape": "list" # Could also be a tuple or tensor of type int32
    },
    "kwargs": {
        "dtype": "dtype",
        "name": "string",
        "layout": "tensor" # Not explicitly defined, but assuming it's a tensor
    },
    "inner": {}
}
signatures["tf.experimental.numpy.true_divide"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.ragged.stack"] = {
    "args": {
        "values": "list"  # List of tensors or ragged tensors
    },
    "kwargs": {
        "axis": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.gradients"] = {
    "args": {
        "ys": "tensor_list",
        "xs": "tensor_list"
    },
    "kwargs": {
        "grad_ys": "tensor_list", # Could also be None, but treating as tensor_list
        "name": "string",
        "gate_gradients": "boolean",
        "aggregation_method": "string", # Should ideally be of type AggregationMethod but it's a string.
        "stop_gradients": "tensor_list",
        "unconnected_gradients": "string" # Should ideally be of type UnconnectedGradients, but it's a string.
    },
    "inner": {}
}
signatures["tf.nn.with_space_to_batch"] = {
    "args": {
        "input": "tensor",
        "dilation_rate": "tensor", # int32 Tensor of *known* shape [num_spatial_dims] is a tensor
        "padding": "string",
        "op": "list" #Function that maps (input, num_spatial_dims, padding) -> output. Closest type is list
    },
    "kwargs": {
        "filter_shape": "tensor", #integer Tensor of shape [>=num_spatial_dims] is a tensor
        "spatial_dims": "list",
        "data_format": "string"
    },
    "inner": {}
}
signatures["tf.data.experimental.unbatch"] = {
    "args": {},
    "kwargs": {},
    "inner": {}
}
signatures["tf.math.top_k"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "k": "integer",
        "sorted": "boolean",
        "index_type": "dtype",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.tensor_scatter_nd_add"] = {
    "args": {
        "tensor": "tensor",
        "indices": "tensor",
        "updates": "tensor"
    },
    "kwargs": {
        "bad_indices_policy": "string",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.iinfo"] = {
    "args": {
        "int_type": "dtype" # Could also be integer or instance, but dtype seems closest
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.linalg.LinearOperatorCirculant2D"] = {
    "args": {
        "spectrum": "tensor"
    },
    "kwargs": {
        "input_output_dtype": "dtype",
        "is_non_singular": "boolean",
        "is_self_adjoint": "boolean",
        "is_positive_definite": "boolean",
        "is_square": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.signal.fft2d"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.gather_nd"] = {
    "args": {
        "params": "tensor",
        "indices": "tensor"
    },
    "kwargs": {
        "batch_dims": "integer",
        "name": "string",
        "bad_indices_policy": "string"
    },
    "inner": {}
}
signatures["tf.random.gamma"] = {
    "args": {
        "shape": "tensor", # Could also be a list of integers
        "alpha": "tensor"
    },
    "kwargs": {
        "beta": "tensor", # Could be float if a python value is passed. Default is 1.
        "dtype": "dtype",
        "seed": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.io.decode_gif"] = {
    "args": {
        "contents": "string"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.split_1"] = {
    "args": {
        "value": "tensor",
        "num_or_size_splits": "integer"
    },
    "kwargs": {
        "axis": "integer",
        "num": "integer",
        "name": "string"
    },
    "inner": {}
}

signatures["tf.split_2"] = {
    "args": {
        "value": "tensor",
        "num_or_size_splits": "list" # Could also be "tensor" since it accepts a 1-D Tensor. But example shows list, so using list.
    },
    "kwargs": {
        "axis": "integer",
        "num": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.image.flip_up_down"] = {
    "args": {
        "image": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.math.reduce_sum"] = {
    "args": {
        "input_tensor": "tensor"
    },
    "kwargs": {
        "axis": "integer", # Could also be a list or tuple of integers
        "keepdims": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.nn.dilation2d"] = {
    "args": {
        "input": "tensor",
        "filters": "tensor",
        "strides": "list",
        "padding": "string",
        "data_format": "string",
        "dilations": "list"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.data.experimental.ThreadingOptions"] = {
    "args": {},
    "kwargs": {},
    "inner": {}
}
signatures["tf.linalg.LinearOperatorKronecker"] = {
    "args": {
        "operators": "list" # Could be a list of LinearOperator objects, treating as list for now.
    },
    "kwargs": {
        "is_non_singular": "boolean",
        "is_self_adjoint": "boolean",
        "is_positive_definite": "boolean",
        "is_square": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.linalg.sqrtm"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.nn.relu6"] = {
    "args": {
        "features": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.dsplit"] = {
    "args": {
        "ary": "tensor",
        "indices_or_sections": "integer" # Can also be a tuple or list of integers
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.math.softplus"] = {
    "args": {
        "features": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.logspace"] = {
    "args": {
        "start": "tensor", # Can be float or integer
        "stop": "tensor", # Can be float or integer
    },
    "kwargs": {
        "num": "integer",
        "endpoint": "boolean",
        "base": "float",
        "dtype": "dtype",
        "axis": "integer"
    },
    "inner": {}
}
signatures["tf.nn.collapse_repeated"] = {
    "args": {
        "labels": "tensor",
        "seq_length": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.atanh"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.nn.log_poisson_loss"] = {
    "args": {
        "targets": "tensor",
        "log_input": "tensor"
    },
    "kwargs": {
        "compute_full_loss": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.image.non_max_suppression"] = {
    "args": {
        "boxes": "tensor",
        "scores": "tensor",
        "max_output_size": "integer"
    },
    "kwargs": {
        "iou_threshold": "float",
        "score_threshold": "float",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.minimum"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.nn.conv1d"] = {
    "args": {
        "input": "tensor",
        "filters": "tensor",
        "stride": "integer" # or list? Creating a second signature
    },
    "kwargs": {
        "padding": "string",
        "data_format": "string",
        "dilations": "list", # or integer? Creating a second signature
        "name": "string"
    },
    "inner": {}
}

signatures["tf.nn.conv1d_2"] = {
    "args": {
        "input": "tensor",
        "filters": "tensor",
        "stride": "list"
    },
    "kwargs": {
        "padding": "string",
        "data_format": "string",
        "dilations": "list",
        "name": "string"
    },
    "inner": {}
}

signatures["tf.nn.conv1d_3"] = {
    "args": {
        "input": "tensor",
        "filters": "tensor",
        "stride": "integer"
    },
    "kwargs": {
        "padding": "string",
        "data_format": "string",
        "dilations": "integer",
        "name": "string"
    },
    "inner": {}
}

signatures["tf.nn.conv1d_4"] = {
    "args": {
        "input": "tensor",
        "filters": "tensor",
        "stride": "list"
    },
    "kwargs": {
        "padding": "string",
        "data_format": "string",
        "dilations": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.nn.sampled_softmax_loss"] = {
    "args": {
        "weights": "tensor", # Could be tensor_list as well, but choosing tensor as the base type
        "biases": "tensor",
        "labels": "tensor",
        "inputs": "tensor",
        "num_sampled": "integer",
        "num_classes": "integer"
    },
    "kwargs": {
        "num_true": "integer",
        "sampled_values": "tuple",
        "remove_accidental_hits": "boolean",
        "seed": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.tile"] = {
    "args": {
        "a": "tensor",
        "reps": "tuple" # could be list as well, but numpy.tile takes tuple or integer
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.linalg.eig"] = {
    "args": {
        "tensor": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.scalar_mul"] = {
    "args": {
        "scalar": "tensor",
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.data.experimental.service.distribute"] = {
    "args": {
        "processing_mode": "string", # Could be ShardingPolicy enum, but string is closest
        "service": "string" # Could also be a tuple of strings
    },
    "kwargs": {
        "job_name": "string",
        "consumer_index": "integer",
        "num_consumers": "integer",
        "max_outstanding_requests": "integer",
        "data_transfer_protocol": "string",
        "compression": "string",
        "cross_trainer_cache": "tuple", # Not sure about type, documentation uses CrossTrainerCache object
        "target_workers": "string"
    },
    "inner": {}
}
signatures["tf.make_tensor_proto"] = {
    "args": {
        "values": "tensor"  # Could also be list or scalar, but representing it as a tensor for simplicity
    },
    "kwargs": {
        "dtype": "dtype",
        "shape": "list", # List of integers
        "verify_shape": "boolean",
        "allow_broadcast": "boolean"
    },
    "inner": {}
}
signatures["tf.math.logical_not"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.nn.relu"] = {
    "args": {
        "features": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.prod"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "dtype": "dtype",
        "keepdims": "boolean"
    },
    "inner": {}
}
signatures["tf.io.TFRecordWriter"] = {
    "args": {
        "path": "string"
    },
    "kwargs": {
        "options": "string" # Could also be something else, unclear from the docstring
    },
    "inner": {}
}
signatures["tf.random.categorical"] = {
    "args": {
        "logits": "tensor",
        "num_samples": "integer"
    },
    "kwargs": {
        "dtype": "dtype",
        "seed": "integer", # A Python integer is used to create a random seed
        "name": "string"
    },
    "inner": {}
}
signatures["tf.random.experimental.stateless_split"] = {
    "args": {
        "seed": "tensor"
    },
    "kwargs": {
        "num": "integer",
        "alg": "string"
    },
    "inner": {}
}
signatures["tf.linalg.pinv"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "rcond": "tensor", # Could also be float, but tensor is more general
        "validate_args": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.signal.frame"] = {
    "args": {
        "signal": "tensor",
        "frame_length": "integer",
        "frame_step": "integer"
    },
    "kwargs": {
        "pad_end": "boolean",
        "pad_value": "tensor", # Should this be a more specific type? It is scalar, but can be int/float
        "axis": "tensor", # Should this be integer? The documentation says it's a scalar integer Tensor.
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.reduce_min"] = {
    "args": {
        "input_tensor": "tensor"
    },
    "kwargs": {
        "axis": "integer", # Could be a list or tuple of integers as well, but defaulting to integer.
        "keepdims": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.angle"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.Module"] = {
    "args": {},
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.io.gfile.rmtree"] = {
    "args": {
        "path": "string"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.math.multiply"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.io.match_filenames_once"] = {
    "args": {
        "pattern": "tensor" # A file pattern (glob), or 1D tensor of file patterns.
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.io.matching_files"] = {
    "args": {
        "pattern": "string"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.nn.crelu"] = {
    "args": {
        "features": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.less_equal"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.load_library"] = {
    "args": {
        "library_location": "string"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.nextafter"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.get_static_value"] = {
    "args": {
        "tensor": "tensor"
    },
    "kwargs": {
        "partial": "boolean"
    },
    "inner": {}
}
signatures["tf.image.rgb_to_yuv"] = {
    "args": {
        "images": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.sparse.retain"] = {
    "args": {
        "sp_input": "tensor", # SparseTensor is a type of tensor
        "to_retain": "boolean" # A bool vector is a boolean
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.saved_model.save"] = {
    "args": {
        "obj": "Trackable", # Best type approximation as it should be a tf.Module or tf.train.Checkpoint
        "export_dir": "string"
    },
    "kwargs": {
        "signatures": "signatures", # Could also be a tf.function or dictionary mapping strings to functions
        "options": "SaveOptions" # Best type approximation for tf.saved_model.SaveOptions
    },
    "inner": {}
}
signatures["tf.math.reduce_variance"] = {
    "args": {
        "input_tensor": "tensor"
    },
    "kwargs": {
        "axis": "integer", # Could also be a list/tuple of integers, but integer is the base case
        "keepdims": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.special.bessel_k1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.reduce_max"] = {
    "args": {
        "input_tensor": "tensor"
    },
    "kwargs": {
        "axis": "list", # Could also be integer or None, but list is the most general
        "keepdims": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.flipud"] = {
    "args": {
        "m": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.linalg.tridiagonal_solve"] = {
    "args": {
        "diagonals": "tensor",
        "rhs": "tensor"
    },
    "kwargs": {
        "diagonals_format": "string",
        "transpose_rhs": "boolean",
        "conjugate_rhs": "boolean",
        "name": "string",
        "partial_pivoting": "boolean",
        "perturb_singular": "boolean"
    },
    "inner": {}
}
signatures["tf.make_ndarray"] = {
    "args": {
        "tensor": "tensor" # A TensorProto is effectively a tensor
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.dtypes.complex"] = {
    "args": {
        "real": "tensor",
        "imag": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.var"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer", # Could also be a tuple or list of integers
        "dtype": "dtype",
        "out": "tensor",
        "ddof": "integer",
        "keepdims": "boolean"
    },
    "inner": {}
}
signatures["tf.sysconfig.get_link_flags"] = {
    "args": {},
    "kwargs": {},
    "inner": {}
}
signatures["tf.math.pow"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.expm1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.data.experimental.Counter"] = {
    "args": {},
    "kwargs": {
        "start": "integer",
        "step": "integer",
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.logaddexp2"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.feature_column.crossed_column"] = {
    "args": {
        "keys": "list",
        "hash_bucket_size": "integer"
    },
    "kwargs": {
        "hash_key": "string" # Could also be integer
    },
    "inner": {}
}
signatures["tf.rank"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.signal.linear_to_mel_weight_matrix"] = {
    "args": {
        "num_mel_bins": "integer",
        "num_spectrogram_bins": "integer", # An integer Tensor, but integer is the closest type
        "sample_rate": "float" # An integer or float Tensor, but float is the closest type
    },
    "kwargs": {
        "lower_edge_hertz": "float",
        "upper_edge_hertz": "float",
        "dtype": "dtype",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.polyval"] = {
    "args": {
        "p": "tensor",
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.random.poisson"] = {
    "args": {
        "shape": "tensor", # "list" can also work
        "lam": "tensor"
    },
    "kwargs": {
        "dtype": "dtype",
        "seed": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.unique"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "out_idx": "dtype",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.repeat"] = {
    "args": {
        "a": "tensor",
        "repeats": "integer" # can also be a tensor but sticking with integer for simplicity
    },
    "kwargs": {
        "axis": "integer" # can also be None
    },
    "inner": {}
}
signatures["tf.linalg.logm"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.quantization.fake_quant_with_min_max_vars_per_channel"] = {
    "args": {
        "inputs": "tensor",
        "min": "tensor",
        "max": "tensor"
    },
    "kwargs": {
        "num_bits": "integer",
        "narrow_range": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.strings.ngrams"] = {
    "args": {
        "data": "tensor"
    },
    "kwargs": {
        "ngram_width": "integer", # Can also be list/tuple of integers, but creating a separate signature for that is not needed based on instructions
        "separator": "string",
        "pad_values": "string", # Can also be tuple of strings or None, assuming string for simplicity
        "padding_width": "integer",
        "preserve_short_sequences": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.broadcast_to"] = {
    "args": {
        "array": "tensor",
        "shape": "tuple" # Could also be a list, but tuple is more common for shapes
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.tensor_scatter_nd_min"] = {
    "args": {
        "tensor": "tensor",
        "indices": "tensor",
        "updates": "tensor"
    },
    "kwargs": {
        "bad_indices_policy": "string",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.triu"] = {
    "args": {
        "m": "tensor"
    },
    "kwargs": {
        "k": "integer"
    },
    "inner": {}
}
signatures["tf.reverse"] = {
    "args": {
        "tensor": "tensor",
        "axis": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.take_along_axis"] = {
    "args": {
        "arr": "tensor",
        "indices": "tensor",
        "axis": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.maximum"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.fill"] = {
    "args": {
        "dims": "list", # Could also be tuple, but list is more general
        "value": "tensor" # Value could be a variety of types, but "tensor" is most general
    },
    "kwargs": {
        "name": "string",
        "layout": "tensor"
    },
    "inner": {}
}
signatures["tf.math.less"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.arcsin"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.train.Coordinator"] = {
    "args": {},
    "kwargs": {
        "clean_stop_exception_types": "tuple" # Or list, depending on what it is
    },
    "inner": {}
}
signatures["tf.image.stateless_random_jpeg_quality"] = {
    "args": {
        "image": "tensor",
        "min_jpeg_quality": "integer",
        "max_jpeg_quality": "integer",
        "seed": "tensor" # A shape [2] Tensor, the seed to the random number generator.
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.io.TFRecordOptions"] = {
    "args": {},
    "kwargs": {
        "compression_type": "string",
        "flush_mode": "string",
        "input_buffer_size": "integer",
        "output_buffer_size": "integer",
        "window_bits": "integer",
        "compression_level": "integer",
        "compression_method": "integer",
        "mem_level": "integer",
        "compression_strategy": "integer"
    },
    "inner": {}
}
signatures["tf.io.gfile.rename"] = {
    "args": {
        "src": "string",
        "dst": "string"
    },
    "kwargs": {
        "overwrite": "boolean"
    },
    "inner": {}
}
signatures["tf.signal.ifft"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.log1p"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.scatter_nd"] = {
    "args": {
        "indices": "tensor",
        "updates": "tensor",
        "shape": "tensor"
    },
    "kwargs": {
        "bad_indices_policy": "string",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.image.ssim"] = {
    "args": {
        "img1": "tensor",
        "img2": "tensor",
        "max_val": "float" # Should ideally be a tensor, but picking float as the closest match
    },
    "kwargs": {
        "filter_size": "integer",
        "filter_sigma": "float",
        "k1": "float",
        "k2": "float",
        "return_index_map": "boolean"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.ceil"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.no_op"] = {
    "args": {},
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.nest.flatten"] = {
    "args": {
        "structure": "list" # Could also be tuple or other nested structure
    },
    "kwargs": {
        "expand_composites": "boolean"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.split"] = {
    "args": {
        "ary": "tensor",
        "indices_or_sections": "integer" # Can also be a list of integers
    },
    "kwargs": {
        "axis": "integer"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.transpose"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axes": "tuple" # could be also a list of integers
    },
    "inner": {}
}
signatures["tf.saved_model.load"] = {
    "args": {
        "export_dir": "string"
    },
    "kwargs": {
        "tags": "list", # Could also be a single string, but defaulting to list as it can be a sequence of tags
        "options": "string" # There is a LoadOptions object but no suitable type available
    },
    "inner": {}
}
signatures["tf.linalg.lu_solve"] = {
    "args": {
        "lower_upper": "tensor",
        "perm": "tensor",
        "rhs": "tensor"
    },
    "kwargs": {
        "validate_args": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.random.seed"] = {
    "args": {
        "s": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.math.igammac"] = {
    "args": {
        "a": "tensor",
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.cumsum"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["tf.strings.to_hash_bucket_fast"] = {
    "args": {
        "input": "tensor",
        "num_buckets": "integer"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.bitwise_not"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.compat.path_to_str"] = {
    "args": {
        "path": "string" # Could also be a pathlike object, but we only have string to work with
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.histogram_fixed_width_bins"] = {
    "args": {
        "values": "tensor",
        "value_range": "tensor"
    },
    "kwargs": {
        "nbins": "integer",
        "dtype": "dtype",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.load_op_library"] = {
    "args": {
        "library_filename": "string"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.math.not_equal"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.subtract"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.io.write_file"] = {
    "args": {
        "filename": "string",
        "contents": "string"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.strings.unicode_script"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.full"] = {
    "args": {
        "shape": "list", # Could also be tuple or integer, but list is the most general
        "fill_value": "tensor" # Can be any value that can be converted to a tensor
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.array"] = {
    "args": {
        "val": "tensor" # Could also be ndarray, but representing as tensor
    },
    "kwargs": {
        "dtype": "dtype",
        "copy": "boolean",
        "ndmin": "integer"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.isfinite"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.vectorized_map"] = {
    "args": {
        "fn": "list", # Callable
        "elems": "tensor"
    },
    "kwargs": {
        "fallback_to_while_loop": "boolean",
        "warn": "boolean"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.sign"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "out": "tensor",
        "where": "boolean"
    },
    "inner": {}
}
signatures["tf.math.expm1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.data.experimental.sample_from_datasets"] = {
    "args": {
        "datasets": "list"
    },
    "kwargs": {
        "weights": "list", # Could also be a tensor, but defaulting to list for simplicity
        "seed": "tensor", # A `tf.int64` scalar `tf.Tensor`
        "stop_on_empty_dataset": "boolean"
    },
    "inner": {}
}
signatures["tf.linalg.LinearOperator"] = {
    "args": {
        "dtype": "dtype"
    },
    "kwargs": {
        "graph_parents": "list", # could also be tuple of tensors
        "is_non_singular": "boolean",
        "is_self_adjoint": "boolean",
        "is_positive_definite": "boolean",
        "is_square": "boolean",
        "name": "string",
        "parameters": "dict"
    },
    "inner": {}
}
signatures["tf.image.random_jpeg_quality"] = {
    "args": {
        "image": "tensor",
        "min_jpeg_quality": "integer",
        "max_jpeg_quality": "integer"
    },
    "kwargs": {
        "seed": "integer" # Could also be a tensor of type int32 or int64. Choosing integer for simplicity.
    },
    "inner": {}
}
signatures["tf.math.log_sigmoid"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.nest.pack_sequence_as"] = {
    "args": {
        "structure": "list",  # Could also be tuple or dict, but list is the most general container type
        "flat_sequence": "list"
    },
    "kwargs": {
        "expand_composites": "boolean"
    },
    "inner": {}
}
signatures["tf.math.sobol_sample"] = {
    "args": {
        "dim": "tensor",
        "num_results": "integer"
    },
    "kwargs": {
        "skip": "integer",
        "dtype": "dtype",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.nn.max_pool"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "ksize": "list", # Could also be integer, but list is more general
        "strides": "list", # Could also be integer, but list is more general
        "padding": "string",
        "data_format": "string",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.linalg.LinearOperatorHouseholder"] = {
    "args": {
        "reflection_axis": "tensor"
    },
    "kwargs": {
        "is_non_singular": "boolean",
        "is_self_adjoint": "boolean",
        "is_positive_definite": "boolean",
        "is_square": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.reduce_logsumexp"] = {
    "args": {
        "input_tensor": "tensor"
    },
    "kwargs": {
        "axis": "list", # Could be integer too, but defaulting to list as that is more general
        "keepdims": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.diag"] = {
    "args": {
        "v": "tensor"
    },
    "kwargs": {
        "k": "integer"
    },
    "inner": {}
}
signatures["tf.nn.compute_average_loss"] = {
    "args": {
        "per_example_loss": "tensor"
    },
    "kwargs": {
        "sample_weight": "tensor", # Could also be None but choosing tensor as the most appropriate type.
        "global_batch_size": "integer" # Integer or None
    },
    "inner": {}
}
signatures["tf.sparse.reduce_sum"] = {
    "args": {
        "sp_input": "tensor"
    },
    "kwargs": {
        "axis": "list", # Could also be integer, but list is more general
        "keepdims": "boolean",
        "output_is_sparse": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.sequence_mask"] = {
    "args": {
        "lengths": "tensor"  # Should ideally be "integer tensor" but closest is "tensor"
    },
    "kwargs": {
        "maxlen": "integer", #scalar integer tensor is closest to integer
        "dtype": "dtype",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.Graph"] = {
    "args": {},
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.atleast_1d"] = {
    "args": {
        "arys": "tensor_list"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.nest.assert_same_structure"] = {
    "args": {
        "nest1": "list",  # Could also be tuple or tensor, but list seems most general
        "nest2": "list"   # Could also be tuple or tensor, but list seems most general
    },
    "kwargs": {
        "check_types": "boolean",
        "expand_composites": "boolean"
    },
    "inner": {}
}
signatures["tf.math.reduce_prod"] = {
    "args": {
        "input_tensor": "tensor"
    },
    "kwargs": {
        "axis": "integer", # Could also be a list of integers, but integer is the simpler case
        "keepdims": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.ragged.segment_ids_to_row_splits"] = {
    "args": {
        "segment_ids": "tensor"
    },
    "kwargs": {
        "num_segments": "integer",
        "out_type": "dtype",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.floor"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.nest.is_nested"] = {
    "args": {
        "seq": "list" # Could also be tuple or a more general structure
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.lookup.KeyValueTensorInitializer"] = {
    "args": {
        "keys": "tensor",
        "values": "tensor"
    },
    "kwargs": {
        "key_dtype": "dtype",
        "value_dtype": "dtype",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.group"] = {
    "args": {
        "inputs": "tensor_list" # Should be a list of tensors or operations, choosing tensor_list to be more specific.
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.exp2"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.audio.encode_wav"] = {
    "args": {
        "audio": "tensor",
        "sample_rate": "integer" # Tensor of type int32 is an integer
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.segment_min"] = {
    "args": {
        "data": "tensor",
        "segment_ids": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.saved_model.Asset"] = {
    "args": {
        "path": "string" # Could also be a 0-D tf.string tensor, but string seems closer
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.sparse.eye"] = {
    "args": {
        "num_rows": "integer"
    },
    "kwargs": {
        "num_columns": "integer",
        "dtype": "dtype",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.signal.kaiser_bessel_derived_window"] = {
    "args": {
        "window_length": "tensor" # Could also be integer, but tensor is more general
    },
    "kwargs": {
        "beta": "float",
        "dtype": "dtype",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.nn.atrous_conv2d"] = {
    "args": {
        "value": "tensor",
        "filters": "tensor",
        "rate": "integer",
        "padding": "string"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.get_logger"] = {
    "args": {},
    "kwargs": {},
    "inner": {}
}
signatures["tf.queue.FIFOQueue"] = {
    "args": {
        "capacity": "integer",
        "dtypes": "list" # list of dtypes
    },
    "kwargs": {
        "shapes": "list", # could also be tuple but list is more general
        "names": "list", # list of strings
        "shared_name": "string",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.image.yuv_to_rgb"] = {
    "args": {
        "images": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.strings.substr"] = {
    "args": {
        "input": "tensor",
        "pos": "tensor",
        "len": "tensor"
    },
    "kwargs": {
        "unit": "string",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.lookup.StaticHashTable"] = {
    "args": {
        "initializer": "tensor", # Initializer can be a KeyValueTensorInitializer
        "default_value": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.add"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.bincount"] = {
    "args": {
        "arr": "tensor"
    },
    "kwargs": {
        "weights": "tensor",
        "minlength": "integer",
        "maxlength": "integer",
        "dtype": "dtype",
        "name": "string",
        "axis": "integer",
        "binary_output": "boolean"
    },
    "inner": {}
}
signatures["tf.data.experimental.OptimizationOptions"] = {
    "args": {},
    "kwargs": {},
    "inner": {}
}
signatures["tf.math.lgamma"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.invert_permutation"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.arccosh"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.data.experimental.shuffle_and_repeat"] = {
    "args": {
        "buffer_size": "tensor" # tf.int64 scalar tf.Tensor
    },
    "kwargs": {
        "count": "tensor", # tf.int64 scalar tf.Tensor
        "seed": "tensor" # tf.int64 scalar tf.Tensor
    },
    "inner": {}
}
signatures["tf.linalg.matrix_rank"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "tol": "float", # Could also be a tensor, but float is more general for the default value None case.
        "validate_args": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.power"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.random.stateless_binomial"] = {
    "args": {
        "shape": "tensor", # Could be list or tuple, but tensor is more general
        "seed": "tensor",
        "counts": "tensor",
        "probs": "tensor"
    },
    "kwargs": {
        "output_dtype": "dtype",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.zeros_like"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dtype": "dtype",
        "name": "string",
        "layout": "tensor" #  tf.experimental.dtensor.Layout is a tensor
    },
    "inner": {}
}
signatures["tf.linalg.tensor_diag_part"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.random.uniform"] = {
    "args": {
        "shape": "tensor" # A 1-D integer Tensor or Python array. Should be integer, but tensor is closest.
    },
    "kwargs": {
        "minval": "tensor", # A Tensor or Python value of type `dtype`. Assuming tensor for the Tensor case
        "maxval": "tensor", # A Tensor or Python value of type `dtype`. Assuming tensor for the Tensor case
        "dtype": "dtype",
        "seed": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.quantization.fake_quant_with_min_max_vars_gradient"] = {
    "args": {
        "gradients": "tensor",
        "inputs": "tensor",
        "min": "tensor",
        "max": "tensor"
    },
    "kwargs": {
        "num_bits": "integer",
        "narrow_range": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.io.gfile.isdir"] = {
    "args": {
        "path": "string"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.isneginf"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.sum"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer", # Can also be a tuple, but creating only one signature.
        "dtype": "dtype",
        "keepdims": "boolean"
    },
    "inner": {}
}
signatures["tf.experimental.dlpack.from_dlpack"] = {
    "args": {
        "dlcapsule": "string" # Could be a more specific type like 'PyCapsule', but string is the closest.
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.image.adjust_hue"] = {
    "args": {
        "image": "tensor",
        "delta": "float"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.data.experimental.parse_example_dataset"] = {
    "args": {
        "features": "dict"
    },
    "kwargs": {
        "num_parallel_calls": "integer",
        "deterministic": "boolean"
    },
    "inner": {}
}
signatures["tf.autodiff.ForwardAccumulator"] = {
    "args": {
        "primals": "tensor",
        "tangents": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.math.segment_sum"] = {
    "args": {
        "data": "tensor",
        "segment_ids": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.any"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer", # Can be an integer or a tuple of integers. Only including the integer case.
        "keepdims": "boolean"
    },
    "inner": {}
}
signatures["tf.image.resize"] = {
    "args": {
        "images": "tensor",
        "size": "tensor"  # A 1-D int32 Tensor of 2 elements
    },
    "kwargs": {
        "method": "string", # Or image.ResizeMethod, but string is closest
        "preserve_aspect_ratio": "boolean",
        "antialias": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.io.parse_sequence_example"] = {
    "args": {
        "serialized": "tensor"
    },
    "kwargs": {
        "context_features": "dict", # Could be a mapping to a more specific type
        "sequence_features": "dict", # Could be a mapping to a more specific type
        "example_names": "tensor",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.nonzero"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.random.rand"] = {
    "args": {
        "size": "integer" # Could be a tuple of integers as well, but choosing integer for now as it accepts variable number of integer arguments.
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.sparse.to_dense"] = {
    "args": {
        "sp_input": "tensor"
    },
    "kwargs": {
        "default_value": "tensor",  # Scalar value is still a tensor
        "validate_indices": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.data.experimental.TFRecordWriter"] = {
    "args": {
        "filename": "string"
    },
    "kwargs": {
        "compression_type": "string" # It accepts None or a string like "ZLIB" or "GZIP"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.hstack"] = {
    "args": {
        "tup": "tuple" # tup is a tuple of tensors
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.not_equal"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.tensordot"] = {
    "args": {
        "a": "tensor",
        "b": "tensor",
    },
    "kwargs": {
        "axes": "integer", # Could be a tuple as well, but creating a separate signature for that is not feasible.
    },
    "inner": {}
}
signatures["tf.shape_n"] = {
    "args": {
        "input": "tensor_list"
    },
    "kwargs": {
        "out_type": "dtype",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.image.random_contrast"] = {
    "args": {
        "image": "tensor",
        "lower": "float",
        "upper": "float"
    },
    "kwargs": {
        "seed": "integer" # Could also be a Tensor, but integer is more likely
    },
    "inner": {}
}
signatures["tf.image.sample_distorted_bounding_box"] = {
    "args": {
        "image_size": "tensor", # Can be uint8, int8, int16, int32, int64. Using "tensor" as it's the closest match.
        "bounding_boxes": "tensor"
    },
    "kwargs": {
        "seed": "integer",
        "min_object_covered": "float",
        "aspect_ratio_range": "list",
        "area_range": "list",
        "max_attempts": "integer",
        "use_image_if_no_bounding_boxes": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.mod"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.strings.unsorted_segment_join"] = {
    "args": {
        "inputs": "tensor_list",
        "segment_ids": "tensor",
        "num_segments": "integer" # Could also be a tensor, but scalar integers are more common
    },
    "kwargs": {
        "separator": "string",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.image.stateless_random_brightness"] = {
    "args": {
        "image": "tensor",
        "max_delta": "float",
        "seed": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.shape"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out_type": "dtype",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.linalg.diag"] = {
    "args": {
        "diagonal": "tensor"
    },
    "kwargs": {
        "name": "string",
        "k": "integer", # Could also be a tuple
        "num_rows": "integer",
        "num_cols": "integer",
        "padding_value": "tensor", # Should technically be same dtype as diagonal
        "align": "string"
    },
    "inner": {}
}
signatures["tf.linalg.diag_2"] = {
    "args": {
        "diagonal": "tensor"
    },
    "kwargs": {
        "name": "string",
        "k": "tuple",
        "num_rows": "integer",
        "num_cols": "integer",
        "padding_value": "tensor", # Should technically be same dtype as diagonal
        "align": "string"
    },
    "inner": {}
}
signatures["tf.nn.conv_transpose"] = {
    "args": {
        "input": "tensor",
        "filters": "tensor",
        "output_shape": "tensor"
    },
    "kwargs": {
        "strides": "list", # Could also be an integer
        "padding": "string",
        "data_format": "string",
        "dilations": "list", # Could also be an integer
        "name": "string"
    },
    "inner": {}
}
signatures["tf.nn.conv_transpose_1"] = {
    "args": {
        "input": "tensor",
        "filters": "tensor",
        "output_shape": "tensor"
    },
    "kwargs": {
        "strides": "integer",
        "padding": "string",
        "data_format": "string",
        "dilations": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.hsplit"] = {
    "args": {
        "ary": "tensor",
        "indices_or_sections": "integer" # or list[integer], but we'll create another signature for the list case
    },
    "kwargs": {},
    "inner": {}
}

signatures["tf.experimental.numpy.hsplit_2"] = {
    "args": {
        "ary": "tensor",
        "indices_or_sections": "list"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.math.rsqrt"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.finfo"] = {
    "args": {
        "dtype": "dtype"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.less_equal"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.linalg.LinearOperatorLowerTriangular"] = {
    "args": {
        "tril": "tensor"
    },
    "kwargs": {
        "is_non_singular": "boolean",
        "is_self_adjoint": "boolean",
        "is_positive_definite": "boolean",
        "is_square": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.linalg.trace"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.strings.as_string"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "precision": "integer",
        "scientific": "boolean",
        "shortest": "boolean",
        "width": "integer",
        "fill": "string",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.switch_case"] = {
    "args": {
        "branch_index": "tensor", # Should be an integer tensor
        "branch_fns": "list" # Could also be a dict
    },
    "kwargs": {
        "default": "list", # Should be a callable but can't represent in types, should return tensors, or None
        "name": "string"
    },
    "inner": {}
}
signatures["tf.data.experimental.save"] = {
    "args": {
        "dataset": "tensor", # Should this be tf.data.Dataset?
        "path": "string"
    },
    "kwargs": {
        "compression": "string",
        "shard_func": "tensor", # Should be callable?
        "checkpoint_args": "list" # Should this be dict?
    },
    "inner": {}
}
signatures["tf.nn.separable_conv2d"] = {
    "args": {
        "input": "tensor",
        "depthwise_filter": "tensor",
        "pointwise_filter": "tensor",
        "strides": "list" # could also be tuple, but list seems more appropriate given the description
    },
    "kwargs": {
        "padding": "string", # could be string or list
        "data_format": "string",
        "dilations": "list", # could also be tuple, but list seems more appropriate given the description
        "name": "string"
    },
    "inner": {}
}
signatures["tf.nn.separable_conv2d_padding_list"] = {
    "args": {
        "input": "tensor",
        "depthwise_filter": "tensor",
        "pointwise_filter": "tensor",
        "strides": "list" # could also be tuple, but list seems more appropriate given the description
    },
    "kwargs": {
        "padding": "list",
        "data_format": "string",
        "dilations": "list", # could also be tuple, but list seems more appropriate given the description
        "name": "string"
    },
    "inner": {}
}
signatures["tf.io.decode_bmp"] = {
    "args": {
        "contents": "string"
    },
    "kwargs": {
        "channels": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.searchsorted"] = {
    "args": {
        "sorted_sequence": "tensor",
        "values": "tensor"
    },
    "kwargs": {
        "side": "string",
        "out_type": "dtype",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.feature_column.categorical_column_with_vocabulary_list"] = {
    "args": {
        "key": "string",
        "vocabulary_list": "list"
    },
    "kwargs": {
        "dtype": "dtype",
        "default_value": "integer",
        "num_oov_buckets": "integer"
    },
    "inner": {}
}
signatures["tf.io.decode_jpeg"] = {
    "args": {
        "contents": "string"
    },
    "kwargs": {
        "channels": "integer",
        "ratio": "integer",
        "fancy_upscaling": "boolean",
        "try_recover_truncated": "boolean",
        "acceptable_fraction": "float",
        "dct_method": "string",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.feature_column.numeric_column"] = {
    "args": {
        "key": "string"
    },
    "kwargs": {
        "shape": "tuple", # or integer?
        "default_value": "list", # or single value compatible with dtype? It could be float/integer
        "dtype": "dtype",
        "normalizer_fn": "list" # Function type is not available, using list
    },
    "inner": {}
}
signatures["tf.strings.unicode_decode_with_offsets"] = {
    "args": {
        "input": "tensor",
        "input_encoding": "string"
    },
    "kwargs": {
        "errors": "string",
        "replacement_char": "integer",
        "replace_control_characters": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.signal.fft3d"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.image.per_image_standardization"] = {
    "args": {
        "image": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.array_equal"] = {
    "args": {
        "a1": "tensor",
        "a2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.data.experimental.unique"] = {
    "args": {},
    "kwargs": {},
    "inner": {}
}
signatures["tf.nn.embedding_lookup_sparse"] = {
    "args": {
        "params": "tensor_list",  # Can be a single tensor or a list of tensors
        "sp_ids": "tensor"  # SparseTensor or RaggedTensor
    },
    "kwargs": {
        "sp_weights": "tensor",  # SparseTensor or RaggedTensor or None
        "combiner": "string",
        "max_norm": "float",
        "name": "string",
        "allow_fast_lookup": "boolean"
    },
    "inner": {}
}
signatures["tf.nn.elu"] = {
    "args": {
        "features": "tensor" # Could also be a float, but tensor seems more general
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.image.central_crop"] = {
    "args": {
        "image": "tensor",
        "central_fraction": "float"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.nn.depthwise_conv2d_backprop_input"] = {
    "args": {
        "input_sizes": "tensor",
        "filter": "tensor",
        "out_backprop": "tensor",
        "strides": "list"
    },
    "kwargs": {
        "padding": "string", # Could also be a list of lists. Creating a separate signature for this isn't required.
        "data_format": "string",
        "dilations": "list",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.dstack"] = {
    "args": {
        "tup": "tuple" # could also be a tensor_list?
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.nn.sigmoid_cross_entropy_with_logits"] = {
    "args": {
        "labels": "tensor",
        "logits": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.sigmoid"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.data.experimental.group_by_reducer"] = {
    "args": {
        "key_func": "list",  # Assuming a callable function is best represented as a list
        "reducer": "list"  # Assuming an instance of Reducer is best represented as a list
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.select"] = {
    "args": {
        "condlist": "list", # List of tensors containing condition
        "choicelist": "list" # List of tensors containing choices
    },
    "kwargs": {
        "default": "tensor" # Default could also be a scalar value, but tensor is the closest type.
    },
    "inner": {}
}
signatures["tf.image.transpose"] = {
    "args": {
        "image": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.equal"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.roll"] = {
    "args": {
        "input": "tensor",
        "shift": "tensor", # Can be int32 or int64 Tensor, but representing it as tensor
        "axis": "tensor" # Can be int32 or int64 Tensor, but representing it as tensor
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.unique_with_counts"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "out_idx": "dtype",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.signal.dct"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "type": "integer",
        "n": "integer",
        "axis": "integer",
        "norm": "string",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.result_type"] = {
    "args": {
        "arrays_and_dtypes": "list" # Could also be tuple, but list seems more general
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.dot"] = {
    "args": {
        "a": "tensor",
        "b": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.isrealobj"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.bitwise.bitwise_xor"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.unsorted_segment_max"] = {
    "args": {
        "data": "tensor",
        "segment_ids": "tensor",
        "num_segments": "integer" # Can also be a tensor, but choosing the most likely type based on description.
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.data.experimental.make_csv_dataset"] = {
    "args": {
        "file_pattern": "list", # Could be a list of strings, so keeping it a list
        "batch_size": "integer"
    },
    "kwargs": {
        "column_names": "list",
        "column_defaults": "list",
        "label_name": "string",
        "select_columns": "list", # This can be either list of integers or strings
        "field_delim": "string",
        "use_quote_delim": "boolean",
        "na_value": "string",
        "header": "boolean",
        "num_epochs": "integer",
        "shuffle": "boolean",
        "shuffle_buffer_size": "integer",
        "shuffle_seed": "integer",
        "prefetch_buffer_size": "integer",
        "num_parallel_reads": "integer",
        "sloppy": "boolean",
        "num_rows_for_inference": "integer",
        "compression_type": "string",
        "ignore_errors": "boolean",
        "encoding": "string"
    },
    "inner": {}
}
signatures["tf.image.rgb_to_grayscale"] = {
    "args": {
        "images": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.logical_and"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.io.gfile.mkdir"] = {
    "args": {
        "path": "string"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.linalg.det"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.xlogy"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.nn.avg_pool"] = {
    "args": {
        "input": "tensor",
        "ksize": "list", # Could also be integer, but defaulting to list as that's more general
        "strides": "list", # Could also be integer, but defaulting to list as that's more general
        "padding": "string"
    },
    "kwargs": {
        "data_format": "string",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.pad"] = {
    "args": {
        "array": "tensor",
        "pad_width": "list", # Or tuple depending on the shape and content, but list seems more general
        "mode": "string"
    },
    "kwargs": {
    },
    "inner": {}
}
signatures["tf.math.special.spence"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.space_to_batch_nd"] = {
    "args": {
        "input": "tensor",
        "block_shape": "tensor",
        "paddings": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.truediv"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.feature_column.sequence_categorical_column_with_identity"] = {
    "args": {
        "key": "string",
        "num_buckets": "integer"
    },
    "kwargs": {
        "default_value": "integer"
    },
    "inner": {}
}
signatures["tf.linalg.LinearOperatorZeros"] = {
    "args": {
        "num_rows": "integer"
    },
    "kwargs": {
        "num_columns": "integer", # Could be None, but assuming integer if provided
        "batch_shape": "list", # Could be None, but assuming list if provided
        "dtype": "dtype",
        "is_non_singular": "boolean",
        "is_self_adjoint": "boolean",
        "is_positive_definite": "boolean",
        "is_square": "boolean",
        "assert_proper_shapes": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.nn.ctc_loss"] = {
    "args": {
        "labels": "tensor",
        "logits": "tensor",
        "label_length": "tensor",
        "logit_length": "tensor"
    },
    "kwargs": {
        "logits_time_major": "boolean",
        "unique": "tensor",  # Should ideally be computed by ctc_unique_labels which can return multiple types of tensors. Assuming tensor type here
        "blank_index": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.random.uniform_candidate_sampler"] = {
    "args": {
        "true_classes": "tensor",
        "num_true": "integer",
        "num_sampled": "integer",
        "unique": "boolean",
        "range_max": "integer"
    },
    "kwargs": {
        "seed": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.image.rgb_to_yiq"] = {
    "args": {
        "images": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.io.decode_image"] = {
    "args": {
        "contents": "string"
    },
    "kwargs": {
        "channels": "integer",
        "dtype": "dtype",
        "name": "string",
        "expand_animations": "boolean"
    },
    "inner": {}
}
signatures["tf.math.reciprocal_no_nan"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.eye"] = {
    "args": {
        "N": "integer"
    },
    "kwargs": {
        "M": "integer",
        "k": "integer",
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["tf.tuple"] = {
    "args": {
        "tensors": "tensor_list"
    },
    "kwargs": {
        "control_inputs": "list", # Could be a list of ops
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.equal"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.nn.RNNCellDropoutWrapper"] = {
    "args": {
        "cell": "tensor" # Should be RNNCell but approximating with tensor
    },
    "kwargs": {
        "input_keep_prob": "float",
        "output_keep_prob": "float",
        "state_keep_prob": "float",
        "variational_recurrent": "boolean",
        "input_size": "integer", # Could be integer or a tensor
        "dtype": "dtype",
        "seed": "integer" # Could also be a Tensor
    },
    "inner": {}
}
signatures["tf.linalg.matrix_transpose"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "name": "string",
        "conjugate": "boolean"
    },
    "inner": {}
}
signatures["tf.stack"] = {
    "args": {
        "values": "tensor_list"
    },
    "kwargs": {
        "axis": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.ptp"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer", # Could also be a tuple of integers
        "keepdims": "boolean"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.abs"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.linalg.tridiagonal_matmul"] = {
    "args": {
        "diagonals": "tensor", # Can also be a tuple of tensors, creating new signature to address it.
        "rhs": "tensor"
    },
    "kwargs": {
        "diagonals_format": "string",
        "name": "string"
    },
    "inner": {}
}

signatures["tf.linalg.tridiagonal_matmul_2"] = {
    "args": {
        "diagonals": "tuple",
        "rhs": "tensor"
    },
    "kwargs": {
        "diagonals_format": "string",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.sparse.expand_dims"] = {
    "args": {
        "sp_input": "tensor"  # SparseTensor is a type of tensor
    },
    "kwargs": {
        "axis": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.diff"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "n": "integer",
        "axis": "integer"
    },
    "inner": {}
}
signatures["tf.math.floormod"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.diag_indices"] = {
    "args": {
        "n": "integer"
    },
    "kwargs": {
        "ndim": "integer"
    },
    "inner": {}
}
signatures["tf.strings.unicode_transcode"] = {
    "args": {
        "input": "tensor",
        "input_encoding": "string",
        "output_encoding": "string"
    },
    "kwargs": {
        "errors": "string",
        "replacement_char": "integer",
        "replace_control_characters": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.GradientTape"] = {
    "args": {},
    "kwargs": {
        "persistent": "boolean",
        "watch_accessed_variables": "boolean"
    },
    "inner": {}
}
signatures["tf.queue.QueueBase"] = {
    "args": {
        "dtypes": "list",  # List of dtypes
        "shapes": "list",  # List of shapes (tuples or lists of integers)
        "names": "list",  # List of strings
        "queue_ref": "tensor" # A queue resource handle.
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.nn.bias_add"] = {
    "args": {
        "value": "tensor",
        "bias": "tensor"
    },
    "kwargs": {
        "data_format": "string",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.bitwise_or"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.signal.hamming_window"] = {
    "args": {
        "window_length": "tensor" # Could also be integer, but tensor is more general
    },
    "kwargs": {
        "periodic": "boolean",
        "dtype": "dtype",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.diagonal"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "offset": "integer",
        "axis1": "integer",
        "axis2": "integer"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.isposinf"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.signal.irfft2d"] = {
    "args": {
        "input_tensor": "tensor"
    },
    "kwargs": {
        "fft_length": "tensor", # Tensor of type int32 and shape [2] is a tensor
        "name": "string"
    },
    "inner": {}
}
signatures["tf.ragged.range"] = {
    "args": {
        "starts": "tensor",
    },
    "kwargs": {
        "limits": "tensor",
        "deltas": "tensor",
        "dtype": "dtype",
        "name": "string",
        "row_splits_dtype": "dtype"
    },
    "inner": {}
}
signatures["tf.nn.selu"] = {
    "args": {
        "features": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.feature_column.sequence_categorical_column_with_vocabulary_list"] = {
    "args": {
        "key": "string",
        "vocabulary_list": "list" # Could be more specific, like list of strings or list of integers depending on dtype
    },
    "kwargs": {
        "dtype": "dtype",
        "default_value": "integer",
        "num_oov_buckets": "integer"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.kron"] = {
    "args": {
        "a": "tensor",
        "b": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.math.argmax"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "output_type": "dtype",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.strings.format"] = {
    "args": {
        "template": "string",
        "inputs": "tensor_list" # Can also be a single tensor, create new signature?
    },
    "kwargs": {
        "placeholder": "string",
        "summarize": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.image.stateless_sample_distorted_bounding_box"] = {
    "args": {
        "image_size": "tensor", # Could be a list or tuple of integers as well
        "bounding_boxes": "tensor",
        "seed": "tensor"
    },
    "kwargs": {
        "min_object_covered": "float",
        "aspect_ratio_range": "list",
        "area_range": "list",
        "max_attempts": "integer",
        "use_image_if_no_bounding_boxes": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.nn.max_pool2d"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "ksize": "list", # could also be integer
        "strides": "list", # could also be integer
        "padding": "string", # could also be list
        "data_format": "string",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.nn.max_pool2d_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "ksize": "integer",
        "strides": "integer",
        "padding": "string", # could also be list
        "data_format": "string",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.nn.max_pool2d_2"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "ksize": "list", # could also be integer
        "strides": "integer",
        "padding": "string", # could also be list
        "data_format": "string",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.nn.max_pool2d_3"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "ksize": "integer",
        "strides": "list",
        "padding": "string", # could also be list
        "data_format": "string",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.nn.max_pool2d_4"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "ksize": "list", # could also be integer
        "strides": "list",
        "padding": "list",
        "data_format": "string",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.nn.max_pool2d_5"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "ksize": "integer",
        "strides": "integer",
        "padding": "list",
        "data_format": "string",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.nn.max_pool2d_6"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "ksize": "list", # could also be integer
        "strides": "integer",
        "padding": "list",
        "data_format": "string",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.nn.max_pool2d_7"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "ksize": "integer",
        "strides": "list",
        "padding": "list",
        "data_format": "string",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.data.experimental.service.register_dataset"] = {
    "args": {
        "service": "string", # Could also be a tuple, but we'll only create one signature
        "dataset": "tensor"
    },
    "kwargs": {
        "compression": "string",
        "dataset_id": "string"
    },
    "inner": {}
}
signatures["tf.math.add_n"] = {
    "args": {
        "inputs": "tensor_list"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.signal.rfft2d"] = {
    "args": {
        "input_tensor": "tensor"
    },
    "kwargs": {
        "fft_length": "tensor", # Tensor of type int32 and shape [2]
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.cosh"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.linalg.LinearOperatorComposition"] = {
    "args": {
        "operators": "list"
    },
    "kwargs": {
        "is_non_singular": "boolean",
        "is_self_adjoint": "boolean",
        "is_positive_definite": "boolean",
        "is_square": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.ravel"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.vander"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "N": "integer", # Could be None, but integer is the closest
        "increasing": "boolean"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.arccos"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.math.erfcinv"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.all"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer", # Could also be a tuple or list of integers
        "keepdims": "boolean"
    },
    "inner": {}
}
signatures["tf.bitwise.bitwise_or"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.nn.gelu"] = {
    "args": {
        "features": "tensor"
    },
    "kwargs": {
        "approximate": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.multiply"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.linalg.eigvalsh"] = {
    "args": {
        "tensor": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.sparse.cross_hashed"] = {
    "args": {
        "inputs": "tensor_list"
    },
    "kwargs": {
        "num_buckets": "integer",
        "hash_key": "integer", # Could also be None, but integer seems like the best fit
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.cbrt"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.vsplit"] = {
    "args": {
        "ary": "tensor",
        "indices_or_sections": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.norm"] = {
    "args": {
        "tensor": "tensor"
    },
    "kwargs": {
        "ord": "string", # Could be a float as well, but the doc says string
        "axis": "tuple", # Can be an integer or None as well.
        "keepdims": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.norm_1"] = {
    "args": {
        "tensor": "tensor"
    },
    "kwargs": {
        "ord": "string", # Could be a float as well, but the doc says string
        "axis": "integer", # Can be an integer or None as well.
        "keepdims": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.norm_2"] = {
    "args": {
        "tensor": "tensor"
    },
    "kwargs": {
        "ord": "string", # Could be a float as well, but the doc says string
        "axis": "list", # If axis is None, the input is considered a vector
        "keepdims": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.float_power"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.min"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer", # Can also be a tuple of integers. But we only have integer and tuple. So we create a new signature for tuple.
        "keepdims": "boolean"
    },
    "inner": {}
}

signatures["tf.experimental.numpy.min_2"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "tuple",
        "keepdims": "boolean"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.size"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "axis": "integer" # or None, but integer seems like the best fit
    },
    "inner": {}
}
signatures["tf.experimental.numpy.nanmean"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer", # Could also be tuple of integers or None
        "dtype": "dtype",
        "keepdims": "boolean"
    },
    "inner": {}
}
signatures["tf.image.total_variation"] = {
    "args": {
        "images": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.io.read_file"] = {
    "args": {
        "filename": "string"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.xdivy"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.recompute_grad"] = {
    "args": {
        "f": "function"  # Could be a function or a tf.keras Model/Layer. No good type available,
                         #  'function' seems closest
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.image.flip_left_right"] = {
    "args": {
        "image": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.math.floor"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.nn.max_pool1d"] = {
    "args": {
        "input": "tensor",
        "ksize": "list", # Could also be integer, but list is more general
        "strides": "list", # Could also be integer, but list is more general
        "padding": "string" # Could also be list
    },
    "kwargs": {
        "data_format": "string",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.data.experimental.copy_to_device"] = {
    "args": {
        "target_device": "string"
    },
    "kwargs": {
        "source_device": "string"
    },
    "inner": {}
}
signatures["tf.where_1"] = {
    "args": {
        "condition": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}

signatures["tf.where_2"] = {
    "args": {
        "condition": "tensor",
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.nn.weighted_moments"] = {
    "args": {
        "x": "tensor",
        "axes": "tensor", # Could also be list or tuple of integers
        "frequency_weights": "tensor"
    },
    "kwargs": {
        "keepdims": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.nanprod"] = {
    "args": {
        "a": "tensor",
    },
    "kwargs": {
        "axis": "integer",  # Could also be a tuple or list of integers
        "dtype": "dtype",
        "keepdims": "boolean",
    },
    "inner": {}
}
signatures["tf.data.experimental.Reducer"] = {
    "args": {
        "init_func": "list",  # Assuming a callable object can be represented as a list
        "reduce_func": "list", # Assuming a callable object can be represented as a list
        "finalize_func": "list" # Assuming a callable object can be represented as a list
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.math.greater_equal"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.reduce_euclidean_norm"] = {
    "args": {
        "input_tensor": "tensor"
    },
    "kwargs": {
        "axis": "integer", # Can also be a list
        "keepdims": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.argsort"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "kind": "string",
        "order": "list" # Or tuple, depending on numpy implementation, but list is closer.
    },
    "inner": {}
}
signatures["tf.sparse.softmax"] = {
    "args": {
        "sp_input": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.sparse.maximum"] = {
    "args": {
        "sp_a": "tensor",
        "sp_b": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.linalg.band_part"] = {
    "args": {
        "input": "tensor",
        "num_lower": "integer",
        "num_upper": "integer"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.image.adjust_gamma"] = {
    "args": {
        "image": "tensor"
    },
    "kwargs": {
        "gamma": "float",
        "gain": "float"
    },
    "inner": {}
}
signatures["tf.image.ssim_multiscale"] = {
    "args": {
        "img1": "tensor",
        "img2": "tensor",
        "max_val": "float"
    },
    "kwargs": {
        "power_factors": "tuple",
        "filter_size": "integer",
        "filter_sigma": "float",
        "k1": "float",
        "k2": "float"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.logical_xor"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.sin"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.train.load_checkpoint"] = {
    "args": {
        "ckpt_dir_or_file": "string"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.ix_"] = {
    "args": {
        "args": "tensor_list" # It accepts multiple tensor arguments
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.math.logical_xor"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.nn.avg_pool1d"] = {
    "args": {
        "input": "tensor",
        "ksize": "list", # Could also be integer
        "strides": "list", # Could also be integer
    },
    "kwargs": {
        "padding": "string",
        "data_format": "string",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.nn.avg_pool1d_1"] = {
    "args": {
        "input": "tensor",
        "ksize": "integer",
        "strides": "integer",
    },
    "kwargs": {
        "padding": "string",
        "data_format": "string",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.size"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out_type": "dtype",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.sparse.mask"] = {
    "args": {
        "a": "tensor",
        "mask_indices": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.linalg.LinearOperatorFullMatrix"] = {
    "args": {
        "matrix": "tensor"
    },
    "kwargs": {
        "is_non_singular": "boolean",
        "is_self_adjoint": "boolean",
        "is_positive_definite": "boolean",
        "is_square": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.nn.conv3d"] = {
    "args": {
        "input": "tensor",
        "filters": "tensor",
        "strides": "list",
        "padding": "string"
    },
    "kwargs": {
        "data_format": "string",
        "dilations": "list",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.nn.depthwise_conv2d"] = {
    "args": {
        "input": "tensor",
        "filter": "tensor",
        "strides": "list" # Could be a tuple too
    },
    "kwargs": {
        "padding": "string", # or list
        "data_format": "string",
        "dilations": "list", # Could be a tuple too
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.special.fresnel_sin"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.linalg.normalize"] = {
    "args": {
        "tensor": "tensor"
    },
    "kwargs": {
        "ord": "string", # Could also be float or integer depending on the context
        "axis": "tuple", # Could also be integer or None depending on the context
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.remainder"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.tensor_scatter_nd_update"] = {
    "args": {
        "tensor": "tensor",
        "indices": "tensor",
        "updates": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.rint"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.squeeze"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "axis": "list", # Could also be integer, but list is more general and includes integer lists
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.rad2deg"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.image.generate_bounding_box_proposals"] = {
    "args": {
        "scores": "tensor",
        "bbox_deltas": "tensor",
        "image_info": "tensor",
        "anchors": "tensor"
    },
    "kwargs": {
        "nms_threshold": "float",
        "pre_nms_topn": "integer",
        "min_size": "float",
        "post_nms_topn": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.strings.length"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "unit": "string",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.image.yiq_to_rgb"] = {
    "args": {
        "images": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.feature_column.weighted_categorical_column"] = {
    "args": {
        "categorical_column": "list", # CategoricalColumn created by categorical_column_with_* functions, but list is the closest match
        "weight_feature_key": "string"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["tf.linalg.LinearOperatorCirculant"] = {
    "args": {
        "spectrum": "tensor"
    },
    "kwargs": {
        "input_output_dtype": "dtype",
        "is_non_singular": "boolean",
        "is_self_adjoint": "boolean",
        "is_positive_definite": "boolean",
        "is_square": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.take"] = {
    "args": {
        "a": "tensor",
        "indices": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "out": "tensor",  # Could also be None, but "tensor" is closest
        "mode": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.allclose"] = {
    "args": {
        "a": "tensor",
        "b": "tensor"
    },
    "kwargs": {
        "rtol": "float",
        "atol": "float",
        "equal_nan": "boolean"
    },
    "inner": {}
}
signatures["tf.unstack"] = {
    "args": {
        "value": "tensor"
    },
    "kwargs": {
        "num": "integer",
        "axis": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.saved_model.contains_saved_model"] = {
    "args": {
        "export_dir": "string"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.nn.sparse_softmax_cross_entropy_with_logits"] = {
    "args": {
        "labels": "tensor", # Could be integer tensor more specifically
        "logits": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.meshgrid"] = {
    "args": {
        "args": "tensor_list"
    },
    "kwargs": {
        "indexing": "string",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.reduce_any"] = {
    "args": {
        "input_tensor": "tensor"
    },
    "kwargs": {
        "axis": "integer", # Could also be a list or tuple of integers
        "keepdims": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.unsorted_segment_sqrt_n"] = {
    "args": {
        "data": "tensor",
        "segment_ids": "tensor",
        "num_segments": "integer"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.ragged.cross_hashed"] = {
    "args": {
        "inputs": "tensor_list"
    },
    "kwargs": {
        "num_buckets": "integer",
        "hash_key": "integer", # Could also be None, but integer is closest
        "name": "string"
    },
    "inner": {}
}
signatures["tf.audio.decode_wav"] = {
    "args": {
        "contents": "string"
    },
    "kwargs": {
        "desired_channels": "integer",
        "desired_samples": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.tri"] = {
    "args": {
        "N": "integer"
    },
    "kwargs": {
        "M": "integer", # Could also be None which isn't one of the specified types
        "k": "integer",
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["tf.compat.as_bytes"] = {
    "args": {
        "bytes_or_text": "string" # could be bytes, bytearray, str, unicode so string is the closest
    },
    "kwargs": {
        "encoding": "string"
    },
    "inner": {}
}
signatures["tf.strings.to_number"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "out_type": "dtype",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.where"] = {
    "args": {
        "condition": "tensor",
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.identity"] = {
    "args": {
        "n": "integer"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["tf.boolean_mask"] = {
    "args": {
        "tensor": "tensor",
        "mask": "tensor"
    },
    "kwargs": {
        "axis": "tensor", # A 0-D int Tensor representing the axis, so it's a tensor
        "name": "string"
    },
    "inner": {}
}
signatures["tf.quantization.quantize_and_dequantize"] = {
    "args": {
        "input": "tensor",
        "input_min": "tensor",
        "input_max": "tensor"
    },
    "kwargs": {
        "signed_input": "boolean",
        "num_bits": "integer",
        "range_given": "boolean",
        "round_mode": "string",
        "name": "string",
        "narrow_range": "boolean",
        "axis": "integer"
    },
    "inner": {}
}
signatures["tf.quantization.fake_quant_with_min_max_args"] = {
    "args": {
        "inputs": "tensor"
    },
    "kwargs": {
        "min": "float",
        "max": "float",
        "num_bits": "integer",
        "narrow_range": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.strings.strip"] = {
    "args": {
        "input": "string" # A `Tensor` of type `string` is string
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.sort"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "kind": "string",
        "order": "list" # Could be list of strings based on numpy documentation.
    },
    "inner": {}
}
signatures["tf.experimental.numpy.copy"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.sinh"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.math.asinh"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.image.crop_and_resize"] = {
    "args": {
        "image": "tensor",
        "boxes": "tensor",
        "box_indices": "tensor",
        "crop_size": "tensor"
    },
    "kwargs": {
        "method": "string",
        "extrapolation_value": "float",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.nn.RNNCellResidualWrapper"] = {
    "args": {
        "cell": "object" # Could be a more specific type if known
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.strings.split"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "sep": "tensor", # it is a 0-D string Tensor, which can be represented by tensor
        "maxsplit": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.special.bessel_k0e"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.dynamic_partition"] = {
    "args": {
        "data": "tensor",
        "partitions": "tensor", # A Tensor of type `int32`.
        "num_partitions": "integer" # An `int` that is `>= 1`.
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.strings.regex_replace"] = {
    "args": {
        "input": "tensor",
        "pattern": "string",  # Could also be a tensor of type string
        "rewrite": "string"  # Could also be a tensor of type string
    },
    "kwargs": {
        "replace_global": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.data.experimental.bucket_by_sequence_length"] = {
    "args": {
        "element_length_func": "list", # Assuming function is represented by list
    },
    "kwargs": {
        "bucket_boundaries": "list",
        "bucket_batch_sizes": "list",
        "padded_shapes": "tuple", # Nested structure of `tf.TensorShape`
        "padding_values": "tensor", # Could also be list
        "pad_to_bucket_boundary": "boolean",
        "no_padding": "boolean",
        "drop_remainder": "boolean"
    },
    "inner": {}
}
signatures["tf.io.gfile.glob"] = {
    "args": {
        "pattern": "string" # Could also be "list" of strings or "tuple" of strings
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.sparse.reshape"] = {
    "args": {
        "sp_input": "tensor",
        "shape": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.signal.ifft2d"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.linspace"] = {
    "args": {
        "start": "tensor",
        "stop": "tensor"
    },
    "kwargs": {
        "num": "integer",
        "endpoint": "boolean",
        "retstep": "boolean",
        "dtype": "dtype",
        "axis": "integer"
    },
    "inner": {}
}
signatures["tf.math.segment_mean"] = {
    "args": {
        "data": "tensor",
        "segment_ids": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.ones"] = {
    "args": {
        "shape": "integer",  # Could also be a tuple or list of integers
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["tf.sparse.split"] = {
    "args": {
        "sp_input": "tensor" # SparseTensor is a type of tensor
    },
    "kwargs": {
        "num_split": "integer",
        "axis": "tensor", # A 0-D `int32` `Tensor` can be a tensor
        "name": "string"
    },
    "inner": {}
}
signatures["tf.io.gfile.remove"] = {
    "args": {
        "path": "string"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.image.extract_glimpse"] = {
    "args": {
        "input": "tensor",
        "size": "tensor", # A tensor of type int32 can be represented as "tensor"
        "offsets": "tensor"
    },
    "kwargs": {
        "centered": "boolean",
        "normalized": "boolean",
        "noise": "string",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.data.experimental.cardinality"] = {
    "args": {
        "dataset": "tensor" # Should be tf.data.Dataset, but "tensor" is the closest available type
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.signal.vorbis_window"] = {
    "args": {
        "window_length": "tensor" # Should this be integer instead of tensor?
    },
    "kwargs": {
        "dtype": "dtype",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.batch_to_space"] = {
    "args": {
        "input": "tensor",
        "block_shape": "tensor",
        "crops": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.erfinv"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.linalg.LinearOperatorDiag"] = {
    "args": {
        "diag": "tensor"
    },
    "kwargs": {
        "is_non_singular": "boolean",
        "is_self_adjoint": "boolean",
        "is_positive_definite": "boolean",
        "is_square": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.case"] = {
    "args": {
        "pred_fn_pairs": "list" # List of pairs of boolean scalar tensor and callable
    },
    "kwargs": {
        "default": "list", # Callable that returns a list of tensors (or None)
        "exclusive": "boolean",
        "strict": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.data.experimental.ignore_errors"] = {
    "args": {},
    "kwargs": {
        "log_warning": "boolean"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.negative"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.feature_column.make_parse_example_spec"] = {
    "args": {
        "feature_columns": "list" # iterable containing FeatureColumn instances.
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.io.gfile.makedirs"] = {
    "args": {
        "path": "string"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.isnan"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.sparse.reorder"] = {
    "args": {
        "sp_input": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.io.gfile.GFile"] = {
    "args": {
        "name": "string"
    },
    "kwargs": {
        "mode": "string"
    },
    "inner": {}
}
signatures["tf.image.non_max_suppression_padded"] = {
    "args": {
        "boxes": "tensor",
        "scores": "tensor",
        "max_output_size": "integer"
    },
    "kwargs": {
        "iou_threshold": "float",
        "score_threshold": "float",
        "pad_to_max_output_size": "boolean",
        "name": "string",
        "sorted_input": "boolean",
        "canonicalized_coordinates": "boolean",
        "tile_size": "integer"
    },
    "inner": {}
}
signatures["tf.math.zeta"] = {
    "args": {
        "x": "tensor",
        "q": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.logaddexp"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.nn.softmax"] = {
    "args": {
        "logits": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.pad"] = {
    "args": {
        "tensor": "tensor",
        "paddings": "tensor"
    },
    "kwargs": {
        "mode": "string",
        "constant_values": "tensor", # Documentation says "Must be same type as `tensor`.", so using tensor as the type.
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.isscalar"] = {
    "args": {
        "num": "tensor" # Could also be a more general type, but tensor is the most relevant in TF
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.math.negative"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.exp"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.queue.PriorityQueue"] = {
    "args": {
        "capacity": "integer",
        "types": "list" # List of dtypes
    },
    "kwargs": {
        "shapes": "list", # Could also be tuple
        "names": "list", # List of strings
        "shared_name": "string",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.train.checkpoints_iterator"] = {
    "args": {
        "checkpoint_dir": "string"
    },
    "kwargs": {
        "min_interval_secs": "integer",  # Could be a float, but integer is closer
        "timeout": "integer", # Could be a float, but integer is closer
        "timeout_fn": "list" # function is of type list because function types don't exist
    },
    "inner": {}
}
signatures["tf.required_space_to_batch_paddings"] = {
    "args": {
        "input_shape": "tensor",
        "block_shape": "tensor"
    },
    "kwargs": {
        "base_paddings": "tensor", # Could be a list of lists, but it is passed as a tensor
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.minimum"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.io.decode_and_crop_jpeg"] = {
    "args": {
        "contents": "string",
        "crop_window": "tensor"
    },
    "kwargs": {
        "channels": "integer",
        "ratio": "integer",
        "fancy_upscaling": "boolean",
        "try_recover_truncated": "boolean",
        "acceptable_fraction": "float",
        "dct_method": "string",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.random.randint"] = {
    "args": {
        "low": "integer",
    },
    "kwargs": {
        "high": "integer", # Could also be None, but integer is the most specific
        "size": "integer", # Could also be tuple of integers or None, but integer is the most specific
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["tf.math.cumulative_logsumexp"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "axis": "integer", # Could be tensor as well
        "exclusive": "boolean",
        "reverse": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.bitwise.left_shift"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.stack"] = {
    "args": {
        "arrays": "tensor_list"
    },
    "kwargs": {
        "axis": "integer"
    },
    "inner": {}
}
signatures["tf.math.accumulate_n"] = {
    "args": {
        "inputs": "tensor_list"
    },
    "kwargs": {
        "shape": "list", # Could also be tuple
        "tensor_dtype": "dtype",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.tanh"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.ragged.boolean_mask"] = {
    "args": {
        "data": "tensor",
        "mask": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.reduce_mean"] = {
    "args": {
        "input_tensor": "tensor"
    },
    "kwargs": {
        "axis": "integer", # Could be a list or tuple of integers
        "keepdims": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.reverse_sequence"] = {
    "args": {
        "input": "tensor",
        "seq_lengths": "tensor"
    },
    "kwargs": {
        "seq_axis": "integer",
        "batch_axis": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.linalg.LinearOperatorBlockDiag"] = {
    "args": {
        "operators": "list" # List of LinearOperator objects
    },
    "kwargs": {
        "is_non_singular": "boolean",
        "is_self_adjoint": "boolean",
        "is_positive_definite": "boolean",
        "is_square": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.image.combined_non_max_suppression"] = {
    "args": {
        "boxes": "tensor",
        "scores": "tensor",
        "max_output_size_per_class": "integer",
        "max_total_size": "integer"
    },
    "kwargs": {
        "iou_threshold": "float",
        "score_threshold": "float",
        "pad_per_class": "boolean",
        "clip_boxes": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.imag"] = {
    "args": {
        "val": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.nn.silu"] = {
    "args": {
        "features": "tensor"
    },
    "kwargs": {
        "beta": "tensor" # The documentation mentions that beta is a Tensor, not float
    },
    "inner": {}
}
signatures["tf.experimental.numpy.logical_or"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.math.acos"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.quantization.fake_quant_with_min_max_vars_per_channel_gradient"] = {
    "args": {
        "gradients": "tensor",
        "inputs": "tensor",
        "min": "tensor",
        "max": "tensor"
    },
    "kwargs": {
        "num_bits": "integer",
        "narrow_range": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.sin"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.strings.to_hash_bucket_strong"] = {
    "args": {
        "input": "tensor",
        "num_buckets": "integer",
        "key": "list"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.isclose"] = {
    "args": {
        "a": "tensor",
        "b": "tensor"
    },
    "kwargs": {
        "rtol": "float",
        "atol": "float",
        "equal_nan": "boolean"
    },
    "inner": {}
}
signatures["tf.math.squared_difference"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.in_top_k"] = {
    "args": {
        "targets": "tensor",
        "predictions": "tensor",
        "k": "integer"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.linalg.LinearOperatorCirculant3D"] = {
    "args": {
        "spectrum": "tensor"
    },
    "kwargs": {
        "input_output_dtype": "dtype",
        "is_non_singular": "boolean",
        "is_self_adjoint": "boolean",
        "is_positive_definite": "boolean",
        "is_square": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.strings.bytes_split"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.vstack"] = {
    "args": {
        "tup": "tuple" # tup is a tuple of tensors
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.image.stateless_random_hue"] = {
    "args": {
        "image": "tensor",
        "max_delta": "float",
        "seed": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.divide"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.vdot"] = {
    "args": {
        "a": "tensor",
        "b": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.ones_like"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "dtype": "dtype",
        "name": "string",
        "layout": "tensor" # Best guess as layout is tf.experimental.dtensor.Layout
    },
    "inner": {}
}
signatures["tf.experimental.numpy.heaviside"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.image.adjust_saturation"] = {
    "args": {
        "image": "tensor",
        "saturation_factor": "float"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.ndim"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.arctan"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.tril"] = {
    "args": {
        "m": "tensor"
    },
    "kwargs": {
        "k": "integer"
    },
    "inner": {}
}
signatures["tf.image.rgb_to_hsv"] = {
    "args": {
        "images": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.random.stateless_poisson"] = {
    "args": {
        "shape": "tensor", # Could be list or tuple of integers, but tensor is the closest
        "seed": "tensor",
        "lam": "tensor"
    },
    "kwargs": {
        "dtype": "dtype",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.sinh"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.bitwise.right_shift"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.arctanh"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.outer"] = {
    "args": {
        "a": "tensor",
        "b": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.io.gfile.exists"] = {
    "args": {
        "path": "string"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.data.experimental.DistributeOptions"] = {
    "args": {},
    "kwargs": {},
    "inner": {}
}
signatures["tf.sparse.slice"] = {
    "args": {
        "sp_input": "tensor", # SparseTensor is a tensor
        "start": "tensor",
        "size": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.gcd"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.nn.ctc_beam_search_decoder"] = {
    "args": {
        "inputs": "tensor",
        "sequence_length": "tensor" # actually it is 1-D int32 vector, so a tensor is the closest
    },
    "kwargs": {
        "beam_width": "integer",
        "top_paths": "integer"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.ascontiguousarray"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.einsum"] = {
    "args": {
        "subscripts": "string",
        "operands": "tensor_list" # operands should be a list/tuple of tensors
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.io.serialize_many_sparse"] = {
    "args": {
        "sp_input": "tensor"
    },
    "kwargs": {
        "out_type": "dtype",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.sign"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.is_strictly_increasing"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.segment_prod"] = {
    "args": {
        "data": "tensor",
        "segment_ids": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.special.bessel_j1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.image.adjust_contrast"] = {
    "args": {
        "images": "tensor",
        "contrast_factor": "float"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.linalg.lu"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "output_idx_type": "dtype",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.linalg.eigvals"] = {
    "args": {
        "tensor": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.polyval"] = {
    "args": {
        "coeffs": "list", # A list of tensors
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.special.bessel_k0"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.geomspace"] = {
    "args": {
        "start": "tensor",
        "stop": "tensor"
    },
    "kwargs": {
        "num": "integer",
        "endpoint": "boolean",
        "dtype": "dtype",
        "axis": "integer"
    },
    "inner": {}
}
signatures["tf.image.stateless_random_saturation"] = {
    "args": {
        "image": "tensor",
        "lower": "float",
        "upper": "float"
    },
    "kwargs": {
        "seed": "tensor" # Should be a tensor of int32 or int64
    },
    "inner": {}
}
signatures["tf.sort"] = {
    "args": {
        "values": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "direction": "string",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.feature_column.categorical_column_with_identity"] = {
    "args": {
        "key": "string",
        "num_buckets": "integer"
    },
    "kwargs": {
        "default_value": "integer"
    },
    "inner": {}
}
signatures["tf.nn.conv3d_transpose"] = {
    "args": {
        "input": "tensor",
        "filters": "tensor",
        "output_shape": "tensor" # Could also be a list or tuple of integers, but tensor seems more appropriate here
    },
    "kwargs": {
        "strides": "list", # It can be an int or list of ints, I'm choosing list
        "padding": "string",
        "data_format": "string",
        "dilations": "list", # It can be an int or list of ints, I'm choosing list
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.igamma"] = {
    "args": {
        "a": "tensor",
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.fingerprint"] = {
    "args": {
        "data": "tensor"
    },
    "kwargs": {
        "method": "string",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.square"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.linalg.matmul"] = {
    "args": {
        "a": "tensor",
        "b": "tensor"
    },
    "kwargs": {
        "transpose_a": "boolean",
        "transpose_b": "boolean",
        "adjoint_a": "boolean",
        "adjoint_b": "boolean",
        "a_is_sparse": "boolean",
        "b_is_sparse": "boolean",
        "output_type": "dtype",
        "grad_a": "boolean",
        "grad_b": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.nn.RNNCellDeviceWrapper"] = {
    "args": {
        "cell": "tuple" # I think it should be an RNNCell, but tuple is the closest type available
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.linalg.cholesky_solve"] = {
    "args": {
        "chol": "tensor",
        "rhs": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.nn.conv2d"] = {
    "args": {
        "input": "tensor",
        "filters": "tensor",
        "strides": "list", # could also be int, but list is more general
        "padding": "string" # can also be list, but string is simpler
    },
    "kwargs": {
        "data_format": "string",
        "dilations": "list", # could also be int, but list is more general
        "name": "string"
    },
    "inner": {}
}
signatures["tf.linalg.lu_reconstruct"] = {
    "args": {
        "lower_upper": "tensor",
        "perm": "tensor"
    },
    "kwargs": {
        "validate_args": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.reciprocal"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.image.psnr"] = {
    "args": {
        "a": "tensor",
        "b": "tensor",
        "max_val": "float" # Could be integer as well, but float is more general
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.squeeze"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer" # Could also be a tuple or a list of integers
    },
    "inner": {}
}
signatures["tf.lookup.StaticVocabularyTable"] = {
    "args": {
        "initializer": "tensor" # initializer can be None, but it can also be a KeyValueTensorInitializer or TextFileInitializer which yield tensors
    },
    "kwargs": {
        "num_oov_buckets": "integer"
    },
    "inner": {}
}
signatures["tf.math.special.bessel_j0"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.bitwise.invert"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.io.decode_base64"] = {
    "args": {
        "input": "string"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.linalg.global_norm"] = {
    "args": {
        "t_list": "tensor_list" # Could also be tuple, but tensor_list seems more appropriate
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.tan"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.data.experimental.enumerate_dataset"] = {
    "args": {},
    "kwargs": {
        "start": "tensor" # tf.int64 scalar tf.Tensor, could also be integer?
    },
    "inner": {}
}
signatures["tf.sysconfig.get_lib"] = {
    "args": {},
    "kwargs": {},
    "inner": {}
}
signatures["tf.io.decode_proto"] = {
    "args": {
        "bytes": "tensor",
        "message_type": "string",
        "field_names": "list",
        "output_types": "list"
    },
    "kwargs": {
        "descriptor_source": "string",
        "message_format": "string",
        "sanitize": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.trace"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "offset": "integer",
        "axis1": "integer",
        "axis2": "integer",
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["tf.tile"] = {
    "args": {
        "input": "tensor",
        "multiples": "tensor" # Could also be list or tuple of integers, but tensor is more general
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.strings.unicode_split_with_offsets"] = {
    "args": {
        "input": "tensor",
        "input_encoding": "string"
    },
    "kwargs": {
        "errors": "string",
        "replacement_char": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.bessel_i0e"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.sparse.cross"] = {
    "args": {
        "inputs": "tensor_list"
    },
    "kwargs": {
        "name": "string",
        "separator": "string"
    },
    "inner": {}
}
signatures["tf.math.special.expint"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.stop_gradient"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.cos"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.io.decode_json_example"] = {
    "args": {
        "json_examples": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.linalg.lu_matrix_inverse"] = {
    "args": {
        "lower_upper": "tensor",
        "perm": "tensor"
    },
    "kwargs": {
        "validate_args": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.signal.ifft3d"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.io.parse_example"] = {
    "args": {
        "serialized": "tensor",
        "features": "dict",  # A mapping is closest to a dictionary
    },
    "kwargs": {
        "example_names": "tensor", # A vector of strings is a tensor
        "name": "string"
    },
    "inner": {}
}
signatures["tf.data.experimental.map_and_batch"] = {
    "args": {
        "map_func": "list", # actually it is a function, but there is no function type
        "batch_size": "tensor"
    },
    "kwargs": {
        "num_parallel_batches": "tensor", # it's a tf.int64 scalar `tf.Tensor`
        "drop_remainder": "tensor", # it's a `tf.bool` scalar `tf.Tensor`
        "num_parallel_calls": "tensor" # it's a `tf.int32` scalar `tf.Tensor`
    },
    "inner": {}
}
signatures["tf.nn.isotonic_regression"] = {
    "args": {
        "inputs": "tensor"
    },
    "kwargs": {
        "decreasing": "boolean",
        "axis": "integer"
    },
    "inner": {}
}
signatures["tf.math.unsorted_segment_min"] = {
    "args": {
        "data": "tensor",
        "segment_ids": "tensor",
        "num_segments": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.atan"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.nn.softmax_cross_entropy_with_logits"] = {
    "args": {
        "labels": "tensor",
        "logits": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.reduce_std"] = {
    "args": {
        "input_tensor": "tensor"
    },
    "kwargs": {
        "axis": "integer", # Could also be a list/tuple of integers
        "keepdims": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.reduce_all"] = {
    "args": {
        "input_tensor": "tensor"
    },
    "kwargs": {
        "axis": "integer",  # Could also be a list or tuple of integers
        "keepdims": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.linalg.LinearOperatorToeplitz"] = {
    "args": {
        "col": "tensor",
        "row": "tensor"
    },
    "kwargs": {
        "is_non_singular": "boolean",
        "is_self_adjoint": "boolean",
        "is_positive_definite": "boolean",
        "is_square": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.isreal"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.image.rot90"] = {
    "args": {
        "image": "tensor"
    },
    "kwargs": {
        "k": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.sparse.to_indicator"] = {
    "args": {
        "sp_input": "tensor",  # SparseTensor is still a tensor
        "vocab_size": "integer"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.zeros"] = {
    "args": {
        "shape": "list" # could also be a tuple or a tensor
    },
    "kwargs": {
        "dtype": "dtype",
        "name": "string",
        "layout": "tensor" # unsure about the type here - layout seems to be related to tensors
    },
    "inner": {}
}
signatures["tf.experimental.numpy.arctan2"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.arange"] = {
    "args": {
        "start": "tensor", # Could be integer or float, but "tensor" is the closest option
    },
    "kwargs": {
        "stop": "tensor", # Could be integer or float, but "tensor" is the closest option
        "step": "tensor", # Could be integer or float, but "tensor" is the closest option
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.hypot"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.feature_column.sequence_categorical_column_with_hash_bucket"] = {
    "args": {
        "key": "string",
        "hash_bucket_size": "integer"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.mean"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer",  # Could also be a tuple or list of integers, but defaulting to integer for simplicity
        "dtype": "dtype",
        "out": "tensor",
        "keepdims": "boolean"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.rot90"] = {
    "args": {
        "m": "tensor"
    },
    "kwargs": {
        "k": "integer",
        "axes": "tuple"
    },
    "inner": {}
}
signatures["tf.random.all_candidate_sampler"] = {
    "args": {
        "true_classes": "tensor",
        "num_true": "integer",
        "num_sampled": "integer"
    },
    "kwargs": {
        "unique": "boolean",
        "seed": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.clip_by_value"] = {
    "args": {
        "t": "tensor",
        "clip_value_min": "tensor", # Can be scalar or broadcastable tensor
        "clip_value_max": "tensor"  # Can be scalar or broadcastable tensor
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.strings.to_hash_bucket"] = {
    "args": {
        "input": "tensor",
        "num_buckets": "integer"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.timestamp"] = {
    "args": {},
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.compress"] = {
    "args": {
        "condition": "tensor",
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer" # Could also be None, but integer is the closest type
    },
    "inner": {}
}
signatures["tf.io.parse_tensor"] = {
    "args": {
        "serialized": "string",
        "out_type": "dtype"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.linspace"] = {
    "args": {
        "start": "tensor",
        "stop": "tensor",
        "num": "integer"
    },
    "kwargs": {
        "name": "string",
        "axis": "integer"
    },
    "inner": {}
}
signatures["tf.tensor_scatter_nd_max"] = {
    "args": {
        "tensor": "tensor",
        "indices": "tensor",
        "updates": "tensor"
    },
    "kwargs": {
        "bad_indices_policy": "string",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.linalg.solve"] = {
    "args": {
        "matrix": "tensor",
        "rhs": "tensor"
    },
    "kwargs": {
        "adjoint": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.diagflat"] = {
    "args": {
        "v": "tensor"
    },
    "kwargs": {
        "k": "integer"
    },
    "inner": {}
}
signatures["tf.dtypes.saturate_cast"] = {
    "args": {
        "value": "tensor",
        "dtype": "dtype"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.io.gfile.walk"] = {
    "args": {
        "top": "string"
    },
    "kwargs": {
        "topdown": "boolean",
        "onerror": "list" # Should be a function
    },
    "inner": {}
}
signatures["tf.io.deserialize_many_sparse"] = {
    "args": {
        "serialized_sparse": "tensor",
        "dtype": "dtype"
    },
    "kwargs": {
        "rank": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.linalg.LinearOperatorAdjoint"] = {
    "args": {
        "operator": "tensor"  # Could also be a LinearOperator object, but representing it as a tensor
    },
    "kwargs": {
        "is_non_singular": "boolean",
        "is_self_adjoint": "boolean",
        "is_positive_definite": "boolean",
        "is_square": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.multiply_no_nan"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.print"] = {
    "args": {
        "inputs": "list" # Represents *inputs
    },
    "kwargs": {
        "output_stream": "string",
        "summarize": "integer",
        "sep": "string",
        "end": "string",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.image.non_max_suppression_with_scores"] = {
    "args": {
        "boxes": "tensor",
        "scores": "tensor",
        "max_output_size": "integer"
    },
    "kwargs": {
        "iou_threshold": "float",
        "score_threshold": "float",
        "soft_nms_sigma": "float",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.xla.experimental.compile"] = {
    "args": {
        "computation": "list"  # Assuming a Python function is best represented as a list for type purposes
    },
    "kwargs": {
        "inputs": "list"  # The input can be a list of Tensors
    },
    "inner": {}
}
signatures["tf.math.erf"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.compat.as_text"] = {
    "args": {
        "bytes_or_text": "string" # Could also be bytes
    },
    "kwargs": {
        "encoding": "string"
    },
    "inner": {}
}
signatures["tf.tensor_scatter_nd_sub"] = {
    "args": {
        "tensor": "tensor",
        "indices": "tensor",
        "updates": "tensor"
    },
    "kwargs": {
        "bad_indices_policy": "string",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.log1p"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.image.random_crop"] = {
    "args": {
        "value": "tensor",
        "size": "tensor" # Should ideally be a list or tuple of integers, but representing it as a tensor for generality
    },
    "kwargs": {
        "seed": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.sparse.map_values"] = {
    "args": {
        "op": "tensor", # op is a function, but we don't have a type for that
        "*args": "list" # Assuming *args is a list of tensors
    },
    "kwargs": {
        "**kwargs": "list" # Assuming **kwargs is a list of tensors
    },
    "inner": {}
}
signatures["tf.io.gfile.stat"] = {
    "args": {
        "path": "string"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.image.draw_bounding_boxes"] = {
    "args": {
        "images": "tensor",
        "boxes": "tensor",
        "colors": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.meshgrid"] = {
    "args": {
        "xi": "tensor_list" # Could also be tuple but doc specifies *xi which implies list
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.dynamic_stitch"] = {
    "args": {
        "indices": "list",  # Should be list of tensors, but simplifying to list
        "data": "list"   # Should be list of tensors, but simplifying to list
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.io.encode_jpeg"] = {
    "args": {
        "image": "tensor"
    },
    "kwargs": {
        "format": "string",
        "quality": "integer",
        "progressive": "boolean",
        "optimize_size": "boolean",
        "chroma_downsampling": "boolean",
        "density_unit": "string",
        "x_density": "integer",
        "y_density": "integer",
        "xmp_metadata": "string",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.linalg.set_diag"] = {
    "args": {
        "input": "tensor",
        "diagonal": "tensor"
    },
    "kwargs": {
        "name": "string",
        "k": "integer",
        "align": "string"
    },
    "inner": {}
}
signatures["tf.image.hsv_to_rgb"] = {
    "args": {
        "images": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.floordiv"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.max"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer", # Could potentially also be a tuple of integers.
        "keepdims": "boolean"
    },
    "inner": {}
}
signatures["tf.math.divide"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.imag"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.linalg.diag_part"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "name": "string",
        "k": "integer", # Could also be a tuple, see tf.linalg.diag_part_2
        "padding_value": "tensor", # The documentation suggests any numeric type, but tf tensors are used to represent those, so I'll use "tensor"
        "align": "string"
    },
    "inner": {}
}

signatures["tf.linalg.diag_part_2"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "name": "string",
        "k": "tuple",
        "padding_value": "tensor", # The documentation suggests any numeric type, but tf tensors are used to represent those, so I'll use "tensor"
        "align": "string"
    },
    "inner": {}
}
signatures["tf.nn.softsign"] = {
    "args": {
        "features": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.signal.hann_window"] = {
    "args": {
        "window_length": "tensor" # Could also be an integer. But the documentation says "A scalar `Tensor` indicating the window length". 
    },
    "kwargs": {
        "periodic": "boolean",
        "dtype": "dtype",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.data.experimental.from_variant"] = {
    "args": {
        "variant": "tensor",
        "structure": "list" # Could also be tuple or nested structure, choosing list as a general container
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.quantization.fake_quant_with_min_max_vars"] = {
    "args": {
        "inputs": "tensor",
        "min": "tensor",
        "max": "tensor"
    },
    "kwargs": {
        "num_bits": "integer",
        "narrow_range": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.empty"] = {
    "args": {
        "shape": "tuple"  # Could also be a list or integer, but tuple is most general for shape
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["tf.train.latest_checkpoint"] = {
    "args": {
        "checkpoint_dir": "string"
    },
    "kwargs": {
        "latest_filename": "string"
    },
    "inner": {}
}
signatures["tf.nn.weighted_cross_entropy_with_logits"] = {
    "args": {
        "labels": "tensor",
        "logits": "tensor",
        "pos_weight": "tensor" # Could be float, but tensor is more general
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.random.stateless_categorical"] = {
    "args": {
        "logits": "tensor",
        "num_samples": "integer",
        "seed": "tensor" # Could also be a list/tuple, but tensor seems more accurate given the description.
    },
    "kwargs": {
        "dtype": "dtype",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.unsorted_segment_prod"] = {
    "args": {
        "data": "tensor",
        "segment_ids": "tensor",
        "num_segments": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.data.experimental.service.from_dataset_id"] = {
    "args": {
        "processing_mode": "string", # Could also be ShardingPolicy enum
        "service": "tuple", # Could also be a string
        "dataset_id": "integer",
    },
    "kwargs": {
        "element_spec": "list", # Assuming nested structure of TypeSpec can be represented as a list
        "job_name": "string",
        "consumer_index": "integer",
        "num_consumers": "integer",
        "max_outstanding_requests": "integer",
        "data_transfer_protocol": "string",
        "cross_trainer_cache": "list", # Assuming a CrossTrainerCache object is a list
        "target_workers": "string"
    },
    "inner": {}
}
signatures["tf.quantization.fake_quant_with_min_max_args_gradient"] = {
    "args": {
        "gradients": "tensor",
        "inputs": "tensor"
    },
    "kwargs": {
        "min": "float",
        "max": "float",
        "num_bits": "integer",
        "narrow_range": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.around"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "decimals": "integer"
    },
    "inner": {}
}
signatures["tf.image.stateless_random_flip_left_right"] = {
    "args": {
        "image": "tensor",
        "seed": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.data.experimental.get_single_element"] = {
    "args": {
        "dataset": "tensor"  # tf.data.Dataset is a type of tensor
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.math.l2_normalize"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "axis": "integer", # Can also be a list of integers
        "epsilon": "float",
        "name": "string",
        "dim": "integer"
    },
    "inner": {}
}
signatures["tf.math.special.dawsn"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.linalg.lstsq"] = {
    "args": {
        "matrix": "tensor",
        "rhs": "tensor"
    },
    "kwargs": {
        "l2_regularizer": "float",
        "fast": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.sparse.minimum"] = {
    "args": {
        "sp_a": "tensor",
        "sp_b": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.strings.join"] = {
    "args": {
        "inputs": "tensor_list"
    },
    "kwargs": {
        "separator": "string",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.atleast_2d"] = {
    "args": {
        "arys": "tensor_list" # It can be a list of tensors
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.io.encode_png"] = {
    "args": {
        "image": "tensor"
    },
    "kwargs": {
        "compression": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.append"] = {
    "args": {
        "arr": "tensor",
        "values": "tensor"
    },
    "kwargs": {
        "axis": "integer"
    },
    "inner": {}
}
signatures["tf.linalg.tensor_diag"] = {
    "args": {
        "diagonal": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.inner"] = {
    "args": {
        "a": "tensor",
        "b": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.random.randn"] = {
    "args": {
        "args": "tuple" # Could also be list but tuple seems more appropriate for shape
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.sparse.from_dense"] = {
    "args": {
        "tensor": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.random.set_seed"] = {
    "args": {
        "seed": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.truncatediv"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.linalg.cross"] = {
    "args": {
        "a": "tensor",
        "b": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.device"] = {
    "args": {
        "device_name": "string"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.mlir.experimental.convert_function"] = {
    "args": {
        "concrete_function": "string"  # ConcreteFunction is an object, but we only have string as an option.
    },
    "kwargs": {
        "pass_pipeline": "string",
        "show_debug_info": "boolean"
    },
    "inner": {}
}
signatures["tf.math.greater"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.moveaxis"] = {
    "args": {
        "a": "tensor",
        "source": "integer",
        "destination": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.image.pad_to_bounding_box"] = {
    "args": {
        "image": "tensor",
        "offset_height": "integer",
        "offset_width": "integer",
        "target_height": "integer",
        "target_width": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.math.maximum"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.is_finite"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.ones_like"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["tf.math.reciprocal"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.nn.log_softmax"] = {
    "args": {
        "logits": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.amax"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer", # Could be a tuple of integers as well, but sticking to the simplest case
        "out": "tensor", # Should ideally be a tensor of the same type as input 'a', but using 'tensor' for simplicity
        "keepdims": "boolean" # This is also a possibility for axis to be None and axis to be integer.
    },
    "inner": {}
}
signatures["tf.data.experimental.scan"] = {
    "args": {
        "initial_state": "tensor", # Could be a nested structure of tensors, so "tensor" is the best single option
        "scan_func": "list" # A function is being passed, so list or tuple is suitable. I think tuple is more appropriate, however, the instructions mention that only very particular types are allowed. I am going to choose list arbitrarily.
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.signal.rfft3d"] = {
    "args": {
        "input_tensor": "tensor"
    },
    "kwargs": {
        "fft_length": "tensor", # int32 tensor of shape [3]
        "name": "string"
    },
    "inner": {}
}
signatures["tf.broadcast_dynamic_shape"] = {
    "args": {
        "shape_x": "tensor", # A rank 1 integer Tensor
        "shape_y": "tensor"  # A rank 1 integer Tensor
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.average"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer", # Can also be a tuple
        "weights": "tensor",
        "returned": "boolean"
    },
    "inner": {}
}
signatures["tf.io.decode_raw"] = {
    "args": {
        "input_bytes": "tensor",
        "out_type": "dtype"
    },
    "kwargs": {
        "little_endian": "boolean",
        "fixed_length": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.random.stateless_truncated_normal"] = {
    "args": {
        "shape": "tensor",
        "seed": "tensor"
    },
    "kwargs": {
        "mean": "float",
        "stddev": "float",
        "dtype": "dtype",
        "name": "string",
        "alg": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.bitwise_xor"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.nn.avg_pool2d"] = {
    "args": {
        "input": "tensor",
        "ksize": "list", # Could be an integer as well, but list is more general
        "strides": "list", # Could be an integer as well, but list is more general
    },
    "kwargs": {
        "padding": "string",
        "data_format": "string",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.convert_to_tensor"] = {
    "args": {
        "value": "list" # Could also be a tensor, but list seems more general based on the description
    },
    "kwargs": {
        "dtype": "dtype",
        "dtype_hint": "dtype",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.image.crop_to_bounding_box"] = {
    "args": {
        "image": "tensor",
        "offset_height": "integer",
        "offset_width": "integer",
        "target_height": "integer",
        "target_width": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.random.truncated_normal"] = {
    "args": {
        "shape": "tensor"
    },
    "kwargs": {
        "mean": "float",
        "stddev": "float",
        "dtype": "dtype",
        "seed": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.tensorrt.Converter"] = {
    "args": {},
    "kwargs": {
        "input_saved_model_dir": "string",
        "input_saved_model_tags": "list", # Could also be a string, but list seems more appropriate for tags
        "input_saved_model_signature_key": "string",
        "use_dynamic_shape": "boolean",
        "dynamic_shape_profile_strategy": "string", # Assuming this is an enum-like string
        "max_workspace_size_bytes": "integer",
        "precision_mode": "string", # Assuming this is an enum-like string
        "minimum_segment_size": "integer",
        "maximum_cached_engines": "integer",
        "use_calibration": "boolean",
        "allow_build_at_runtime": "boolean",
        "conversion_params": "string" # Assuming this is a class instance, so representing as string to simplify
    },
    "inner": {}
}
signatures["tf.io.serialize_sparse"] = {
    "args": {
        "sp_input": "tensor"  # Should be SparseTensor
    },
    "kwargs": {
        "out_type": "dtype",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.sparse.reduce_max"] = {
    "args": {
        "sp_input": "tensor"
    },
    "kwargs": {
        "axis": "list", # Could also be an integer
        "keepdims": "boolean",
        "output_is_sparse": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.linalg.inv"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "adjoint": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.positive"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.linalg.matvec"] = {
    "args": {
        "a": "tensor",
        "b": "tensor"
    },
    "kwargs": {
        "transpose_a": "boolean",
        "adjoint_a": "boolean",
        "a_is_sparse": "boolean",
        "b_is_sparse": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.nextafter"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.feature_column.indicator_column"] = {
    "args": {
        "categorical_column": "list" # Should it be CategoricalColumn type? Since that isn't an option, I chose list.
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.image.random_saturation"] = {
    "args": {
        "image": "tensor",
        "lower": "float",
        "upper": "float"
    },
    "kwargs": {
        "seed": "integer" # Could also be a tensor, but integer is more appropriate given context
    },
    "inner": {}
}
signatures["tf.nn.max_pool3d"] = {
    "args": {
        "input": "tensor",
        "ksize": "list", # Could also be integer
        "strides": "list", # Could also be integer
        "padding": "string"
    },
    "kwargs": {
        "data_format": "string",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.nn.conv2d_transpose"] = {
    "args": {
        "input": "tensor",
        "filters": "tensor",
        "output_shape": "tensor", # Should ideally be a "list" of integers
        "strides": "list" # Can also be an "integer"
    },
    "kwargs": {
        "padding": "string", # Can also be a "list"
        "data_format": "string",
        "dilations": "list", # Can also be an "integer"
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.sqrt"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.nn.fractional_avg_pool"] = {
    "args": {
        "value": "tensor",
        "pooling_ratio": "list"
    },
    "kwargs": {
        "pseudo_random": "boolean",
        "overlapping": "boolean",
        "seed": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.ragged.cross"] = {
    "args": {
        "inputs": "tensor_list"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.register_filesystem_plugin"] = {
    "args": {
        "plugin_location": "string"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.strings.unicode_split"] = {
    "args": {
        "input": "tensor",
        "input_encoding": "string"
    },
    "kwargs": {
        "errors": "string",
        "replacement_char": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.transpose"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "perm": "list", # or None, representing a list of integers
        "conjugate": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.signal.overlap_and_add"] = {
    "args": {
        "signal": "tensor",
        "frame_step": "integer"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.train.ClusterSpec"] = {
    "args": {
        "cluster": "dict"  # The cluster parameter accepts a dictionary.
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.math.real"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.io.decode_png"] = {
    "args": {
        "contents": "string"
    },
    "kwargs": {
        "channels": "integer",
        "dtype": "dtype",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.tanh"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.image.stateless_random_crop"] = {
    "args": {
        "value": "tensor",
        "size": "tensor",
        "seed": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.data.experimental.assert_cardinality"] = {
    "args": {
        "expected_cardinality": "integer" # Could be tf.data.UNKNOWN_CARDINALITY which is a tf.int64 scalar tensor, but integer is the closest type
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.math.asin"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.IndexedSlices"] = {
    "args": {
        "values": "tensor",
        "indices": "tensor"
    },
    "kwargs": {
        "dense_shape": "tensor" # Could potentially be a list or tuple of integers
    },
    "inner": {}
}
signatures["tf.experimental.numpy.asarray"] = {
    "args": {
        "a": "tensor" # Could also be a list or tuple, but tensor is the closest match
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.conj"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.identity_n"] = {
    "args": {
        "input": "tensor_list"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.linalg.cholesky"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.swapaxes"] = {
    "args": {
        "a": "tensor",
        "axis1": "integer",
        "axis2": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.lookup.experimental.DenseHashTable"] = {
    "args": {},
    "kwargs": {
        "key_dtype": "dtype",
        "value_dtype": "dtype",
        "default_value": "tensor",
        "empty_key": "tensor",
        "deleted_key": "tensor",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.random.set_global_generator"] = {
    "args": {
        "generator": "tensor" # Assuming that the Generator object is a tf.Variable that is a tensor
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.clip_by_global_norm"] = {
    "args": {
        "t_list": "tensor_list",
        "clip_norm": "tensor" # A 0-D (scalar) `Tensor` > 0.
    },
    "kwargs": {
        "use_norm": "tensor", # A 0-D (scalar) `Tensor` of type `float` (optional).
        "name": "string"
    },
    "inner": {}
}
signatures["tf.signal.fftshift"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "axes": "tuple", # could also be integer, but will make a separate entry for that
        "name": "string"
    },
    "inner": {}
}

signatures["tf.signal.fftshift_2"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "axes": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.cos"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.space_to_batch"] = {
    "args": {
        "input": "tensor",
        "block_shape": "tensor", # Could also be list/tuple of integers, but tensor is more general
        "paddings": "tensor" # Could also be list/tuple of lists/tuples of integers, but tensor is more general
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.ragged.stack_dynamic_partitions"] = {
    "args": {
        "data": "tensor",
        "partitions": "tensor",
        "num_partitions": "integer"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.betainc"] = {
    "args": {
        "a": "tensor",
        "b": "tensor",
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.image.image_gradients"] = {
    "args": {
        "image": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.sets.size"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "validate_indices": "boolean"
    },
    "inner": {}
}
signatures["tf.image.adjust_jpeg_quality"] = {
    "args": {
        "image": "tensor",
        "jpeg_quality": "integer"
    },
    "kwargs": {
        "dct_method": "string",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.bitwise.bitwise_and"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.image.non_max_suppression_overlaps"] = {
    "args": {
        "overlaps": "tensor",
        "scores": "tensor",
        "max_output_size": "integer"
    },
    "kwargs": {
        "overlap_threshold": "float",
        "score_threshold": "float",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.signal.fft"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.random.random"] = {
    "args": {},
    "kwargs": {
        "size": "tuple" # It could also be an integer or a list/tuple of integers defining the shape
    },
    "inner": {}
}
signatures["tf.data.experimental.rejection_resample"] = {
    "args": {
        "class_func": "function", # Function is not a valid type, but there is no other close type.
        "target_dist": "tensor"
    },
    "kwargs": {
        "initial_dist": "tensor",
        "seed": "integer"
    },
    "inner": {}
}
signatures["tf.gather"] = {
    "args": {
        "params": "tensor",
        "indices": "tensor"
    },
    "kwargs": {
        "validate_indices": "boolean", # Deprecated, so boolean is appropriate
        "axis": "integer",
        "batch_dims": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.zeros_like"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.deg2rad"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.nondifferentiable_batch_function"] = {
    "args": {
        "num_batch_threads": "integer",
        "max_batch_size": "integer",
        "batch_timeout_micros": "integer"
    },
    "kwargs": {
        "allowed_batch_sizes": "list", # Could be a list of integers, but 'list' is closest match
        "max_enqueued_batches": "integer",
        "autograph": "boolean",
        "enable_large_batch_splitting": "boolean"
    },
    "inner": {}
}
signatures["tf.no_gradient"] = {
    "args": {
        "op_type": "string"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.sqrt"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.cosh"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.logical_not"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.broadcast_static_shape"] = {
    "args": {
        "shape_x": "tensor", # TensorShape is close to tensor
        "shape_y": "tensor"  # TensorShape is close to tensor
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.nn.erosion2d"] = {
    "args": {
        "value": "tensor",
        "filters": "tensor",
        "strides": "list",
        "padding": "string",
        "data_format": "string",
        "dilations": "list"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.quantization.quantize_and_dequantize_v2"] = {
    "args": {
        "input": "tensor",
        "input_min": "tensor",
        "input_max": "tensor"
    },
    "kwargs": {
        "signed_input": "boolean",
        "num_bits": "integer",
        "range_given": "boolean",
        "round_mode": "string",
        "name": "string",
        "narrow_range": "boolean",
        "axis": "integer"
    },
    "inner": {}
}
signatures["tf.math.confusion_matrix"] = {
    "args": {
        "labels": "tensor",
        "predictions": "tensor"
    },
    "kwargs": {
        "num_classes": "integer",
        "weights": "tensor",
        "dtype": "dtype",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.image.random_brightness"] = {
    "args": {
        "image": "tensor",
        "max_delta": "float"
    },
    "kwargs": {
        "seed": "integer" # Could also be a tensor, but integer is the best match given the options
    },
    "inner": {}
}
signatures["tf.hessians"] = {
    "args": {
        "ys": "tensor_list",  # Could also be a tensor, but defaulting to list as multiple tensors are allowed.
        "xs": "tensor_list"   # Could also be a tensor, but defaulting to list as multiple tensors are allowed.
    },
    "kwargs": {
        "gate_gradients": "boolean",
        "aggregation_method": "string",  # Assuming aggregation_method accepts a string representation. Might need more specific enum type.
        "name": "string"
    },
    "inner": {}
}
signatures["tf.bitcast"] = {
    "args": {
        "input": "tensor",
        "type": "dtype"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.argmin"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "axis": "tensor", # Could also be integer, but specified as a Tensor in the documentation
        "output_type": "dtype",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.logical_and"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.sets.union"] = {
    "args": {
        "a": "tensor",
        "b": "tensor"
    },
    "kwargs": {
        "validate_indices": "boolean"
    },
    "inner": {}
}
signatures["tf.nn.nce_loss"] = {
    "args": {
        "weights": "tensor", # Could be tensor_list
        "biases": "tensor",
        "labels": "tensor",
        "inputs": "tensor",
        "num_sampled": "integer",
        "num_classes": "integer"
    },
    "kwargs": {
        "num_true": "integer",
        "sampled_values": "tuple",
        "remove_accidental_hits": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.nn.depth_to_space"] = {
    "args": {
        "input": "tensor",
        "block_size": "integer"
    },
    "kwargs": {
        "data_format": "string",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.log"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.divmod"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.linalg.LinearOperatorLowRankUpdate"] = {
    "args": {
        "base_operator": "tensor", # LinearOperator is a tensor
        "u": "tensor"
    },
    "kwargs": {
        "diag_update": "tensor",
        "v": "tensor",
        "is_diag_update_positive": "boolean",
        "is_non_singular": "boolean",
        "is_self_adjoint": "boolean",
        "is_positive_definite": "boolean",
        "is_square": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.linalg.svd"] = {
    "args": {
        "tensor": "tensor"
    },
    "kwargs": {
        "full_matrices": "boolean",
        "compute_uv": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.strings.regex_full_match"] = {
    "args": {
        "input": "tensor",
        "pattern": "string"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.random.stateless_uniform"] = {
    "args": {
        "shape": "tensor", # Should this be more specific like "list" or "tuple"?
        "seed": "tensor"
    },
    "kwargs": {
        "minval": "tensor", # Can also be Python value, so "float" or "integer" depending on dtype
        "maxval": "tensor", # Can also be Python value, so "float" or "integer" depending on dtype
        "dtype": "dtype",
        "name": "string",
        "alg": "string"
    },
    "inner": {}
}
signatures["tf.math.erfc"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.linalg.qr"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "full_matrices": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.tensordot"] = {
    "args": {
        "a": "tensor",
        "b": "tensor",
        "axes": "integer"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}

signatures["tf.tensordot_1"] = {
    "args": {
        "a": "tensor",
        "b": "tensor",
        "axes": "list" # Could also be a tensor of type int32, creating a separate signature for that case
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.nn.space_to_depth"] = {
    "args": {
        "input": "tensor",
        "block_size": "integer"
    },
    "kwargs": {
        "data_format": "string",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.data.experimental.to_variant"] = {
    "args": {
        "dataset": "tensor" # DatasetV2 is a tensor
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.math.divide_no_nan"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.digamma"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.io.encode_base64"] = {
    "args": {
        "input": "string" # A `Tensor` of type `string` is taken as string
    },
    "kwargs": {
        "pad": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.quantization.quantize"] = {
    "args": {
        "input": "tensor",
        "min_range": "tensor",
        "max_range": "tensor",
        "T": "dtype"
    },
    "kwargs": {
        "mode": "string",
        "round_mode": "string",
        "name": "string",
        "narrow_range": "boolean",
        "axis": "integer",
        "ensure_minimum_range": "float"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.cross"] = {
    "args": {
        "a": "tensor",
        "b": "tensor"
    },
    "kwargs": {
        "axisa": "integer",
        "axisb": "integer",
        "axisc": "integer",
        "axis": "integer"
    },
    "inner": {}
}
signatures["tf.image.stateless_random_flip_up_down"] = {
    "args": {
        "image": "tensor",
        "seed": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.sparse.concat"] = {
    "args": {
        "axis": "integer",
        "sp_inputs": "tensor_list"
    },
    "kwargs": {
        "expand_nonconcat_dims": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.flip"] = {
    "args": {
        "m": "tensor"
    },
    "kwargs": {
        "axis": "integer" # Could be an integer or a tuple of integers. Assuming integer first.
    },
    "inner": {}
}
signatures["tf.math.is_nan"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.ensure_shape"] = {
    "args": {
        "x": "tensor",
        "shape": "list" # Could also be tuple or TensorShape
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.is_tensor"] = {
    "args": {
        "x": "list" # Could also be a tensor, tuple, string, boolean, integer, float or dtype, but list seems most general as it can hold any object.
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.random.stateless_gamma"] = {
    "args": {
        "shape": "tensor", # Could be list/tuple of integers
        "seed": "tensor", # Could be list/tuple of integers
        "alpha": "tensor"
    },
    "kwargs": {
        "beta": "tensor",
        "dtype": "dtype",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.sysconfig.get_build_info"] = {
    "args": {},
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.tan"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.signal.irfft3d"] = {
    "args": {
        "input_tensor": "tensor"
    },
    "kwargs": {
        "fft_length": "tensor", # A `Tensor` of type `int32`.
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.bessel_i1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.data.experimental.load"] = {
    "args": {
        "path": "string"
    },
    "kwargs": {
        "element_spec": "list", # Could also be a tuple, but list is more general
        "compression": "string",
        "reader_func": "list" # Callable function
    },
    "inner": {}
}
signatures["tf.feature_column.bucketized_column"] = {
    "args": {
        "source_column": "object", # Should be more specific, like a numeric_column object. But there's no such option.
        "boundaries": "list" # Could also be a tuple, but list seems more general based on the docstring examples
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.ragged.row_splits_to_segment_ids"] = {
    "args": {
        "splits": "tensor"
    },
    "kwargs": {
        "name": "string",
        "out_type": "dtype"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.log2"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.nn.convolution"] = {
    "args": {
        "input": "tensor",
        "filters": "tensor"
    },
    "kwargs": {
        "strides": "list", # Could also be tuple? Documentation specifies "Sequence"
        "padding": "string",
        "data_format": "string",
        "dilations": "list", # Could also be tuple? Documentation specifies "Sequence"
        "name": "string"
    },
    "inner": {}
}
signatures["tf.image.resize_with_crop_or_pad"] = {
    "args": {
        "image": "tensor",
        "target_height": "integer",
        "target_width": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.ragged.map_flat_values"] = {
    "args": {
        "op": "tensor",  # This should ideally be a callable, but "tensor" is the closest type available.
        "*args": "list" # Can be a list of tensors, but list is the only option
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.sparse.segment_sum"] = {
    "args": {
        "data": "tensor",
        "indices": "tensor",
        "segment_ids": "tensor"
    },
    "kwargs": {
        "num_segments": "integer", # Can be None, but integer is the closest type
        "name": "string",
        "sparse_gradient": "boolean"
    },
    "inner": {}
}
signatures["tf.math.is_inf"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.image.random_flip_up_down"] = {
    "args": {
        "image": "tensor"
    },
    "kwargs": {
        "seed": "integer"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.lcm"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.linalg.LinearOperatorIdentity"] = {
    "args": {
        "num_rows": "integer"
    },
    "kwargs": {
        "batch_shape": "list", # Could also be a tensor of integers
        "dtype": "dtype",
        "is_non_singular": "boolean",
        "is_self_adjoint": "boolean",
        "is_positive_definite": "boolean",
        "is_square": "boolean",
        "assert_proper_shapes": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.quantization.dequantize"] = {
    "args": {
        "input": "tensor",
        "min_range": "tensor",
        "max_range": "tensor"
    },
    "kwargs": {
        "mode": "string",
        "name": "string",
        "axis": "integer",
        "narrow_range": "boolean",
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["tf.feature_column.sequence_categorical_column_with_vocabulary_file"] = {
    "args": {
        "key": "string",
        "vocabulary_file": "string"
    },
    "kwargs": {
        "vocabulary_size": "integer",
        "num_oov_buckets": "integer",
        "default_value": "integer",
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["tf.math.ndtri"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.special.bessel_y1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.polygamma"] = {
    "args": {
        "a": "tensor",
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.CriticalSection"] = {
    "args": {},
    "kwargs": {
        "name": "string",
        "shared_name": "string",
        "critical_section_def": "string", # Assuming critical_section_def is serialized string
        "import_scope": "string"
    },
    "inner": {}
}
signatures["tf.nn.atrous_conv2d_transpose"] = {
    "args": {
        "value": "tensor",
        "filters": "tensor",
        "output_shape": "tensor", # Could also be list or tuple
        "rate": "integer"
    },
    "kwargs": {
        "padding": "string",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.sparse.SparseTensor"] = {
    "args": {
        "indices": "tensor",
        "values": "tensor",
        "dense_shape": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.iscomplex"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.strings.lower"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "encoding": "string",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.linalg.eigh"] = {
    "args": {
        "tensor": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.asanyarray"] = {
    "args": {
        "a": "tensor" # Could also be a list or tuple
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.broadcast_arrays"] = {
    "args": {
        "args": "tensor_list"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.random.Generator"] = {
    "args": {},
    "kwargs": {
        "copy_from": "string", # This could also be an object of type Generator
        "state": "tensor",
        "alg": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.fabs"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.compat.as_str"] = {
    "args": {
        "bytes_or_text": "string" # Could be bytes or unicode, but string is the closest
    },
    "kwargs": {
        "encoding": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.subtract"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.executing_eagerly"] = {
    "args": {},
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.angle"] = {
    "args": {
        "z": "tensor"
    },
    "kwargs": {
        "deg": "boolean"
    },
    "inner": {}
}
signatures["tf.guarantee_const"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.compat.forward_compatible"] = {
    "args": {
        "year": "integer",
        "month": "integer",
        "day": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.sparse.fill_empty_rows"] = {
    "args": {
        "sp_input": "tensor", # SparseTensor is a type of tensor
        "default_value": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.sparse.transpose"] = {
    "args": {
        "sp_input": "tensor"
    },
    "kwargs": {
        "perm": "list", # Could also be tuple, but list seems more appropriate given the examples. Also could be tensor of type int32.
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.expand_dims"] = {
    "args": {
        "a": "tensor",
        "axis": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.train.ExponentialMovingAverage"] = {
    "args": {
        "decay": "float" # Could be tensor or variable
    },
    "kwargs": {
        "num_updates": "integer", # Could be tensor or variable
        "zero_debias": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.linalg.LinearOperatorScaledIdentity"] = {
    "args": {
        "num_rows": "integer",
        "multiplier": "tensor"
    },
    "kwargs": {
        "is_non_singular": "boolean",
        "is_self_adjoint": "boolean",
        "is_positive_definite": "boolean",
        "is_square": "boolean",
        "assert_proper_shapes": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.linalg.triangular_solve"] = {
    "args": {
        "matrix": "tensor",
        "rhs": "tensor"
    },
    "kwargs": {
        "lower": "boolean",
        "adjoint": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.compat.dimension_at_index"] = {
    "args": {
        "shape": "tensor", # TensorShape instance could be considered a tensor
        "index": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.sparse.segment_mean"] = {
    "args": {
        "data": "tensor",
        "indices": "tensor",
        "segment_ids": "tensor"
    },
    "kwargs": {
        "num_segments": "integer",
        "name": "string",
        "sparse_gradient": "boolean"
    },
    "inner": {}
}
signatures["tf.random.fixed_unigram_candidate_sampler"] = {
    "args": {
        "true_classes": "tensor",
        "num_true": "integer",
        "num_sampled": "integer",
        "unique": "boolean",
        "range_max": "integer"
    },
    "kwargs": {
        "vocab_file": "string",
        "distortion": "float",
        "num_reserved_ids": "integer",
        "num_shards": "integer",
        "shard": "integer",
        "unigrams": "list", # Could be a list of floats or integers
        "seed": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.train.CheckpointManager"] = {
    "args": {
        "checkpoint": "tensor", # Assume checkpoint is a tensor since it contains optimizer and model
        "directory": "string",
        "max_to_keep": "integer"
    },
    "kwargs": {
        "keep_checkpoint_every_n_hours": "float",
        "checkpoint_name": "string",
        "step_counter": "integer", # Could be a tensor, but assuming integer based on typical usage
        "checkpoint_interval": "integer",
        "init_fn": "tensor" # Assuming init_fn is a function that returns a tensor
    },
    "inner": {}
}
signatures["tf.experimental.numpy.nansum"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer", # Could also be a tuple of integers
        "dtype": "dtype",
        "keepdims": "boolean"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.cumprod"] = {
    "args": {
        "a": "tensor",
    },
    "kwargs": {
        "axis": "integer",  # Could also be None, but integer is the closest type
        "dtype": "dtype",
    },
    "inner": {}
}
signatures["tf.random_normal_initializer"] = {
    "args": {},
    "kwargs": {
        "mean": "float", # Could also be tensor
        "stddev": "float", # Could also be tensor
        "seed": "integer"
    },
    "inner": {}
}
signatures["tf.reshape"] = {
    "args": {
        "tensor": "tensor",
        "shape": "tensor" # Could also be list or tuple but tensor is most accurate
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.linalg.banded_triangular_solve"] = {
    "args": {
        "bands": "tensor",
        "rhs": "tensor"
    },
    "kwargs": {
        "lower": "boolean",
        "adjoint": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.log10"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.signbit"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.full_like"] = {
    "args": {
        "a": "tensor",
        "fill_value": "tensor" # could also be float or integer, assuming tensor for generality
    },
    "kwargs": {
        "dtype": "dtype",
        "order": "string",
        "subok": "boolean",
        "shape": "tuple" #could also be a list of integers or a tensor of integers
    },
    "inner": {}
}
signatures["tf.sparse.segment_sqrt_n"] = {
    "args": {
        "data": "tensor",
        "indices": "tensor",
        "segment_ids": "tensor"
    },
    "kwargs": {
        "num_segments": "integer",
        "name": "string",
        "sparse_gradient": "boolean"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.amin"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "out": "tensor",
        "keepdims": "boolean"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.iscomplexobj"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.roll"] = {
    "args": {
        "a": "tensor",
        "shift": "integer" # can also be tuple/list, but we're only creating one signature.
    },
    "kwargs": {
        "axis": "integer" # can also be tuple/list, but we're only creating one signature.
    },
    "inner": {}
}
signatures["tf.experimental.numpy.matmul"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.identity"] = {
    "args": {
        "input": "tensor" # Could also be composite tensor but simplifying to tensor
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.square"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.nn.ctc_greedy_decoder"] = {
    "args": {
        "inputs": "tensor",
        "sequence_length": "tensor" # Should be integer tensor, but using more general "tensor" type
    },
    "kwargs": {
        "merge_repeated": "boolean",
        "blank_index": "integer"
    },
    "inner": {}
}
signatures["tf.experimental.async_clear_error"] = {
    "args": {},
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.reshape"] = {
    "args": {
        "a": "tensor",
        "newshape": "tuple" # Could also be list
    },
    "kwargs": {
        "order": "string"
    },
    "inner": {}
}
signatures["tf.nn.max_pool_with_argmax"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "ksize": "list", # could be integer too, but chose list because it says list of ints.
        "strides": "list", # could be integer too, but chose list because it says list of ints.
        "padding": "string",
        "data_format": "string",
        "output_dtype": "dtype",
        "include_batch_in_index": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.signal.ifftshift"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "axes": "tuple", # Could also be an integer, so might need another signature
        "name": "string"
    },
    "inner": {}
}
signatures["tf.signal.ifftshift_2"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "axes": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.data.experimental.get_structure"] = {
    "args": {
        "dataset_or_iterator": "tuple" # Could also be tf.data.Dataset or tf.data.Iterator, choosing "tuple" as a general container type.
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.issubdtype"] = {
    "args": {
        "arg1": "dtype", # dtype or object coercible to one
        "arg2": "dtype"  # dtype or object coercible to one
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.eye"] = {
    "args": {
        "num_rows": "integer"
    },
    "kwargs": {
        "num_columns": "integer",
        "batch_shape": "list", # Could also be a tensor of integers
        "dtype": "dtype",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.signal.irfft"] = {
    "args": {
        "input_tensor": "tensor"
    },
    "kwargs": {
        "fft_length": "integer", # Tensor of type int32 and shape [1] is an integer
        "name": "string"
    },
    "inner": {}
}
signatures["tf.feature_column.categorical_column_with_vocabulary_file"] = {
    "args": {
        "key": "string",
        "vocabulary_file": "string"
    },
    "kwargs": {
        "vocabulary_size": "integer", # Could be None, but integer is the most appropriate type when it's not None
        "dtype": "dtype",
        "default_value": "integer",
        "num_oov_buckets": "integer",
        "file_format": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.greater_equal"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.image.grayscale_to_rgb"] = {
    "args": {
        "images": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.sinc"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.nn.depthwise_conv2d_backprop_filter"] = {
    "args": {
        "input": "tensor",
        "filter_sizes": "tensor",
        "out_backprop": "tensor",
        "strides": "list",
        "padding": "string" # Could also be list
    },
    "kwargs": {
        "data_format": "string",
        "dilations": "list",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.data.experimental.choose_from_datasets"] = {
    "args": {
        "datasets": "list",
        "choice_dataset": "tensor"
    },
    "kwargs": {
        "stop_on_empty_dataset": "boolean"
    },
    "inner": {}
}
signatures["tf.sparse.bincount"] = {
    "args": {
        "values": "tensor"
    },
    "kwargs": {
        "weights": "tensor",
        "axis": "integer",
        "minlength": "integer",
        "maxlength": "integer",
        "binary_output": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.data.experimental.service.WorkerServer"] = {
    "args": {
        "config": "string" # Actually it's tf.data.experimental.service.WorkerConfig, but string is the closest
    },
    "kwargs": {
        "start": "boolean"
    },
    "inner": {}
}
signatures["tf.math.round"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.clip_by_norm"] = {
    "args": {
        "t": "tensor",
        "clip_norm": "tensor" # A 0-D (scalar) Tensor > 0, but Tensor is more general
    },
    "kwargs": {
        "axes": "tensor", # A 1-D (vector) Tensor of type int32
        "name": "string"
    },
    "inner": {}
}
signatures["tf.concat"] = {
    "args": {
        "values": "tensor_list",
        "axis": "integer"  # A 0-D `int32` `Tensor` is an integer
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.log"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.io.is_jpeg"] = {
    "args": {
        "contents": "string"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.unsorted_segment_sum"] = {
    "args": {
        "data": "tensor",
        "segment_ids": "tensor",
        "num_segments": "integer" # Could be tensor too, but integer seems more appropriate from the description.
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.bessel_i0"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.compat.as_str_any"] = {
    "args": {
        "value": "string" # Could also be a more general type like "object" but string seems most appropriate given the description
    },
    "kwargs": {
        "encoding": "string"
    },
    "inner": {}
}
signatures["tf.experimental.dlpack.to_dlpack"] = {
    "args": {
        "tf_tensor": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.bitwise_and"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.math.exp"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.random.log_uniform_candidate_sampler"] = {
    "args": {
        "true_classes": "tensor",
        "num_true": "integer",
        "num_sampled": "integer",
        "unique": "boolean",
        "range_max": "integer"
    },
    "kwargs": {
        "seed": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.std"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer", # Can also be tuple or list, creating different signatures is not required based on the specs
        "keepdims": "boolean"
    },
    "inner": {}
}
signatures["tf.io.encode_proto"] = {
    "args": {
        "sizes": "tensor",
        "values": "tensor_list",
        "field_names": "list" #list of strings
    },
    "kwargs": {
        "message_type": "string",
        "descriptor_source": "string",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.realdiv"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.data.experimental.take_while"] = {
    "args": {
        "predicate": "list" # A function is being passed, but list seems like the closest type.
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.zeros"] = {
    "args": {
        "shape": "integer" # Could also be a tuple or list, but defaulting to integer as most basic.
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["tf.train.get_checkpoint_state"] = {
    "args": {
        "checkpoint_dir": "string"
    },
    "kwargs": {
        "latest_filename": "string"
    },
    "inner": {}
}
signatures["tf.random.create_rng_state_1"] = {
    "args": {
        "seed": "integer",
        "alg": "string"
    },
    "kwargs": {},
    "inner": {}
}

signatures["tf.random.create_rng_state_2"] = {
    "args": {
        "seed": "list", # Assume 1-D numpy array is represented as list
        "alg": "string"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.cast"] = {
    "args": {
        "x": "tensor",
        "dtype": "dtype"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.argmin"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.concatenate"] = {
    "args": {
        "arys": "tensor_list"
    },
    "kwargs": {
        "axis": "integer"
    },
    "inner": {}
}
signatures["tf.math.segment_max"] = {
    "args": {
        "data": "tensor",
        "segment_ids": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.nn.batch_normalization"] = {
    "args": {
        "x": "tensor",
        "mean": "tensor",
        "variance": "tensor",
        "offset": "tensor",
        "scale": "tensor"
    },
    "kwargs": {
        "variance_epsilon": "float",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.nn.compute_accidental_hits"] = {
    "args": {
        "true_classes": "tensor",
        "sampled_candidates": "tensor",
        "num_true": "integer"
    },
    "kwargs": {
        "seed": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.count_nonzero"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer" # can also be a tuple, but only integer is allowed for now
    },
    "inner": {}
}
signatures["tf.io.extract_jpeg_shape"] = {
    "args": {
        "contents": "string"
    },
    "kwargs": {
        "output_type": "dtype",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.shape"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.math.cumsum"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "axis": "integer", # Could also be a tensor of type int32
        "exclusive": "boolean",
        "reverse": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.linalg.LinearOperatorInversion"] = {
    "args": {
        "operator": "tensor"
    },
    "kwargs": {
        "is_non_singular": "boolean",
        "is_self_adjoint": "boolean",
        "is_positive_definite": "boolean",
        "is_square": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.linalg.adjoint"] = {
    "args": {
        "matrix": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.image.convert_image_dtype"] = {
    "args": {
        "image": "tensor",
        "dtype": "dtype"
    },
    "kwargs": {
        "saturate": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.image.stateless_random_contrast"] = {
    "args": {
        "image": "tensor",
        "lower": "float",
        "upper": "float",
        "seed": "tensor" # Should ideally be a tuple of integers but approximating with a tensor
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.math.count_nonzero"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "axis": "list", # Could also be an integer, but list covers the general case.
        "keepdims": "boolean",
        "dtype": "dtype",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.register_tensor_conversion_function"] = {
    "args": {
        "base_type": "tuple", # Could also be a single type, but tuple handles both
        "conversion_func": "list" # Function is not a type. List or tuple seems most appropriate, but not 100% sure
    },
    "kwargs": {
        "priority": "integer"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.fix"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.conjugate"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.nn.scale_regularization_loss"] = {
    "args": {
        "regularization_loss": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.quantization.quantized_concat"] = {
    "args": {
        "concat_dim": "integer", # Should be tensor, but integer is closest
        "values": "tensor_list",
        "input_mins": "tensor_list",
        "input_maxes": "tensor_list"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.signal.idct"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "type": "integer",
        "n": "integer", # Should ideally be None, but closest type is integer
        "axis": "integer", # Should ideally be None, but closest type is integer
        "norm": "string",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.io.decode_compressed"] = {
    "args": {
        "bytes": "string" # A `Tensor` of type `string`.
    },
    "kwargs": {
        "compression_type": "string", # An optional `string`.
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.add"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.math.unsorted_segment_mean"] = {
    "args": {
        "data": "tensor",
        "segment_ids": "tensor",
        "num_segments": "integer"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.random.learned_unigram_candidate_sampler"] = {
    "args": {
        "true_classes": "tensor",
        "num_true": "integer",
        "num_sampled": "integer",
        "unique": "boolean",
        "range_max": "integer"
    },
    "kwargs": {
        "seed": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.linalg.experimental.conjugate_gradient"] = {
    "args": {
        "operator": "tensor", # LinearOperator is a type of tensor
        "rhs": "tensor"
    },
    "kwargs": {
        "preconditioner": "tensor", # LinearOperator is a type of tensor
        "x": "tensor",
        "tol": "float",
        "max_iter": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.random.stateless_parameterized_truncated_normal"] = {
    "args": {
        "shape": "tensor", # Could also be list/tuple of integers
        "seed": "tensor",
        "means": "tensor", # Could also be float
        "stddevs": "tensor", # Could also be float
        "minvals": "tensor", # Could also be float
        "maxvals": "tensor" # Could also be float
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.zero_fraction"] = {
    "args": {
        "value": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.random.normal"] = {
    "args": {
        "shape": "tensor" # Could also be a list or tuple, but tensor is closest
    },
    "kwargs": {
        "mean": "tensor", # Could also be a float or integer
        "stddev": "tensor", # Could also be a float or integer
        "dtype": "dtype",
        "seed": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.ceil"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.nn.batch_norm_with_global_normalization"] = {
    "args": {
        "input": "tensor",
        "mean": "tensor",
        "variance": "tensor",
        "beta": "tensor",
        "gamma": "tensor",
        "variance_epsilon": "float"
    },
    "kwargs": {
        "scale_after_normalization": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.nn.moments"] = {
    "args": {
        "x": "tensor",
        "axes": "list" # Could also be a tuple of integers
    },
    "kwargs": {
        "shift": "tensor", # although documentation says "Not used", keep it as a tensor since shift is a tensor in other TF functions
        "keepdims": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.image.random_flip_left_right"] = {
    "args": {
        "image": "tensor"
    },
    "kwargs": {
        "seed": "integer"
    },
    "inner": {}
}
signatures["tf.strided_slice"] = {
    "args": {
        "input_": "tensor",
        "begin": "tensor",
        "end": "tensor"
    },
    "kwargs": {
        "strides": "tensor",
        "begin_mask": "integer",
        "end_mask": "integer",
        "ellipsis_mask": "integer",
        "new_axis_mask": "integer",
        "shrink_axis_mask": "integer",
        "var": "tensor", # It could also be None, but assuming tensor is a better fit
        "name": "string"
    },
    "inner": {}
}
signatures["tf.random.shuffle"] = {
    "args": {
        "value": "tensor"
    },
    "kwargs": {
        "seed": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.nn.dropout"] = {
    "args": {
        "x": "tensor",
        "rate": "float"
    },
    "kwargs": {
        "noise_shape": "tensor", # A 1-D integer Tensor
        "seed": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.nn.l2_loss"] = {
    "args": {
        "t": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}

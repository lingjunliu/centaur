signatures = {}
signatures["tf.raw_ops.GreaterEqual"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.image.adjust_saturation_1"] = {
    "args": {
        "image": "tensor",
        "saturation_factor": "float"  # could also be a scalar tensor
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}

signatures["tf.image.adjust_saturation_2"] = {
    "args": {
        "image": "tensor",
        "saturation_factor": "tensor"  # scalar tensor
    },
    "kwargs": {
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
        "default_value": "integer"  # Optional; None allowed, but when set must be an integer in [0, num_buckets)
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
signatures["tf.raw_ops.Empty"] = {
    "args": {
        "shape": "tensor",  # 1-D int32 Tensor representing shape
        "dtype": "dtype"
    },
    "kwargs": {
        "init": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.SparseReduceSumSparse"] = {
    "args": {
        "input_indices": "tensor",
        "input_values": "tensor",
        "input_shape": "tensor",
        "reduction_axes": "tensor"
    },
    "kwargs": {
        "keep_dims": "boolean",
        "name": "string"
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
signatures["tf.raw_ops.Relu"] = {
    "args": {
        "features": "tensor"
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
signatures["tf.get_static_value"] = {
    "args": {
        "tensor": "tensor"
    },
    "kwargs": {
        "partial": "boolean"
    },
    "inner": {}
}
signatures["tf.compat.path_to_str"] = {
    "args": {
        "path": "string"  # Also accepts os.PathLike; mapped to string
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
signatures["tf.identity"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.ComputeAccidentalHits"] = {
    "args": {
        "true_classes": "tensor",
        "sampled_candidates": "tensor",
        "num_true": "integer"
    },
    "kwargs": {
        "seed": "integer",
        "seed2": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Real"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "Tout": "dtype",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.SparseSegmentSum"] = {
    "args": {
        "data": "tensor",
        "indices": "tensor",
        "segment_ids": "tensor"
    },
    "kwargs": {
        "sparse_gradient": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.zeros_1"] = {
    "args": {
        "shape": "list"
    },
    "kwargs": {
        "dtype": "dtype",
        "name": "string",
        "layout": "tensor"  # dtensor.Layout object
    },
    "inner": {}
}
signatures["tf.zeros_2"] = {
    "args": {
        "shape": "tuple"
    },
    "kwargs": {
        "dtype": "dtype",
        "name": "string",
        "layout": "tensor"  # dtensor.Layout object
    },
    "inner": {}
}
signatures["tf.zeros_3"] = {
    "args": {
        "shape": "tensor"  # 1-D Tensor of int32
    },
    "kwargs": {
        "dtype": "dtype",
        "name": "string",
        "layout": "tensor"  # dtensor.Layout object
    },
    "inner": {}
}
signatures["tf.experimental.numpy.compress"] = {
    "args": {
        "condition": "tensor",  # boolean tensor
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer"  # can be None
    },
    "inner": {}
}
signatures["tf.image.adjust_hue"] = {
    "args": {
        "image": "tensor",
        "delta": "float"  # Could also be a scalar tensor; using float per docs
    },
    "kwargs": {
        "name": "string"
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
signatures["tf.raw_ops.FakeQuantWithMinMaxVarsPerChannel"] = {
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
signatures["tf.image.stateless_random_crop"] = {
    "args": {
        "value": "tensor",
        "size": "tensor",  # Also accepts list/tuple of ints; TF converts to Tensor
        "seed": "tensor"
    },
    "kwargs": {
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
signatures["tf.Graph"] = {
    "args": {},
    "kwargs": {},
    "inner": {}
}
signatures["tf.ones_1"] = {
    "args": {
        "shape": "list"
    },
    "kwargs": {
        "dtype": "dtype",
        "name": "string",
        "layout": "tensor"  # DTensor Layout object
    },
    "inner": {}
}
signatures["tf.ones_2"] = {
    "args": {
        "shape": "tuple"
    },
    "kwargs": {
        "dtype": "dtype",
        "name": "string",
        "layout": "tensor"  # DTensor Layout object
    },
    "inner": {}
}
signatures["tf.ones_3"] = {
    "args": {
        "shape": "tensor"  # 1-D int32 tensor
    },
    "kwargs": {
        "dtype": "dtype",
        "name": "string",
        "layout": "tensor"  # DTensor Layout object
    },
    "inner": {}
}
signatures["tf.experimental.numpy.rot90"] = {
    "args": {
        "m": "tensor"
    },
    "kwargs": {
        "k": "integer",
        "axes": "tuple"  # tuple of two integers
    },
    "inner": {}
}
signatures["tf.image.flip_left_right"] = {
    "args": {
        "image": "tensor"
    },
    "kwargs": {
        # "name": "string"  # Usually exists in TF APIs, but not described in the provided doc.
    },
    "inner": {}
}
signatures["tf.signal.rfft2d"] = {
    "args": {
        "input_tensor": "tensor"
    },
    "kwargs": {
        "fft_length": "list",  # Tensor of type int32 and shape [2] is a list of 2 integers
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
signatures["tf.math.bessel_i0e"] = {
    "args": {
        "x": "tensor"  # Tensor or SparseTensor
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.DrawBoundingBoxes"] = {
    "args": {
        "images": "tensor",
        "boxes": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.NoOp"] = {
    "args": {},
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.ControlTrigger"] = {
    "args": {},
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Abort"] = {
    "args": {},
    "kwargs": {
        "error_msg": "string",
        "exit_without_error": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.OrderedMapClear"] = {
    "args": {
        "dtypes": "list"  # list of dtype
    },
    "kwargs": {
        "capacity": "integer",
        "memory_limit": "integer",
        "container": "string",
        "shared_name": "string",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Fact"] = {
    "args": {},
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.LoopCond"] = {
    "args": {
        "input": "tensor"  # boolean scalar Tensor
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.UnicodeScript"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.CheckNumerics"] = {
    "args": {
        "tensor": "tensor",
        "message": "string"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.All"] = {
    "args": {
        "input": "tensor",
        "axis": "tensor"  # Tensor of type int32/int64 specifying reduction dims
    },
    "kwargs": {
        "keep_dims": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Max"] = {
    "args": {
        "input": "tensor",
        "axis": "tensor"  # int32/int64 tensor; conceptually dims (often passed as list/int)
    },
    "kwargs": {
        "keep_dims": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Prod"] = {
    "args": {
        "input": "tensor",
        "axis": "tensor"  # Tensor of int32/int64; may be scalar or 1-D vector
    },
    "kwargs": {
        "keep_dims": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.GuaranteeConst"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.DebugGradientIdentity"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Identity"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.InvertPermutation"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Any"] = {
    "args": {
        "input": "tensor",
        "axis": "tensor"
    },
    "kwargs": {
        "keep_dims": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.NextIteration"] = {
    "args": {
        "data": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Where"] = {
    "args": {
        "condition": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Sum"] = {
    "args": {
        "input": "tensor",
        "axis": "tensor"
    },
    "kwargs": {
        "keep_dims": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Mean"] = {
    "args": {
        "input": "tensor",
        "axis": "tensor"  # int32/int64 tensor of axes
    },
    "kwargs": {
        "keep_dims": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.PlaceholderWithDefault"] = {
    "args": {
        "input": "tensor",
        "shape": "list"  # Also accepts tf.TensorShape
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.SerializeTensor"] = {
    "args": {
        "tensor": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.IsInf"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.IsNan"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.PreventGradient"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "message": "string",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.IsFinite"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Floor"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Asinh"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.DiagPart"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Lgamma"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Softsign"] = {
    "args": {
        "features": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Cosh"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Sinh"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.DataFormatDimMap"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "src_format": "string",
        "dst_format": "string",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.DataFormatVecPermute"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "src_format": "string",
        "dst_format": "string",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Selu"] = {
    "args": {
        "features": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.SparseSliceGrad"] = {
    "args": {
        "backprop_val_grad": "tensor",
        "input_indices": "tensor",
        "input_start": "tensor",
        "output_indices": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Digamma"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.L2Loss"] = {
    "args": {
        "t": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Exp"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Log"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Elu"] = {
    "args": {
        "features": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.OnesLike"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Sin"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Softplus"] = {
    "args": {
        "features": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.InTopK"] = {
    "args": {
        "predictions": "tensor",
        "targets": "tensor",  # int32 or int64 tensor of class ids
        "k": "integer"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Merge"] = {
    "args": {
        "inputs": "tensor_list"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Erf"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Sigmoid"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Log1p"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Square"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Relu6"] = {
    "args": {
        "features": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Sign"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Bucketize"] = {
    "args": {
        "input": "tensor",
        "boundaries": "list"  # list of floats
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Sqrt"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Diag"] = {
    "args": {
        "diagonal": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Rsqrt"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.AllCandidateSampler"] = {
    "args": {
        "true_classes": "tensor",
        "num_true": "integer",
        "num_sampled": "integer",
        "unique": "boolean"
    },
    "kwargs": {
        "seed": "integer",
        "seed2": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.SegmentMin"] = {
    "args": {
        "data": "tensor",
        "segment_ids": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.SegmentMean"] = {
    "args": {
        "data": "tensor",
        "segment_ids": "tensor"  # int32 or int64 tensor
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.RandomShuffle"] = {
    "args": {
        "value": "tensor"
    },
    "kwargs": {
        "seed": "integer",
        "seed2": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.SparseSegmentSqrtN"] = {
    "args": {
        "data": "tensor",
        "indices": "tensor",
        "segment_ids": "tensor"
    },
    "kwargs": {
        "sparse_gradient": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.ComplexAbs"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "Tout": "dtype",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.SegmentSum"] = {
    "args": {
        "data": "tensor",
        "segment_ids": "tensor"  # 1-D Tensor of int32/int64
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.SegmentMax"] = {
    "args": {
        "data": "tensor",
        "segment_ids": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.SegmentProd"] = {
    "args": {
        "data": "tensor",
        "segment_ids": "tensor"  # int32 or int64 tensor
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.LogUniformCandidateSampler"] = {
    "args": {
        "true_classes": "tensor",
        "num_true": "integer",
        "num_sampled": "integer",
        "unique": "boolean",
        "range_max": "integer"
    },
    "kwargs": {
        "seed": "integer",
        "seed2": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.LearnedUnigramCandidateSampler"] = {
    "args": {
        "true_classes": "tensor",
        "num_true": "integer",
        "num_sampled": "integer",
        "unique": "boolean",
        "range_max": "integer"
    },
    "kwargs": {
        "seed": "integer",
        "seed2": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.SparseTensorDenseAdd"] = {
    "args": {
        "a_indices": "tensor",
        "a_values": "tensor",
        "a_shape": "tensor",
        "b": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.ZerosLike"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.WriteFile"] = {
    "args": {
        "filename": "tensor",
        "contents": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.SparseSegmentMean"] = {
    "args": {
        "data": "tensor",
        "indices": "tensor",
        "segment_ids": "tensor"
    },
    "kwargs": {
        "sparse_gradient": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.ShardedFilespec"] = {
    "args": {
        "basename": "tensor",
        "num_shards": "integer"  # int32 scalar tensor
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Rint"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.HSVToRGB"] = {
    "args": {
        "images": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Imag"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "Tout": "dtype",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Angle"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "Tout": "dtype",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.DeepCopy"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.SparseDenseCwiseAdd"] = {
    "args": {
        "sp_indices": "tensor",
        "sp_values": "tensor",
        "sp_shape": "tensor",
        "dense": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.EnsureShape"] = {
    "args": {
        "input": "tensor",
        "shape": "list"  # tf.TensorShape also accepted; best mapped as list of ints
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Ndtri"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.ResizeBilinear"] = {
    "args": {
        "images": "tensor",
        "size": "tensor"  # 1-D int32 Tensor of length 2; treated as tensor
    },
    "kwargs": {
        "align_corners": "boolean",
        "half_pixel_centers": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.FractionalAvgPool"] = {
    "args": {
        "value": "tensor",
        "pooling_ratio": "list"  # list of floats
    },
    "kwargs": {
        "pseudo_random": "boolean",
        "overlapping": "boolean",
        "deterministic": "boolean",
        "seed": "integer",
        "seed2": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.FractionalMaxPool"] = {
    "args": {
        "value": "tensor",
        "pooling_ratio": "list"  # list of floats
    },
    "kwargs": {
        "pseudo_random": "boolean",
        "overlapping": "boolean",
        "deterministic": "boolean",
        "seed": "integer",
        "seed2": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.SparseSlice"] = {
    "args": {
        "indices": "tensor",
        "values": "tensor",
        "shape": "tensor",
        "start": "tensor",
        "size": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Inv"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Tan"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Cross"] = {
    "args": {
        "a": "tensor",
        "b": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Round"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Reciprocal"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.InplaceAdd"] = {
    "args": {
        "x": "tensor",
        "i": "tensor",  # vector tensor of int32 indices
        "v": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.ApproximateEqual"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "tolerance": "float",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.InplaceSub"] = {
    "args": {
        "x": "tensor",
        "i": "tensor",  # int32 vector tensor of indices
        "v": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Atan"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.InplaceUpdate"] = {
    "args": {
        "x": "tensor",
        "i": "tensor",  # vector of int32 indices
        "v": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.SparseFillEmptyRowsGrad"] = {
    "args": {
        "reverse_index_map": "tensor",
        "grad_values": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.ResizeBicubic_1"] = {
    "args": {
        "images": "tensor",
        "size": "tensor"  # 1-D int32 Tensor of length 2
    },
    "kwargs": {
        "align_corners": "boolean",
        "half_pixel_centers": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.ResizeBicubic_2"] = {
    "args": {
        "images": "tensor",
        "size": "list"  # Python list of 2 ints; will be converted to int32 Tensor
    },
    "kwargs": {
        "align_corners": "boolean",
        "half_pixel_centers": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.ResizeBicubic_3"] = {
    "args": {
        "images": "tensor",
        "size": "tuple"  # Python tuple of 2 ints; will be converted to int32 Tensor
    },
    "kwargs": {
        "align_corners": "boolean",
        "half_pixel_centers": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Acos"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.SparseSoftmaxCrossEntropyWithLogits"] = {
    "args": {
        "features": "tensor",
        "labels": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.MatrixDiagPart"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.MatrixSetDiag"] = {
    "args": {
        "input": "tensor",
        "diagonal": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.SoftmaxCrossEntropyWithLogits"] = {
    "args": {
        "features": "tensor",
        "labels": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.MatrixDiag"] = {
    "args": {
        "diagonal": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.BatchToSpace"] = {
    "args": {
        "input": "tensor",
        "crops": "tensor",  # int32/int64 tensor of shape [2, 2]
        "block_size": "integer"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.LogSoftmax"] = {
    "args": {
        "logits": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.MatrixSetDiagV2_1"] = {
    "args": {
        "input": "tensor",
        "diagonal": "tensor",
        "k": "integer"  # int32 scalar tensor is treated as integer
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.MatrixSetDiagV2_2"] = {
    "args": {
        "input": "tensor",
        "diagonal": "tensor",
        "k": "tuple"  # pair of integers (akin to int32 tensor of shape [2])
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Gather"] = {
    "args": {
        "params": "tensor",
        "indices": "tensor"  # integer tensor (int32/int64)
    },
    "kwargs": {
        "validate_indices": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.RandomUniform"] = {
    "args": {
        "shape": "tensor",
        "dtype": "dtype"
    },
    "kwargs": {
        "seed": "integer",
        "seed2": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.RGBToHSV"] = {
    "args": {
        "images": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.RandomGamma"] = {
    "args": {
        "shape": "tensor",  # 1-D int32/int64 tensor
        "alpha": "tensor"   # half/float32/float64 tensor
    },
    "kwargs": {
        "seed": "integer",
        "seed2": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Betainc"] = {
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
signatures["tf.raw_ops.Complex"] = {
    "args": {
        "real": "tensor",
        "imag": "tensor"
    },
    "kwargs": {
        "Tout": "dtype",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Less"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.NextAfter"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Greater"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.LogicalOr"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.LessEqual"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Pow"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Pad"] = {
    "args": {
        "input": "tensor",
        "paddings": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Maximum"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.FloorMod"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Xlogy"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.MatMul"] = {
    "args": {
        "a": "tensor",
        "b": "tensor"
    },
    "kwargs": {
        "transpose_a": "boolean",
        "transpose_b": "boolean",
        "grad_a": "boolean",
        "grad_b": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.DivNoNan"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.MulNoNan"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Xlog1py"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Xdivy"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "name": "string"  # could be None or string; using string per spec
    },
    "inner": {}
}
signatures["tf.raw_ops.Div"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Mod"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.BatchMatMulV3"] = {
    "args": {
        "x": "tensor",
        "y": "tensor",
        "Tout": "dtype"
    },
    "kwargs": {
        "adj_x": "boolean",
        "adj_y": "boolean",
        "grad_x": "boolean",
        "grad_y": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Equal"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "incompatible_shape_error": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.AddV2"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.TruncateMod"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Igammac"] = {
    "args": {
        "a": "tensor",
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.SelectV2"] = {
    "args": {
        "condition": "tensor",
        "t": "tensor",
        "e": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.TruncateDiv"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.FloorDiv"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Add"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.SparseMatMul"] = {
    "args": {
        "a": "tensor",
        "b": "tensor"
    },
    "kwargs": {
        "transpose_a": "boolean",
        "transpose_b": "boolean",
        "a_is_sparse": "boolean",
        "b_is_sparse": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Softmax"] = {
    "args": {
        "logits": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.BatchMatMul"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "adj_x": "boolean",
        "adj_y": "boolean",
        "grad_x": "boolean",
        "grad_y": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.BatchMatMulV2"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "adj_x": "boolean",
        "adj_y": "boolean",
        "grad_x": "boolean",
        "grad_y": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.ParallelDynamicStitch"] = {
    "args": {
        "indices": "tensor_list",
        "data": "tensor_list"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.EncodePng"] = {
    "args": {
        "image": "tensor"
    },
    "kwargs": {
        "compression": "integer",  # int (0-9 or -1)
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.LRN"] = {
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
signatures["tf.raw_ops.BroadcastTo"] = {
    "args": {
        "input": "tensor",
        "shape": "tensor"  # 1-D int32/int64 tensor representing the output shape
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.MatchingFiles"] = {
    "args": {
        "pattern": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.FakeQuantWithMinMaxVarsPerChannelGradient"] = {
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
signatures["tf.raw_ops.Erfinv"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.Bitcast"] = {
    "args": {
        "input": "tensor",
        "type": "dtype"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.FakeQuantWithMinMaxArgs"] = {
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
signatures["tf.raw_ops.FakeQuantWithMinMaxArgsGradient"] = {
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
signatures["tf.raw_ops.ApproxTopK"] = {
    "args": {
        "input": "tensor",
        "k": "integer"
    },
    "kwargs": {
        "reduction_dimension": "integer",
        "recall_target": "float",
        "is_max_k": "boolean",
        "reduction_input_size_override": "integer",
        "aggregate_to_topk": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.raw_ops.ParameterizedTruncatedNormal"] = {
    "args": {
        "shape": "tensor",
        "means": "tensor",
        "stdevs": "tensor",
        "minvals": "tensor",
        "maxvals": "tensor"
    },
    "kwargs": {
        "seed": "integer",
        "seed2": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.linalg.pinv_1"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "rcond": "tensor",  # could also be a Python float
        "validate_args": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.linalg.pinv_2"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "rcond": "float",  # could also be provided as a Tensor
        "validate_args": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.image.sobel_edges"] = {
    "args": {
        "image": "tensor"
    },
    "kwargs": {},
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
signatures["tf.experimental.numpy.einsum_1"] = {
    "args": {
        "subscripts": "string",
        "operands": "tensor_list"  # varargs operands
    },
    "kwargs": {
        "out": "tensor",  # could be None; using tensor as closest
        "dtype": "dtype",
        "order": "string",
        "casting": "string",
        "optimize": "boolean"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.einsum_2"] = {
    "args": {
        "subscripts": "string",
        "operands": "tensor_list"
    },
    "kwargs": {
        "out": "tensor",
        "dtype": "dtype",
        "order": "string",
        "casting": "string",
        "optimize": "string"  # e.g., "greedy", "optimal"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.einsum_3"] = {
    "args": {
        "subscripts": "string",
        "operands": "tensor_list"
    },
    "kwargs": {
        "out": "tensor",
        "dtype": "dtype",
        "order": "string",
        "casting": "string",
        "optimize": "list"  # custom contraction path as a list
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
signatures["tf.linalg.lstsq_1"] = {
    "args": {
        "matrix": "tensor",
        "rhs": "tensor"
    },
    "kwargs": {
        "l2_regularizer": "float",  # Documented as 0-D Tensor; Python float also commonly accepted
        "fast": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.linalg.lstsq_2"] = {
    "args": {
        "matrix": "tensor",
        "rhs": "tensor"
    },
    "kwargs": {
        "l2_regularizer": "tensor",  # 0-D double Tensor
        "fast": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.nn.softmax"] = {
    "args": {
        "logits": "tensor"
    },
    "kwargs": {
        "axis": "integer",  # could be negative; docs indicate a single dimension
        "name": "string"
    },
    "inner": {}
}
signatures["tf.image.rot90"] = {
    "args": {
        "image": "tensor"
    },
    "kwargs": {
        "k": "integer",  # scalar int or int Tensor
        "name": "string"
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
signatures["tf.linalg.adjoint"] = {
    "args": {
        "matrix": "tensor"
    },
    "kwargs": {
        "name": "string"
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
signatures["tf.experimental.numpy.dot"] = {
    "args": {
        "a": "tensor",
        "b": "tensor"
    },
    "kwargs": {
        "out": "tensor"  # Can be None in NumPy; represented as tensor here
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
        "seed": "integer"  # could also be a scalar tensor; treated as integer here
    },
    "inner": {}
}
signatures["tf.nn.local_response_normalization"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "depth_radius": "integer",  # could also be a 0-D int32 tensor in TF
        "bias": "float",
        "alpha": "float",
        "beta": "float",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.image.rgb_to_yiq"] = {
    "args": {
        "images": "tensor"
    },
    "kwargs": {
        "name": "string"  # optional op name
    },
    "inner": {}
}
signatures["tf.image.yiq_to_rgb"] = {
    "args": {
        "images": "tensor"
    },
    "kwargs": {
        "name": "string"  # optional op name; typical in TF though not shown explicitly in snippet
    },
    "inner": {}
}
signatures["tf.image.yuv_to_rgb"] = {
    "args": {
        "images": "tensor"
    },
    "kwargs": {
        "name": "string"  # optional
    },
    "inner": {}
}
signatures["tf.experimental.numpy.moveaxis_1"] = {
    "args": {
        "a": "tensor",
        "source": "integer",
        "destination": "integer"
    },
    "kwargs": {
    },
    "inner": {}
}
signatures["tf.experimental.numpy.moveaxis_2"] = {
    "args": {
        "a": "tensor",
        "source": "list",  # sequence of ints
        "destination": "list"  # sequence of ints
    },
    "kwargs": {
    },
    "inner": {}
}
signatures["tf.experimental.numpy.moveaxis_3"] = {
    "args": {
        "a": "tensor",
        "source": "tuple",  # sequence of ints
        "destination": "tuple"  # sequence of ints
    },
    "kwargs": {
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
signatures["tf.experimental.numpy.vdot"] = {
    "args": {
        "a": "tensor",
        "b": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.ragged.boolean_mask"] = {
    "args": {
        "data": "tensor",
        "mask": "tensor"  # boolean tensor, but still a tensor
    },
    "kwargs": {
        "name": "string"
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
signatures["tf.nn.collapse_repeated"] = {
    "args": {
        "labels": "tensor",
        "seq_length": "tensor"  # int32/64 tensor of shape [batch]
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.hstack"] = {
    "args": {
        "tup": "tensor_list"  # sequence (list/tuple) of tensors
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.math.unsorted_segment_mean"] = {
    "args": {
        "data": "tensor",
        "segment_ids": "tensor",
        "num_segments": "integer"  # Can be Python int or scalar int Tensor
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.reduce_variance_1"] = {
    "args": {
        "input_tensor": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "keepdims": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.reduce_variance_2"] = {
    "args": {
        "input_tensor": "tensor"
    },
    "kwargs": {
        "axis": "list",
        "keepdims": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.reduce_variance_3"] = {
    "args": {
        "input_tensor": "tensor"
    },
    "kwargs": {
        "axis": "tuple",
        "keepdims": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.reduce_variance_4"] = {
    "args": {
        "input_tensor": "tensor"
    },
    "kwargs": {
        "axis": "tensor",  # int32/int64 tensor for axes
        "keepdims": "boolean",
        "name": "string"
    },
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
signatures["tf.image.per_image_standardization"] = {
    "args": {
        "image": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.image.random_flip_left_right"] = {
    "args": {
        "image": "tensor"
    },
    "kwargs": {
        "seed": "integer"  # Python int seed
    },
    "inner": {}
}
signatures["tf.image.total_variation"] = {
    "args": {
        "images": "tensor"
    },
    "kwargs": {
        "name": "string"  # Optional op name
    },
    "inner": {}
}
signatures["tf.experimental.numpy.equal"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {
        # NumPy allows array-like (e.g., lists), but in TF this will be converted to tensors.
    },
    "inner": {}
}
signatures["tf.experimental.numpy.nanmean_1"] = {
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
signatures["tf.experimental.numpy.nanmean_2"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "tuple",  # tuple of integers
        "dtype": "dtype",
        "keepdims": "boolean"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.nanprod_1"] = {
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
signatures["tf.experimental.numpy.nanprod_2"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "tuple",  # tuple of integers
        "dtype": "dtype",
        "keepdims": "boolean"
    },
    "inner": {}
}
signatures["tf.io.encode_png"] = {
    "args": {
        "image": "tensor"  # uint8 or uint16 tensor
    },
    "kwargs": {
        "compression": "integer",
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
signatures["tf.image.image_gradients"] = {
    "args": {
        "image": "tensor"
    },
    "kwargs": {
        # "name": "string"  # Likely exists in TF but not documented above, so omitted per instructions
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
signatures["tf.image.grayscale_to_rgb"] = {
    "args": {
        "images": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.atleast_2d"] = {
    "args": {
        "arys": "tensor_list"  # varargs of array-like; lists/tuples also accepted but treated as tensors
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.image.stateless_random_flip_up_down_1"] = {
    "args": {
        "image": "tensor",
        "seed": "tensor"  # shape [2] int32/int64 tensor
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.image.stateless_random_flip_up_down_2"] = {
    "args": {
        "image": "tensor",
        "seed": "tuple"  # length-2 tuple of ints, converted to tensor
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.image.stateless_random_flip_up_down_3"] = {
    "args": {
        "image": "tensor",
        "seed": "list"  # length-2 list of ints, converted to tensor
    },
    "kwargs": {},
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
signatures["tf.linalg.eig"] = {
    "args": {
        "tensor": "tensor"
    },
    "kwargs": {
        "name": "string"
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
signatures["tf.experimental.numpy.triu"] = {
    "args": {
        "m": "tensor"
    },
    "kwargs": {
        "k": "integer"
    },
    "inner": {}
}
signatures["tf.math.truediv"] = {
    "args": {
        "x": "tensor",  # may also accept Python numbers; treated as tensors
        "y": "tensor"   # may also accept Python numbers; treated as tensors
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.append_1"] = {
    "args": {
        "arr": "tensor",  # accepts array-like; mapped to tensor
        "values": "tensor"  # accepts array-like; mapped to tensor
    },
    "kwargs": {
        "axis": "integer"  # None also allowed; omitted in _2
    },
    "inner": {}
}
signatures["tf.experimental.numpy.append_2"] = {
    "args": {
        "arr": "tensor",  # accepts array-like; mapped to tensor
        "values": "tensor"  # accepts array-like; mapped to tensor
    },
    "kwargs": {
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
signatures["tf.math.reduce_std_1"] = {
    "args": {
        "input_tensor": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "keepdims": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.reduce_std_2"] = {
    "args": {
        "input_tensor": "tensor"
    },
    "kwargs": {
        "axis": "list",  # list of integers
        "keepdims": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.reduce_std_3"] = {
    "args": {
        "input_tensor": "tensor"
    },
    "kwargs": {
        "axis": "tuple",  # tuple of integers
        "keepdims": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.image.stateless_random_flip_left_right"] = {
    "args": {
        "image": "tensor",
        "seed": "tensor"  # accepts shape [2] int tensor; tuples like (2,3) are also commonly accepted
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.bitwise_or"] = {
    "args": {
        "x1": "tensor",  # array-like; treated as tensor
        "x2": "tensor"   # array-like; treated as tensor
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.dstack"] = {
    "args": {
        "tup": "tensor_list"  # sequence (list/tuple) of array-like; choosing tensor_list
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.nansum_1"] = {
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
signatures["tf.experimental.numpy.nansum_2"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "tuple",  # tuple of integers
        "dtype": "dtype",
        "keepdims": "boolean"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.nansum_3"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "list",  # list of integers
        "dtype": "dtype",
        "keepdims": "boolean"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.allclose"] = {
    "args": {
        "a": "tensor",
        "b": "tensor"
    },
    "kwargs": {
        "rtol": "float",  # could also accept tensor-like scalars
        "atol": "float",  # could also accept tensor-like scalars
        "equal_nan": "boolean"
    },
    "inner": {}
}
signatures["tf.image.random_flip_up_down"] = {
    "args": {
        "image": "tensor"
    },
    "kwargs": {
        "seed": "integer"  # Python integer; TF may also accept int-like tensors, but docs specify Python int
    },
    "inner": {}
}
signatures["tf.experimental.numpy.diag"] = {
    "args": {
        "v": "tensor"  # Could also accept array-like; treating as tensor
    },
    "kwargs": {
        "k": "integer"
    },
    "inner": {}
}
signatures["tf.nn.moments_1"] = {
    "args": {
        "x": "tensor",
        "axes": "integer"  # axes can be a single int
    },
    "kwargs": {
        "shift": "tensor",  # Not used; historically same dtype as x
        "keepdims": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.nn.moments_2"] = {
    "args": {
        "x": "tensor",
        "axes": "list"  # array of ints
    },
    "kwargs": {
        "shift": "tensor",  # Not used; historically same dtype as x
        "keepdims": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.nn.moments_3"] = {
    "args": {
        "x": "tensor",
        "axes": "tuple"  # array of ints
    },
    "kwargs": {
        "shift": "tensor",  # Not used; historically same dtype as x
        "keepdims": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.std_1"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "keepdims": "boolean"  # In TF docs default may be None; treated as boolean flag
    },
    "inner": {}
}
signatures["tf.experimental.numpy.std_2"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "tuple",
        "keepdims": "boolean"  # In TF docs default may be None; treated as boolean flag
    },
    "inner": {}
}
signatures["tf.experimental.numpy.std_3"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "list",
        "keepdims": "boolean"  # In TF docs default may be None; treated as boolean flag
    },
    "inner": {}
}
signatures["tf.math.reduce_logsumexp_1"] = {
    "args": {
        "input_tensor": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "keepdims": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.reduce_logsumexp_2"] = {
    "args": {
        "input_tensor": "tensor"
    },
    "kwargs": {
        "axis": "list",  # list of integers
        "keepdims": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.reduce_logsumexp_3"] = {
    "args": {
        "input_tensor": "tensor"
    },
    "kwargs": {
        "axis": "tuple",  # tuple of integers
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
signatures["tf.signal.irfft2d"] = {
    "args": {
        "input_tensor": "tensor"
    },
    "kwargs": {
        "fft_length": "list",  # int32 Tensor of shape [2]; using list to represent 2D lengths
        "name": "string"
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
signatures["tf.linalg.lu_solve"] = {
    "args": {
        "lower_upper": "tensor",
        "perm": "tensor",  # int Tensor (permutation indices) but still a Tensor
        "rhs": "tensor"
    },
    "kwargs": {
        "validate_args": "boolean",
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
signatures["tf.linalg.trace"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.confusion_matrix_1"] = {
    "args": {
        "labels": "tensor",
        "predictions": "tensor"
    },
    "kwargs": {
        "num_classes": "integer",  # could also be a tensor
        "weights": "tensor",
        "dtype": "dtype",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.confusion_matrix_2"] = {
    "args": {
        "labels": "tensor",
        "predictions": "tensor"
    },
    "kwargs": {
        "num_classes": "tensor",
        "weights": "tensor",
        "dtype": "dtype",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.reduce_min_1"] = {
    "args": {
        "input_tensor": "tensor"
    },
    "kwargs": {
        "axis": "integer",  # could also be an int tensor, but using integer per spec
        "keepdims": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.reduce_min_2"] = {
    "args": {
        "input_tensor": "tensor"
    },
    "kwargs": {
        "axis": "list",  # list of integers
        "keepdims": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.reduce_min_3"] = {
    "args": {
        "input_tensor": "tensor"
    },
    "kwargs": {
        "axis": "tuple",  # tuple of integers
        "keepdims": "boolean",
        "name": "string"
    },
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
signatures["tf.image.adjust_contrast"] = {
    "args": {
        "images": "tensor",
        "contrast_factor": "float"  # Could also accept a tensor-like float in TF
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.linalg.matrix_rank_1"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "tol": "tensor",
        "validate_args": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.linalg.matrix_rank_2"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "tol": "float",  # could also accept int, but best treated as float
        "validate_args": "boolean",
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
signatures["tf.image.random_contrast"] = {
    "args": {
        "image": "tensor",
        "lower": "float",
        "upper": "float"
    },
    "kwargs": {
        "seed": "integer"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.tril"] = {
    "args": {
        "m": "tensor"  # could accept array-like; using tensor as closest match
    },
    "kwargs": {
        "k": "integer"
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
signatures["tf.image.psnr_1"] = {
    "args": {
        "a": "tensor",
        "b": "tensor",
        "max_val": "float"  # also accepts integer / scalar tensor
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.image.psnr_2"] = {
    "args": {
        "a": "tensor",
        "b": "tensor",
        "max_val": "integer"  # also accepts float / scalar tensor
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.identity"] = {
    "args": {
        "n": "integer"
    },
    "kwargs": {
        "dtype": "dtype"  # could also accept string dtype names in practice
    },
    "inner": {}
}
signatures["tf.image.crop_to_bounding_box"] = {
    "args": {
        "image": "tensor",
        "offset_height": "integer",  # accepts Python int or 0-D int32 tensor
        "offset_width": "integer",   # accepts Python int or 0-D int32 tensor
        "target_height": "integer",  # accepts Python int or 0-D int32 tensor
        "target_width": "integer"    # accepts Python int or 0-D int32 tensor
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.math.reduce_euclidean_norm_1"] = {
    "args": {
        "input_tensor": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "keepdims": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.reduce_euclidean_norm_2"] = {
    "args": {
        "input_tensor": "tensor"
    },
    "kwargs": {
        "axis": "list",
        "keepdims": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.reduce_euclidean_norm_3"] = {
    "args": {
        "input_tensor": "tensor"
    },
    "kwargs": {
        "axis": "tuple",
        "keepdims": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.reduce_euclidean_norm_4"] = {
    "args": {
        "input_tensor": "tensor"
    },
    "kwargs": {
        "axis": "tensor",  # axis can be an int32/int64 Tensor
        "keepdims": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.random.stateless_gamma_1"] = {
    "args": {
        "shape": "tensor",  # 1-D int tensor
        "seed": "tensor",   # shape [2] int tensor
        "alpha": "tensor"
    },
    "kwargs": {
        "beta": "tensor",   # could also be list
        "dtype": "dtype",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.random.stateless_gamma_2"] = {
    "args": {
        "shape": "list",  # Python list of ints
        "seed": "list",   # Python list of 2 ints
        "alpha": "list"
    },
    "kwargs": {
        "beta": "list",
        "dtype": "dtype",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.image.random_crop_1"] = {
    "args": {
        "value": "tensor",
        "size": "tensor"  # 1-D int tensor; Python list/tuple also accepted in other signatures
    },
    "kwargs": {
        "seed": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.image.random_crop_2"] = {
    "args": {
        "value": "tensor",
        "size": "list"
    },
    "kwargs": {
        "seed": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.image.random_crop_3"] = {
    "args": {
        "value": "tensor",
        "size": "tuple"
    },
    "kwargs": {
        "seed": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.broadcast_arrays"] = {
    "args": {
        "arrays": "tensor_list"  # Represents variadic *args of array-like/tensors
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.nn.dropout_1"] = {
    "args": {
        "x": "tensor",
        "rate": "tensor"  # Also accepts Python float; see _2, _5, _6
    },
    "kwargs": {
        "noise_shape": "tensor",  # Also accepts list/tuple; see _2 and _3
        "seed": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.nn.dropout_2"] = {
    "args": {
        "x": "tensor",
        "rate": "tensor"
    },
    "kwargs": {
        "noise_shape": "list",  # e.g., [1,10]
        "seed": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.nn.dropout_3"] = {
    "args": {
        "x": "tensor",
        "rate": "tensor"
    },
    "kwargs": {
        "noise_shape": "tuple",
        "seed": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.nn.dropout_4"] = {
    "args": {
        "x": "tensor",
        "rate": "float"  # Python float scalar is accepted
    },
    "kwargs": {
        "noise_shape": "tensor",
        "seed": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.nn.dropout_5"] = {
    "args": {
        "x": "tensor",
        "rate": "float"
    },
    "kwargs": {
        "noise_shape": "list",
        "seed": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.nn.dropout_6"] = {
    "args": {
        "x": "tensor",
        "rate": "float"
    },
    "kwargs": {
        "noise_shape": "tuple",
        "seed": "integer",
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
signatures["tf.experimental.numpy.outer"] = {
    "args": {
        "a": "tensor",
        "b": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.signal.fftshift_1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "axes": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.signal.fftshift_2"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "axes": "tuple",
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
signatures["tf.experimental.numpy.array_equal"] = {
    "args": {
        "a1": "tensor",
        "a2": "tensor"
    },
    "kwargs": {
    },
    "inner": {}
}
signatures["tf.experimental.numpy.isposinf"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
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
signatures["tf.experimental.numpy.argmax"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer"  # Optional; None means omitted
    },
    "inner": {}
}
signatures["tf.reverse_1"] = {
    "args": {
        "tensor": "tensor",
        "axis": "tensor"  # 1-D int tensor; TF also accepts Python sequences
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.reverse_2"] = {
    "args": {
        "tensor": "tensor",
        "axis": "list"  # list of integers
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.reverse_3"] = {
    "args": {
        "tensor": "tensor",
        "axis": "tuple"  # tuple of integers
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.is_non_decreasing"] = {
    "args": {
        "x": "tensor"  # TensorLike accepted in practice, but docs say Numeric Tensor
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.ix_"] = {
    "args": {
        "args": "tensor_list"  # variadic 1-D index arrays; Python lists are typically converted to tensors
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.cumprod"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer",  # could be None when omitted
        "dtype": "dtype"
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
signatures["tf.experimental.numpy.floor_divide"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {},
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
signatures["tf.experimental.numpy.ravel_1"] = {
    "args": {
        "a": "tensor"  # could also accept Python scalars, treated as tensor
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.ravel_2"] = {
    "args": {
        "a": "list"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.ravel_3"] = {
    "args": {
        "a": "tuple"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.histogram_fixed_width_bins_1"] = {
    "args": {
        "values": "tensor",
        "value_range": "tensor"
    },
    "kwargs": {
        "nbins": "integer",  # Scalar int32 Tensor or Python int treated as integer
        "dtype": "dtype",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.histogram_fixed_width_bins_2"] = {
    "args": {
        "values": "tensor",
        "value_range": "list"  # Typically a length-2 list of numbers
    },
    "kwargs": {
        "nbins": "integer",
        "dtype": "dtype",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.histogram_fixed_width_bins_3"] = {
    "args": {
        "values": "tensor",
        "value_range": "tuple"  # Typically a length-2 tuple of numbers
    },
    "kwargs": {
        "nbins": "integer",
        "dtype": "dtype",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.expand_dims"] = {
    "args": {
        "input": "tensor",
        "axis": "integer"  # Could also be a scalar int tensor; using "integer"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.arctan2"] = {
    "args": {
        "x1": "tensor",  # array-like or scalar; treated as tensor
        "x2": "tensor"   # array-like or scalar; treated as tensor
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.ascontiguousarray_1"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.ascontiguousarray_2"] = {
    "args": {
        "a": "list"  # also accepts array-like
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.ascontiguousarray_3"] = {
    "args": {
        "a": "tuple"  # also accepts array-like
    },
    "kwargs": {
        "dtype": "dtype"
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
signatures["tf.experimental.numpy.expand_dims_1"] = {
    "args": {
        "a": "tensor",
        "axis": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.expand_dims_2"] = {
    "args": {
        "a": "tensor",
        "axis": "tuple"  # tuple of integers
    },
    "kwargs": {},
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
signatures["tf.nn.ctc_beam_search_decoder"] = {
    "args": {
        "inputs": "tensor",
        "sequence_length": "tensor"
    },
    "kwargs": {
        "beam_width": "integer",
        "top_paths": "integer"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.atleast_1d"] = {
    "args": {
        "arys": "tensor_list"  # could include array-like/scalars; best mapped to tensor_list
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
signatures["tf.sparse.eye_1"] = {
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
signatures["tf.sparse.eye_2"] = {
    "args": {
        "num_rows": "integer"
    },
    "kwargs": {
        "num_columns": "tensor",  # int32 scalar tensor
        "dtype": "dtype",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.sparse.eye_3"] = {
    "args": {
        "num_rows": "tensor"  # int32 scalar tensor
    },
    "kwargs": {
        "num_columns": "integer",
        "dtype": "dtype",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.sparse.eye_4"] = {
    "args": {
        "num_rows": "tensor"  # int32 scalar tensor
    },
    "kwargs": {
        "num_columns": "tensor",  # int32 scalar tensor
        "dtype": "dtype",
        "name": "string"
    },
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
signatures["tf.experimental.numpy.fix"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.math.reduce_all_1"] = {
    "args": {
        "input_tensor": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "keepdims": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.reduce_all_2"] = {
    "args": {
        "input_tensor": "tensor"
    },
    "kwargs": {
        "axis": "list",
        "keepdims": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.reduce_all_3"] = {
    "args": {
        "input_tensor": "tensor"
    },
    "kwargs": {
        "axis": "tuple",
        "keepdims": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.reduce_all_4"] = {
    "args": {
        "input_tensor": "tensor"
    },
    "kwargs": {
        "axis": "tensor",  # Could be an int32/int64 Tensor for dynamic axes
        "keepdims": "boolean",
        "name": "string"
    },
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
signatures["tf.math.zero_fraction"] = {
    "args": {
        "value": "tensor"
    },
    "kwargs": {
        "name": "string"
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
signatures["tf.unravel_index_1"] = {
    "args": {
        "indices": "tensor",
        "dims": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.unravel_index_2"] = {
    "args": {
        "indices": "tensor",
        "dims": "list"  # Also accepts Python list for dims
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.unravel_index_3"] = {
    "args": {
        "indices": "list",  # Also accepts Python list for indices
        "dims": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.unravel_index_4"] = {
    "args": {
        "indices": "list",  # Also accepts Python list for indices
        "dims": "list"      # Also accepts Python list for dims
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
    "kwargs": {
        # NumPy's out/where/etc. are unsupported; TF variant doesn't expose extra kwargs
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
signatures["tf.math.log1p"] = {
    "args": {
        "x": "tensor"
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
signatures["tf.is_tensor"] = {
    "args": {
        "x": "tensor"  # accepts any Python object; using "tensor" as closest match
    },
    "kwargs": {},
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
signatures["tf.experimental.numpy.flip_1"] = {
    "args": {
        "m": "tensor"
    },
    "kwargs": {
        "axis": "integer"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.flip_2"] = {
    "args": {
        "m": "tensor"
    },
    "kwargs": {
        "axis": "tuple"  # tuple of integers
    },
    "inner": {}
}
signatures["tf.math.cumsum_1"] = {
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
signatures["tf.math.cumsum_2"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "axis": "tensor",  # Tensor of type int32 is also accepted
        "exclusive": "boolean",
        "reverse": "boolean",
        "name": "string"
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
signatures["tf.experimental.numpy.empty_like"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "dtype": "dtype"  # could also accept dtype as string in some TF APIs, but using dtype here
    },
    "inner": {}
}
signatures["tf.experimental.numpy.shape"] = {
    "args": {
        "a": "tensor"  # Accepts array-like (e.g., list/tuple) but treated as tensor
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
signatures["tf.signal.ifft2d"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
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
signatures["tf.where_1"] = {
    "args": {
        "condition": "tensor"  # accepts TensorLike; dtype can be bool or numeric in this mode
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.where_2"] = {
    "args": {
        "condition": "tensor",  # accepts TensorLike; must be bool in this mode
        "x": "tensor",          # accepts TensorLike
        "y": "tensor"           # accepts TensorLike
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.nextafter"] = {
    "args": {
        "x1": "tensor",  # float32/float64 tensor
        "x2": "tensor"   # same dtype as x1
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.nest.flatten_1"] = {
    "args": {
        "structure": "list"  # Also supports dict/nested structures; "dict" not in allowed types
    },
    "kwargs": {
        "expand_composites": "boolean"
    },
    "inner": {}
}
signatures["tf.nest.flatten_2"] = {
    "args": {
        "structure": "tuple"  # Also supports dict/nested structures; "dict" not in allowed types
    },
    "kwargs": {
        "expand_composites": "boolean"
    },
    "inner": {}
}
signatures["tf.nest.flatten_3"] = {
    "args": {
        "structure": "tensor"  # Atom case (e.g., tf.Tensor, numpy array treated as atom)
    },
    "kwargs": {
        "expand_composites": "boolean"
    },
    "inner": {}
}
signatures["tf.concat_1"] = {
    "args": {
        "values": "tensor_list",
        "axis": "integer"  # also accepts a 0-D int32 Tensor
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}

signatures["tf.concat_2"] = {
    "args": {
        "values": "tensor",
        "axis": "integer"  # also accepts a 0-D int32 Tensor
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.IndexedSlices_1"] = {
    "args": {
        "values": "tensor",
        "indices": "tensor"  # 1-D integer Tensor
    },
    "kwargs": {
        "dense_shape": "tensor"  # 1-D integer Tensor or TensorShape; best fit: tensor
    },
    "inner": {}
}
signatures["tf.IndexedSlices_2"] = {
    "args": {
        "values": "tensor",
        "indices": "tensor"  # 1-D integer Tensor
    },
    "kwargs": {
        "dense_shape": "list"  # Could also be TensorShape; using list for shape-like input
    },
    "inner": {}
}
signatures["tf.IndexedSlices_3"] = {
    "args": {
        "values": "tensor",
        "indices": "tensor"  # 1-D integer Tensor
    },
    "kwargs": {
        "dense_shape": "tuple"  # Could also be TensorShape; using tuple for shape-like input
    },
    "inner": {}
}
signatures["tf.no_op"] = {
    "args": {},
    "kwargs": {
        "name": "string"  # optional
    },
    "inner": {}
}
signatures["tf.experimental.numpy.prod_1"] = {
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
signatures["tf.experimental.numpy.prod_2"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "tuple",
        "dtype": "dtype",
        "keepdims": "boolean"
    },
    "inner": {}
}
signatures["tf.nest.pack_sequence_as_1"] = {
    "args": {
        "structure": "list",  # could also be a dict, but "dict" type not allowed by spec
        "flat_sequence": "list"
    },
    "kwargs": {
        "expand_composites": "boolean"
    },
    "inner": {}
}
signatures["tf.nest.pack_sequence_as_2"] = {
    "args": {
        "structure": "tuple",  # could also be a dict, but "dict" type not allowed by spec
        "flat_sequence": "list"
    },
    "kwargs": {
        "expand_composites": "boolean"
    },
    "inner": {}
}
signatures["tf.nn.sparse_softmax_cross_entropy_with_logits"] = {
    "args": {
        "labels": "tensor",
        "logits": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.all_1"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "keepdims": "boolean"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.all_2"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "tuple",  # axis may also accept list of ints in TF; using tuple per NumPy docs
        "keepdims": "boolean"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.count_nonzero_1"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.count_nonzero_2"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "tuple"  # tuple of integers
    },
    "inner": {}
}
signatures["tf.experimental.numpy.count_nonzero_3"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "list"  # list of integers
    },
    "inner": {}
}
signatures["tf.random.set_global_generator"] = {
    "args": {
        "generator": "tensor"  # should be a tf.random.Generator object
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.sets.intersection"] = {
    "args": {
        "a": "tensor",  # Can also be a SparseTensor
        "b": "tensor"   # Can also be a SparseTensor
    },
    "kwargs": {
        "validate_indices": "boolean"
    },
    "inner": {}
}
signatures["tf.math.sqrt"] = {
    "args": {
        "x": "tensor"  # Can also be a SparseTensor, but using "tensor" per allowed types
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.nn.weighted_cross_entropy_with_logits_1"] = {
    "args": {
        "labels": "tensor",
        "logits": "tensor",
        "pos_weight": "tensor"  # typically scalar tensor; also accepts Python float
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.nn.weighted_cross_entropy_with_logits_2"] = {
    "args": {
        "labels": "tensor",
        "logits": "tensor",
        "pos_weight": "float"  # also accepts tensor
    },
    "kwargs": {
        "name": "string"
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
signatures["tf.image.random_hue"] = {
    "args": {
        "image": "tensor",
        "max_delta": "float"
    },
    "kwargs": {
        "seed": "integer"  # could also accept an int tensor scalar in TF
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
signatures["tf.experimental.numpy.heaviside"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.sum_1"] = {
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
signatures["tf.experimental.numpy.sum_2"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "tuple",
        "dtype": "dtype",
        "keepdims": "boolean"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.sum_3"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "list",  # could also be a tuple of ints; covered in _2
        "dtype": "dtype",
        "keepdims": "boolean"
    },
    "inner": {}
}
signatures["tf.math.top_k_1"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "k": "integer",  # Can also be a 0-D int tensor
        "sorted": "boolean",
        "index_type": "dtype",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.top_k_2"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "k": "tensor",  # 0-D integer tensor
        "sorted": "boolean",
        "index_type": "dtype",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.image.stateless_random_hue_1"] = {
    "args": {
        "image": "tensor",
        "max_delta": "float",  # scalar float; TF may also accept a float tensor, but docs specify float
        "seed": "tensor"  # shape [2] int32/int64 tensor
    },
    "kwargs": {},
    "inner": {}
}

signatures["tf.image.stateless_random_hue_2"] = {
    "args": {
        "image": "tensor",
        "max_delta": "float",
        "seed": "tuple"  # commonly a Python tuple of two ints (e.g., (1, 2))
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.isscalar_1"] = {
    "args": {
        "num": "tensor"  # Could also be Python scalars; best-fit type chosen
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.isscalar_2"] = {
    "args": {
        "num": "integer"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.isscalar_3"] = {
    "args": {
        "num": "float"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.isscalar_4"] = {
    "args": {
        "num": "boolean"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.isscalar_5"] = {
    "args": {
        "num": "string"  # Accepts any Python object; using closest allowed type
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.iscomplexobj"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.atleast_3d"] = {
    "args": {
        "arys": "tensor_list"  # varargs of array-like; each treated as tensor
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.nn.log_softmax"] = {
    "args": {
        "logits": "tensor"
    },
    "kwargs": {
        "axis": "integer",  # could also accept an int32 Tensor; treating as integer per spec
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.roll_1"] = {
    "args": {
        "a": "tensor",  # can be array-like; mapped to tensor
        "shift": "integer"
    },
    "kwargs": {
        "axis": "integer"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.roll_2"] = {
    "args": {
        "a": "tensor",
        "shift": "integer"
    },
    "kwargs": {
        "axis": "tuple"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.roll_3"] = {
    "args": {
        "a": "tensor",
        "shift": "integer"
    },
    "kwargs": {
        "axis": "list"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.roll_4"] = {
    "args": {
        "a": "tensor",
        "shift": "tuple"
    },
    "kwargs": {
        "axis": "integer"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.roll_5"] = {
    "args": {
        "a": "tensor",
        "shift": "tuple"
    },
    "kwargs": {
        "axis": "tuple"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.roll_6"] = {
    "args": {
        "a": "tensor",
        "shift": "tuple"
    },
    "kwargs": {
        "axis": "list"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.roll_7"] = {
    "args": {
        "a": "tensor",
        "shift": "list"
    },
    "kwargs": {
        "axis": "integer"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.roll_8"] = {
    "args": {
        "a": "tensor",
        "shift": "list"
    },
    "kwargs": {
        "axis": "tuple"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.roll_9"] = {
    "args": {
        "a": "tensor",
        "shift": "list"
    },
    "kwargs": {
        "axis": "list"
    },
    "inner": {}
}
signatures["tf.random.stateless_parameterized_truncated_normal_1"] = {
    "args": {
        "shape": "tensor",
        "seed": "tensor"
    },
    "kwargs": {
        "means": "tensor",  # can also be Python scalar/list
        "stddevs": "tensor",  # can also be Python scalar/list
        "minvals": "tensor",  # can also be Python scalar/list
        "maxvals": "tensor",  # can also be Python scalar/list
        "name": "string"
    },
    "inner": {}
}
signatures["tf.random.stateless_parameterized_truncated_normal_2"] = {
    "args": {
        "shape": "list",
        "seed": "tensor"
    },
    "kwargs": {
        "means": "tensor",  # can also be Python scalar/list
        "stddevs": "tensor",  # can also be Python scalar/list
        "minvals": "tensor",  # can also be Python scalar/list
        "maxvals": "tensor",  # can also be Python scalar/list
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.ptp_1"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "keepdims": "boolean"  # None also allowed
    },
    "inner": {}
}
signatures["tf.experimental.numpy.ptp_2"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "tuple",
        "keepdims": "boolean"  # None also allowed
    },
    "inner": {}
}
signatures["tf.experimental.numpy.ptp_3"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "list",  # list of integers; numpy specifies tuple, TF may accept list
        "keepdims": "boolean"  # None also allowed
    },
    "inner": {}
}
signatures["tf.image.stateless_random_brightness_1"] = {
    "args": {
        "image": "tensor",
        "max_delta": "float",  # could also accept a 0-D float tensor in practice
        "seed": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.image.stateless_random_brightness_2"] = {
    "args": {
        "image": "tensor",
        "max_delta": "float",
        "seed": "tuple"  # typically a length-2 tuple of ints
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.image.stateless_random_brightness_3"] = {
    "args": {
        "image": "tensor",
        "max_delta": "float",
        "seed": "list"  # typically a length-2 list of ints
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.nest.assert_same_structure"] = {
    "args": {
        "nest1": "list",  # Ideally accepts any nested structure or atom (arbitrary Python object)
        "nest2": "list"   # Ideally accepts any nested structure or atom (arbitrary Python object)
    },
    "kwargs": {
        "check_types": "boolean",
        "expand_composites": "boolean"
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
signatures["tf.math.reduce_prod_1"] = {
    "args": {
        "input_tensor": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "keepdims": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.reduce_prod_2"] = {
    "args": {
        "input_tensor": "tensor"
    },
    "kwargs": {
        "axis": "list",  # list of integers
        "keepdims": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.reduce_prod_3"] = {
    "args": {
        "input_tensor": "tensor"
    },
    "kwargs": {
        "axis": "tuple",  # tuple of integers
        "keepdims": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.reduce_prod_4"] = {
    "args": {
        "input_tensor": "tensor"
    },
    "kwargs": {
        "axis": "tensor",  # integer tensor
        "keepdims": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.reduce_sum_1"] = {
    "args": {
        "input_tensor": "tensor"
    },
    "kwargs": {
        "axis": "integer",  # axis can be a single integer dimension
        "keepdims": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.reduce_sum_2"] = {
    "args": {
        "input_tensor": "tensor"
    },
    "kwargs": {
        "axis": "list",  # axis can be a list of integers
        "keepdims": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.reduce_sum_3"] = {
    "args": {
        "input_tensor": "tensor"
    },
    "kwargs": {
        "axis": "tuple",  # axis can be a tuple of integers
        "keepdims": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.reduce_sum_4"] = {
    "args": {
        "input_tensor": "tensor"
    },
    "kwargs": {
        "axis": "tensor",  # axis can be an int32/int64 tensor of indices
        "keepdims": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.image.random_saturation"] = {
    "args": {
        "image": "tensor",
        "lower": "float",
        "upper": "float"
    },
    "kwargs": {
        "seed": "integer"  # could also be a 0-D int tensor in TF, choosing integer
    },
    "inner": {}
}
signatures["tf.autodiff.ForwardAccumulator"] = {
    "args": {
        "primals": "tensor",  # could also accept a structure/tensor_list
        "tangents": "tensor"  # should match primals' shape/structure
    },
    "kwargs": {},
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
signatures["tf.random.categorical"] = {
    "args": {
        "logits": "tensor",
        "num_samples": "integer"  # can also be a scalar int32/int64 Tensor
    },
    "kwargs": {
        "dtype": "dtype",
        "seed": "integer",
        "name": "string"
    },
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
signatures["tf.linalg.eigvalsh"] = {
    "args": {
        "tensor": "tensor"
    },
    "kwargs": {
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
signatures["tf.image.random_brightness"] = {
    "args": {
        "image": "tensor",
        "max_delta": "float"  # In practice may accept a scalar tensor
    },
    "kwargs": {
        "seed": "integer"
    },
    "inner": {}
}
signatures["tf.math.reduce_mean_1"] = {
    "args": {
        "input_tensor": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "keepdims": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.reduce_mean_2"] = {
    "args": {
        "input_tensor": "tensor"
    },
    "kwargs": {
        "axis": "list",  # list of integers
        "keepdims": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.reduce_mean_3"] = {
    "args": {
        "input_tensor": "tensor"
    },
    "kwargs": {
        "axis": "tuple",  # tuple of integers
        "keepdims": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.reduce_mean_4"] = {
    "args": {
        "input_tensor": "tensor"
    },
    "kwargs": {
        "axis": "tensor",  # integer tensor (e.g., 1-D int32)
        "keepdims": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.reduce_any_1"] = {
    "args": {
        "input_tensor": "tensor"  # expects boolean tensor
    },
    "kwargs": {
        "axis": "integer",
        "keepdims": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.reduce_any_2"] = {
    "args": {
        "input_tensor": "tensor"  # expects boolean tensor
    },
    "kwargs": {
        "axis": "list",  # list of integers
        "keepdims": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.reduce_any_3"] = {
    "args": {
        "input_tensor": "tensor"  # expects boolean tensor
    },
    "kwargs": {
        "axis": "tuple",  # tuple of integers
        "keepdims": "boolean",
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
signatures["tf.math.reduce_max_1"] = {
    "args": {
        "input_tensor": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "keepdims": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.reduce_max_2"] = {
    "args": {
        "input_tensor": "tensor"
    },
    "kwargs": {
        "axis": "list",
        "keepdims": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.reduce_max_3"] = {
    "args": {
        "input_tensor": "tensor"
    },
    "kwargs": {
        "axis": "tuple",
        "keepdims": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.reduce_max_4"] = {
    "args": {
        "input_tensor": "tensor"
    },
    "kwargs": {
        "axis": "tensor",  # Tensor of ints for axis
        "keepdims": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.dynamic_partition"] = {
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
signatures["tf.experimental.numpy.random.rand_1"] = {
    "args": {
    },
    "kwargs": {
    },
    "inner": {}
}
signatures["tf.experimental.numpy.random.rand_2"] = {
    "args": {
        "size": "integer"  # Varargs of integers; here shown for a single dimension
    },
    "kwargs": {
    },
    "inner": {}
}
signatures["tf.experimental.numpy.random.rand_3"] = {
    "args": {
        "size": "tuple"  # Represents multiple dims passed via varargs; using a tuple to capture shape
    },
    "kwargs": {
    },
    "inner": {}
}
signatures["tf.experimental.numpy.argmin_1"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer"  # Can be None (default)
    },
    "inner": {}
}
signatures["tf.experimental.numpy.argmin_2"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "tuple"  # Tuple of integers; can be None (default)
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
signatures["tf.experimental.numpy.fliplr_1"] = {
    "args": {
        "m": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.fliplr_2"] = {
    "args": {
        "m": "list"  # array-like inputs (could also be tuple)
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.fliplr_3"] = {
    "args": {
        "m": "tuple"  # array-like inputs (could also be list)
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.max_1"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer",
        "keepdims": "boolean"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.max_2"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "tuple",  # tuple of integers
        "keepdims": "boolean"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.ones_like"] = {
    "args": {
        "a": "tensor"  # array-like treated as tensor
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["tf.lookup.StaticVocabularyTable"] = {
    "args": {
        "initializer": "tensor",  # lookup table initializer object (e.g., KeyValueTensorInitializer); not a tensor per se
        "num_oov_buckets": "integer"
    },
    "kwargs": {
        "lookup_key_dtype": "dtype",
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
signatures["tf.sets.union"] = {
    "args": {
        "a": "tensor",  # Tensor or SparseTensor
        "b": "tensor"   # Tensor or SparseTensor
    },
    "kwargs": {
        "validate_indices": "boolean"
    },
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
signatures["tf.math.cumulative_logsumexp"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "axis": "integer",  # Scalar int or int tensor; treated as integer per spec
        "exclusive": "boolean",
        "reverse": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.random.random_1"] = {
    "args": {},
    "kwargs": {
        "size": "integer"  # Can also be None (omitted)
    },
    "inner": {}
}
signatures["tf.experimental.numpy.random.random_2"] = {
    "args": {},
    "kwargs": {
        "size": "tuple"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.random.random_3"] = {
    "args": {},
    "kwargs": {
        "size": "list"
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
signatures["tf.math.angle"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.cumprod_1"] = {
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
signatures["tf.math.cumprod_2"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "axis": "tensor",  # Typically accepts a Python int too; providing tensor variant separately
        "exclusive": "boolean",
        "reverse": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.flipud"] = {
    "args": {
        "m": "tensor"  # array-like accepted but treated as tensor
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.nn.embedding_lookup_1"] = {
    "args": {
        "params": "tensor",
        "ids": "tensor"
    },
    "kwargs": {
        "max_norm": "float",  # can also be a scalar tensor
        "name": "string"
    },
    "inner": {}
}
signatures["tf.nn.embedding_lookup_2"] = {
    "args": {
        "params": "tensor_list",
        "ids": "tensor"
    },
    "kwargs": {
        "max_norm": "float",  # can also be a scalar tensor
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.isneginf"] = {
    "args": {
        "x": "tensor"  # array_like; using "tensor" as closest type
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.math.argmax"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "axis": "integer",  # Can also be an int32/int64 scalar Tensor; treated as integer
        "output_type": "dtype",
        "name": "string"
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
signatures["tf.linalg.slogdet"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.strings.unicode_script"] = {
    "args": {
        "input": "tensor"  # int32 tensor of Unicode code points
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.nn.isotonic_regression"] = {
    "args": {
        "inputs": "tensor"
    },
    "kwargs": {
        "decreasing": "boolean",
        "axis": "integer"  # could also accept a scalar int Tensor, but using integer
    },
    "inner": {}
}
signatures["tf.linalg.band_part"] = {
    "args": {
        "input": "tensor",
        "num_lower": "integer",  # accepts int or 0-D int tensor
        "num_upper": "integer"   # accepts int or 0-D int tensor
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.random.normal_1"] = {
    "args": {
        "shape": "list"
    },
    "kwargs": {
        "mean": "tensor",  # can also be float
        "stddev": "tensor",  # can also be float
        "dtype": "dtype",
        "seed": "integer",
        "name": "string"
    },
    "inner": {}
}

signatures["tf.random.normal_2"] = {
    "args": {
        "shape": "tuple"
    },
    "kwargs": {
        "mean": "tensor",  # can also be float
        "stddev": "tensor",  # can also be float
        "dtype": "dtype",
        "seed": "integer",
        "name": "string"
    },
    "inner": {}
}

signatures["tf.random.normal_3"] = {
    "args": {
        "shape": "tensor"
    },
    "kwargs": {
        "mean": "tensor",  # can also be float
        "stddev": "tensor",  # can also be float
        "dtype": "dtype",
        "seed": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.empty_1"] = {
    "args": {
        "shape": "integer"
    },
    "kwargs": {
        "dtype": "dtype"  # could also accept NumPy dtype or Python type mapping to a dtype
    },
    "inner": {}
}
signatures["tf.experimental.numpy.empty_2"] = {
    "args": {
        "shape": "tuple"
    },
    "kwargs": {
        "dtype": "dtype"  # could also accept NumPy dtype or Python type mapping to a dtype
    },
    "inner": {}
}
signatures["tf.experimental.numpy.empty_3"] = {
    "args": {
        "shape": "list"
    },
    "kwargs": {
        "dtype": "dtype"  # could also accept NumPy dtype or Python type mapping to a dtype
    },
    "inner": {}
}
signatures["tf.space_to_batch_1"] = {
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
signatures["tf.space_to_batch_2"] = {
    "args": {
        "input": "tensor",
        "block_shape": "list",  # often also accepted as Python sequence
        "paddings": "list"      # often also accepted as Python sequence of pairs
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.space_to_batch_3"] = {
    "args": {
        "input": "tensor",
        "block_shape": "tuple",  # sometimes provided as tuple
        "paddings": "tuple"      # sometimes provided as tuple of pairs
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.nn.fractional_max_pool_1"] = {
    "args": {
        "value": "tensor",
        "pooling_ratio": "list"  # Typically a list of floats (e.g., [1.0, 1.44, 1.73, 1.0])
    },
    "kwargs": {
        "pseudo_random": "boolean",
        "overlapping": "boolean",
        "seed": "integer",
        "name": "string"
    },
    "inner": {}
}

signatures["tf.nn.fractional_max_pool_2"] = {
    "args": {
        "value": "tensor",
        "pooling_ratio": "integer"  # Doc also mentions single int; list form is more common
    },
    "kwargs": {
        "pseudo_random": "boolean",
        "overlapping": "boolean",
        "seed": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.io.matching_files"] = {
    "args": {
        "pattern": "tensor"  # string tensor (scalar or vector)
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.polygamma"] = {
    "args": {
        "a": "tensor",  # Order 'a' is conceptually an integer, but docs specify a Tensor
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.unsorted_segment_sum_1"] = {
    "args": {
        "data": "tensor",
        "segment_ids": "tensor",  # Accepts lists convertible to tensor
        "num_segments": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.unsorted_segment_sum_2"] = {
    "args": {
        "data": "tensor",
        "segment_ids": "tensor",  # Accepts lists convertible to tensor
        "num_segments": "integer"  # Commonly passed as Python int in examples
    },
    "kwargs": {
        "name": "string"
    },
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
signatures["tf.math.unsorted_segment_max_1"] = {
    "args": {
        "data": "tensor",
        "segment_ids": "tensor",
        "num_segments": "integer"  # Often passed as a Python int; scalar int tensor also accepted
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.unsorted_segment_max_2"] = {
    "args": {
        "data": "tensor",
        "segment_ids": "tensor",
        "num_segments": "tensor"  # Scalar int32/int64 tensor
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.image.non_max_suppression_1"] = {
    "args": {
        "boxes": "tensor",
        "scores": "tensor",
        "max_output_size": "integer"  # scalar int tensor accepted; treating as integer
    },
    "kwargs": {
        "iou_threshold": "float",  # 0-D float tensor or Python float; using float
        "score_threshold": "float",  # 0-D float tensor or Python float; using float
        "name": "string"
    },
    "inner": {}
}
signatures["tf.image.non_max_suppression_2"] = {
    "args": {
        "boxes": "tensor",
        "scores": "tensor",
        "max_output_size": "integer"  # scalar int tensor accepted; treating as integer
    },
    "kwargs": {
        "iou_threshold": "tensor",  # alternative: pass as 0-D float tensor
        "score_threshold": "tensor",  # alternative: pass as 0-D float tensor
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.stack"] = {
    "args": {
        "arrays": "tensor_list"  # sequence of tensors/array-like
    },
    "kwargs": {
        "axis": "integer"
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
signatures["tf.image.adjust_brightness"] = {
    "args": {
        "image": "tensor",
        "delta": "float"  # also accepts a scalar tensor
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.image.adjust_brightness_2"] = {
    "args": {
        "image": "tensor",
        "delta": "tensor"  # scalar tensor
    },
    "kwargs": {},
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
signatures["tf.broadcast_dynamic_shape_1"] = {
    "args": {
        "shape_x": "tensor",
        "shape_y": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.broadcast_dynamic_shape_2"] = {
    "args": {
        "shape_x": "tuple",
        "shape_y": "tuple"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.broadcast_dynamic_shape_3"] = {
    "args": {
        "shape_x": "list",
        "shape_y": "list"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.log2"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
    },
    "inner": {}
}
signatures["tf.reshape_1"] = {
    "args": {
        "tensor": "tensor",
        "shape": "tensor"  # Could also accept TensorShape, but using tensor per docs
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}

signatures["tf.reshape_2"] = {
    "args": {
        "tensor": "tensor",
        "shape": "list"  # Commonly a Python list of ints
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}

signatures["tf.reshape_3"] = {
    "args": {
        "tensor": "tensor",
        "shape": "tuple"  # Also accepts a Python tuple of ints
    },
    "kwargs": {
        "name": "string"
    },
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
signatures["tf.experimental.numpy.reciprocal"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
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
signatures["tf.nn.gelu"] = {
    "args": {
        "features": "tensor"  # expects a floating-point tensor
    },
    "kwargs": {
        "approximate": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.histogram_fixed_width"] = {
    "args": {
        "values": "tensor",
        "value_range": "tensor"
    },
    "kwargs": {
        "nbins": "integer",  # Scalar int32 Tensor or Python int treated as integer
        "dtype": "dtype",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.sin"] = {
    "args": {
        "x": "tensor"  # accepts array-like; treated as tensor
    },
    "kwargs": {},
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
signatures["tf.experimental.numpy.arcsin"] = {
    "args": {
        "x": "tensor"
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
signatures["tf.image.hsv_to_rgb"] = {
    "args": {
        "images": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.compat.as_str_any"] = {
    "args": {
        "value": "string"  # accepts any Python object; closest available type
    },
    "kwargs": {
        "encoding": "string"
    },
    "inner": {}
}
signatures["tf.math.segment_prod"] = {
    "args": {
        "data": "tensor",
        "segment_ids": "tensor"  # int32/int64 tensor; treated as "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.ndim"] = {
    "args": {
        "a": "tensor"  # array-like accepted; using 'tensor' as closest match
    },
    "kwargs": {},
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
signatures["tf.experimental.numpy.logical_not"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
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
signatures["tf.math.segment_min"] = {
    "args": {
        "data": "tensor",
        "segment_ids": "tensor"  # int32/int64 1-D tensor
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.image.draw_bounding_boxes"] = {
    "args": {
        "images": "tensor",
        "boxes": "tensor",
        "colors": "tensor"  # Doc mentions list of RGBA; API accepts a float32 tensor
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.segment_max"] = {
    "args": {
        "data": "tensor",
        "segment_ids": "tensor"  # 1-D int tensor
    },
    "kwargs": {
        "name": "string"
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
signatures["tf.image.non_max_suppression_overlaps"] = {
    "args": {
        "overlaps": "tensor",
        "scores": "tensor",
        "max_output_size": "integer"  # Can be Python int or scalar int tensor
    },
    "kwargs": {
        "overlap_threshold": "float",  # 0-D float tensor or Python float
        "score_threshold": "float",    # 0-D float tensor or Python float
        "name": "string"
    },
    "inner": {}
}
signatures["tf.linalg.LinearOperatorDiag_1"] = {
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
signatures["tf.linalg.LinearOperatorDiag_2"] = {
    "args": {
        "diag": "list"  # Could also be a tensor; TF accepts Python lists which are converted to tensors.
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
signatures["tf.nn.leaky_relu"] = {
    "args": {
        "features": "tensor"
    },
    "kwargs": {
        "alpha": "float",  # Could also accept a scalar tensor in TF
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.special.bessel_y0"] = {
    "args": {
        "x": "tensor"  # Tensor or SparseTensor
    },
    "kwargs": {
        "name": "string"
    },
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
signatures["tf.experimental.dlpack.to_dlpack"] = {
    "args": {
        "tf_tensor": "tensor"
    },
    "kwargs": {},
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
signatures["tf.math.logical_not"] = {
    "args": {
        "x": "tensor"  # Tensor of dtype bool
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.rank"] = {
    "args": {
        "input": "tensor"  # Also accepts SparseTensor
    },
    "kwargs": {
        "name": "string"
    },
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
signatures["tf.math.erf"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.linalg.LinearOperatorPermutation_1"] = {
    "args": {
        "perm": "tensor"  # int32/int64 tensor preferred
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
signatures["tf.linalg.LinearOperatorPermutation_2"] = {
    "args": {
        "perm": "list"  # list of ints also accepted
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
signatures["tf.linalg.LinearOperatorPermutation_3"] = {
    "args": {
        "perm": "tuple"  # tuple of ints also accepted
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
signatures["tf.make_tensor_proto_1"] = {
    "args": {
        "values": "tensor"  # also accepts Python/numpy scalars; best mapped to tensor
    },
    "kwargs": {
        "dtype": "dtype",
        "shape": "list",  # list of integers
        "verify_shape": "boolean",
        "allow_broadcast": "boolean"
    },
    "inner": {}
}
signatures["tf.make_tensor_proto_2"] = {
    "args": {
        "values": "list"  # also accepts Python scalar or numpy ndarray; best alternative mapping
    },
    "kwargs": {
        "dtype": "dtype",
        "shape": "list",  # list of integers
        "verify_shape": "boolean",
        "allow_broadcast": "boolean"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.asarray"] = {
    "args": {
        "a": "tensor"  # accepts array-like; using 'tensor' as the closest match
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["tf.sparse.SparseTensor_1"] = {
    "args": {
        "indices": "tensor",  # also accepts list/ndarray-like
        "values": "tensor",   # also accepts list-like
        "dense_shape": "tensor"  # also accepts list-like
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.sparse.SparseTensor_2"] = {
    "args": {
        "indices": "list",
        "values": "tensor",
        "dense_shape": "list"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.sparse.SparseTensor_3"] = {
    "args": {
        "indices": "list",
        "values": "list",
        "dense_shape": "list"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.ensure_shape_1"] = {
    "args": {
        "x": "tensor",
        "shape": "list"  # also accepts TensorShape/TensorShapeProto; using list signature
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.ensure_shape_2"] = {
    "args": {
        "x": "tensor",
        "shape": "tuple"  # also accepts TensorShape/TensorShapeProto; using tuple signature
    },
    "kwargs": {
        "name": "string"
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
signatures["tf.experimental.numpy.minimum"] = {
    "args": {
        "x1": "tensor",
        "x2": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.squeeze_1"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "integer"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.squeeze_2"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "tuple"  # tuple of integers
    },
    "inner": {}
}
signatures["tf.experimental.numpy.squeeze_3"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "axis": "list"  # list of integers
    },
    "inner": {}
}
signatures["tf.math.logical_and_1"] = {
    "args": {
        "x": "tensor",
        "y": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.logical_and_2"] = {
    "args": {
        "x": "tensor",
        "y": "boolean"  # Will be converted to a 0-D bool tensor
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.logical_and_3"] = {
    "args": {
        "x": "boolean",  # Will be converted to a 0-D bool tensor
        "y": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.logical_and_4"] = {
    "args": {
        "x": "boolean",  # Will be converted to a 0-D bool tensor
        "y": "boolean"   # Will be converted to a 0-D bool tensor
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
signatures["tf.experimental.numpy.imag"] = {
    "args": {
        "val": "tensor"
    },
    "kwargs": {},
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
signatures["tf.math.erfcinv"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.unique"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "out_idx": "dtype",  # expects tf.int32 or tf.int64
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
signatures["tf.lookup.TextFileInitializer"] = {
    "args": {
        "filename": "string",
        "key_dtype": "dtype",
        "key_index": "integer",  # Enum values map to integers
        "value_dtype": "dtype",
        "value_index": "integer"  # Enum values map to integers
    },
    "kwargs": {
        "vocab_size": "integer",
        "delimiter": "string",
        "name": "string",
        "value_index_offset": "integer"
    },
    "inner": {}
}
signatures["tf.audio.encode_wav"] = {
    "args": {
        "audio": "tensor",
        "sample_rate": "integer"  # scalar int32 Tensor treated as integer
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.group"] = {
    "args": {
        "inputs": "tensor_list"  # varargs; may include ops or nested lists—treated as tensor_list
    },
    "kwargs": {
        "name": "string"
    },
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
signatures["tf.linalg.LinearOperatorFullMatrix"] = {
    "args": {
        "matrix": "tensor"  # May accept array-like convertible to Tensor
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
signatures["tf.tuple"] = {
    "args": {
        "tensors": "tensor_list"
    },
    "kwargs": {
        "control_inputs": "list",  # list of Operation or Tensor
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.result_type_1"] = {
    "args": {
        "arrays_and_dtypes": "tensor_list"  # Elements can include tensors, dtype objects, or Python scalars
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.result_type_2"] = {
    "args": {
        "arrays_and_dtypes": "list"  # List of dtype-like objects or Python scalars; varargs modeled as a list here
    },
    "kwargs": {},
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
signatures["tf.linalg.LinearOperatorZeros_1"] = {
    "args": {
        "num_rows": "integer"
    },
    "kwargs": {
        "num_columns": "integer",  # could also be tensor
        "batch_shape": "list",  # could also be tuple or 1-D int tensor
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
signatures["tf.linalg.LinearOperatorZeros_2"] = {
    "args": {
        "num_rows": "tensor"
    },
    "kwargs": {
        "num_columns": "tensor",
        "batch_shape": "tensor",
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
signatures["tf.linalg.LinearOperatorZeros_3"] = {
    "args": {
        "num_rows": "integer"
    },
    "kwargs": {
        "num_columns": "integer",
        "batch_shape": "tuple",
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
signatures["tf.experimental.numpy.rad2deg"] = {
    "args": {
        "x": "tensor"  # Could also accept array-like; using "tensor" as closest match
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.positive"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.convert_to_tensor"] = {
    "args": {
        "value": "tensor"  # also accepts numpy arrays, Python lists, and scalars
    },
    "kwargs": {
        "dtype": "dtype",
        "dtype_hint": "dtype",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.signbit"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
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
signatures["tf.experimental.numpy.bitwise_not"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        # No "name" kwarg exposed in tf.experimental.numpy APIs typically
    },
    "inner": {}
}
signatures["tf.experimental.numpy.isrealobj_1"] = {
    "args": {
        "x": "tensor"  # array_like is accepted; mapped to tensor
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.isrealobj_2"] = {
    "args": {
        "x": "list"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.isrealobj_3"] = {
    "args": {
        "x": "tuple"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.math.special.bessel_j0"] = {
    "args": {
        "x": "tensor"  # Tensor or SparseTensor treated as "tensor"
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
signatures["tf.math.special.bessel_k1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.special.bessel_k1e"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"  # TensorFlow op name
    },
    "inner": {}
}
signatures["tf.experimental.numpy.asanyarray_1"] = {
    "args": {
        "a": "tensor"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.asanyarray_2"] = {
    "args": {
        "a": "list"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.asanyarray_3"] = {
    "args": {
        "a": "tuple"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.asanyarray_4"] = {
    "args": {
        "a": "integer"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.asanyarray_5"] = {
    "args": {
        "a": "float"
    },
    "kwargs": {
        "dtype": "dtype"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.asanyarray_6"] = {
    "args": {
        "a": "boolean"
    },
    "kwargs": {
        "dtype": "dtype"
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
signatures["tf.experimental.numpy.fabs"] = {
    "args": {
        "x": "tensor"  # array_like, best mapped to tensor
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
signatures["tf.math.conj"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.conjugate"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.image.adjust_gamma_1"] = {
    "args": {
        "image": "tensor"
    },
    "kwargs": {
        "gamma": "float",
        "gain": "float"
    },
    "inner": {}
}
signatures["tf.image.adjust_gamma_2"] = {
    "args": {
        "image": "tensor"
    },
    "kwargs": {
        "gamma": "tensor",
        "gain": "float"
    },
    "inner": {}
}
signatures["tf.image.adjust_gamma_3"] = {
    "args": {
        "image": "tensor"
    },
    "kwargs": {
        "gamma": "float",
        "gain": "tensor"
    },
    "inner": {}
}
signatures["tf.image.adjust_gamma_4"] = {
    "args": {
        "image": "tensor"
    },
    "kwargs": {
        "gamma": "tensor",
        "gain": "tensor"
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
signatures["tf.experimental.numpy.conj"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {},
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
signatures["tf.math.bessel_i1"] = {
    "args": {
        "x": "tensor"  # Accepts Tensor or SparseTensor
    },
    "kwargs": {
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
signatures["tf.math.special.bessel_j1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
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
signatures["tf.math.reciprocal"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.special.expint"] = {
    "args": {
        "x": "tensor"  # Tensor or SparseTensor
    },
    "kwargs": {
        "name": "string"
    },
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
signatures["tf.math.digamma"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.math.bessel_i1e"] = {
    "args": {
        "x": "tensor"  # Tensor or SparseTensor
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
signatures["tf.math.bessel_i0"] = {
    "args": {
        "x": "tensor"  # Can also be a SparseTensor
    },
    "kwargs": {
        "name": "string"
    },
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
signatures["tf.math.is_finite"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
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
signatures["tf.math.ndtri"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
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
signatures["tf.experimental.numpy.isinf"] = {
    "args": {
        "x": "tensor"  # array-like mapped to tensor
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.arccos"] = {
    "args": {
        "x": "tensor"  # array-like is treated as tensor
    },
    "kwargs": {},
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
signatures["tf.math.tanh"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.tanh"] = {
    "args": {
        "x": "tensor"  # Accepts array-like; using 'tensor' as closest type
    },
    "kwargs": {},
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
signatures["tf.math.asin"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.sign_1"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "out": "tensor",  # could be a tuple of tensors in NumPy ufuncs
        "where": "tensor"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.sign_2"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "out": "tensor",  # could be a tuple of tensors in NumPy ufuncs
        "where": "boolean"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.sign_3"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "out": "tuple",  # tuple of one tensor (NumPy ufunc convention)
        "where": "tensor"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.sign_4"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "out": "tuple",  # tuple of one tensor (NumPy ufunc convention)
        "where": "boolean"
    },
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
signatures["tf.math.acosh"] = {
    "args": {
        "x": "tensor"  # Tensor or TensorLike
    },
    "kwargs": {
        "name": "string"
    },
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
signatures["tf.experimental.numpy.arctanh"] = {
    "args": {
        "x": "tensor"  # array_like input
    },
    "kwargs": {},
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
signatures["tf.math.erfc"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
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
signatures["tf.experimental.numpy.ceil"] = {
    "args": {
        "x": "tensor"  # could also accept Python scalars/array-like; mapped to tensor
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.data.experimental.TFRecordWriter"] = {
    "args": {
        "filename": "string"
    },
    "kwargs": {
        "compression_type": "string"  # optionally None
    },
    "inner": {}
}
signatures["tf.experimental.numpy.floor"] = {
    "args": {
        "x": "tensor"  # array_like also accepted, but mapped to "tensor"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.math.abs"] = {
    "args": {
        "x": "tensor"  # Tensor or SparseTensor
    },
    "kwargs": {
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
signatures["tf.guarantee_const"] = {
    "args": {
        "input": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.array_1"] = {
    "args": {
        "val": "tensor"
    },
    "kwargs": {
        "dtype": "dtype",
        "copy": "boolean",
        "ndmin": "integer"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.array_2"] = {
    "args": {
        "val": "list"
    },
    "kwargs": {
        "dtype": "dtype",
        "copy": "boolean",
        "ndmin": "integer"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.array_3"] = {
    "args": {
        "val": "tuple"
    },
    "kwargs": {
        "dtype": "dtype",
        "copy": "boolean",
        "ndmin": "integer"
    },
    "inner": {}
}
signatures["tf.experimental.numpy.copy"] = {
    "args": {
        "a": "tensor"  # array-like accepted; mapped to tensor
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.saved_model.Asset_1"] = {
    "args": {
        "path": "string"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.saved_model.Asset_2"] = {
    "args": {
        "path": "tensor"  # 0-D tf.string tensor
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.device"] = {
    "args": {
        "device_name": "string"  # could also accept a DeviceSpec object in practice
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.train.latest_checkpoint"] = {
    "args": {
        "checkpoint_dir": "string"  # could also be path-like
    },
    "kwargs": {
        "latest_filename": "string"  # optional; name of checkpoint state file
    },
    "inner": {}
}
signatures["tf.train.get_checkpoint_state"] = {
    "args": {
        "checkpoint_dir": "string"  # path-like accepted; using string
    },
    "kwargs": {
        "latest_filename": "string"
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
signatures["tf.experimental.async_clear_error"] = {
    "args": {},
    "kwargs": {},
    "inner": {}
}
signatures["tf.data.experimental.ThreadingOptions"] = {
    "args": {},
    "kwargs": {},
    "inner": {}
}
signatures["tf.data.experimental.OptimizationOptions"] = {
    "args": {},
    "kwargs": {},
    "inner": {}
}
signatures["tf.data.experimental.DistributeOptions"] = {
    "args": {},
    "kwargs": {},
    "inner": {}
}
signatures["tf.data.experimental.Reducer"] = {
    "args": {
        "init_func": "tensor",  # should be a callable/function
        "reduce_func": "tensor",  # should be a callable/function
        "finalize_func": "tensor"  # should be a callable/function
    },
    "kwargs": {},
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
signatures["tf.data.experimental.from_variant"] = {
    "args": {
        "variant": "tensor",
        "structure": "list"  # Nested structure of tf.TypeSpec; could be list/tuple-like
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.executing_eagerly"] = {
    "args": {},
    "kwargs": {},
    "inner": {}
}
signatures["tf.experimental.numpy.promote_types"] = {
    "args": {
        "type1": "dtype",  # dtype-like (e.g., NumPy dtype or tf.dtypes.DType)
        "type2": "dtype"   # dtype-like (e.g., NumPy dtype or tf.dtypes.DType)
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.dtypes.as_dtype_1"] = {
    "args": {
        "type_value": "dtype"  # also accepts numpy.dtype
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.dtypes.as_dtype_2"] = {
    "args": {
        "type_value": "string"  # string type name like "float32"
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.dtypes.as_dtype_3"] = {
    "args": {
        "type_value": "integer"  # DataType enum values are integers
    },
    "kwargs": {},
    "inner": {}
}
signatures["tf.get_logger"] = {
    "args": {},
    "kwargs": {},
    "inner": {}
}
signatures["tf.train.CheckpointManager_1"] = {
    "args": {
        "checkpoint": "tensor",  # expected a tf.train.Checkpoint/Trackable object
        "directory": "string",
        "max_to_keep": "integer"
    },
    "kwargs": {
        "keep_checkpoint_every_n_hours": "float",
        "checkpoint_name": "string",
        "step_counter": "tensor",  # can be a tf.Variable (int scalar)
        "checkpoint_interval": "integer",
        "init_fn": "tensor"  # expected a callable
    },
    "inner": {}
}

signatures["tf.train.CheckpointManager_2"] = {
    "args": {
        "checkpoint": "tensor",  # expected a tf.train.Checkpoint/Trackable object
        "directory": "string",
        "max_to_keep": "integer"
    },
    "kwargs": {
        "keep_checkpoint_every_n_hours": "float",
        "checkpoint_name": "string",
        "step_counter": "integer",  # can also be provided as a Python int
        "checkpoint_interval": "integer",
        "init_fn": "tensor"  # expected a callable
    },
    "inner": {}
}
signatures["tf.nn.fractional_avg_pool"] = {
    "args": {
        "value": "tensor",
        "pooling_ratio": "list"  # list of floats; tuples may also work in practice
    },
    "kwargs": {
        "pseudo_random": "boolean",
        "overlapping": "boolean",
        "seed": "integer",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.train.Coordinator"] = {
    "args": {},
    "kwargs": {
        "clean_stop_exception_types": "tuple"  # tuple of Exception types
    },
    "inner": {}
}
signatures["tf.train.ExponentialMovingAverage_1"] = {
    "args": {
        "decay": "float"
    },
    "kwargs": {
        "num_updates": "tensor",  # could also be integer or tf.Variable
        "zero_debias": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.train.ExponentialMovingAverage_2"] = {
    "args": {
        "decay": "tensor"
    },
    "kwargs": {
        "num_updates": "tensor",  # could also be integer or tf.Variable
        "zero_debias": "boolean",
        "name": "string"
    },
    "inner": {}
}
signatures["tf.train.checkpoints_iterator"] = {
    "args": {
        "checkpoint_dir": "string"
    },
    "kwargs": {
        "min_interval_secs": "float",
        "timeout": "float",  # could be None
        "timeout_fn": "string"  # should be a callable/function
    },
    "inner": {}
}

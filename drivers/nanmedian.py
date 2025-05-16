import numpy as np
import random

# Function to set seed for reproducibility
def torch_nanmedian_variant1(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input["input"])
    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.nanmedian(input_tensor)
    values = result.item() if not torch.isnan(result) else float('nan')

    if not cpu:
        values = result.cpu().item()

    return {"nanmedian_value": values}

# TensorFlow: Equivalent for torch.nanmedian(input)
def tf_nanmedian_variant1(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input["input"])
        
        values = tf.boolean_mask(input_tensor, ~tf.math.is_nan(input_tensor))
        result = tf.numpy_function(np.median, [values.numpy()], tf.float32)
        
        result_value = result.numpy().item() if values.numpy().size > 0 else float('nan')

    return {"nanmedian_value": result_value}

# PyTorch: torch.nanmedian(input, dim=-1, keepdim=False, *, out=None)
def torch_nanmedian_variant2(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input["input"])
    dim = input.get("dim", -1)
    keepdim = input.get("keepdim", False)

    if not cpu:
        input_tensor = input_tensor.cuda()

    values, indices = torch.nanmedian(input_tensor, dim=dim, keepdim=keepdim)

    if not cpu:
        values = values.cpu()
        indices = indices.cpu()

    return {"nanmedian_values": values.numpy(), "nanmedian_indices": indices.numpy()}

# TensorFlow: Equivalent for torch.nanmedian(input, dim=-1, keepdim=False, *, out=None)
def tf_nanmedian_variant2(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input["input"])
        dim = input.get("dim", -1)
        keepdim = input.get("keepdim", False)

        values = tf.boolean_mask(input_tensor, ~tf.math.is_nan(input_tensor))
        
        processed_result = tf.numpy_function(lambda x: np.nanmedian(x, axis=dim, keepdims=keepdim), 
                                             [values.numpy()], [tf.float32, tf.int64])
        
        nanmedian_values, nanmedian_indices = map(lambda x: x.numpy(), processed_result)

    return {"nanmedian_values": nanmedian_values, "nanmedian_indices": nanmedian_indices}


def main():
    input_data_variant1 = {
        "input": np.array([1, float('nan'), 3, 2], dtype=np.float32)
    }

    input_data_variant2 = {
        "input": np.array([[2, 3, 1], [float('nan'), 1, float('nan')]], dtype=np.float32),
        "dim": 0,
        "keepdim": False
    }

    # Testing Variant 1
    torch_result_v1 = torch_nanmedian_variant1(input_data_variant1)
    tf_result_v1 = tf_nanmedian_variant1(input_data_variant1)
    print("Torch result for variant 1:", torch_result_v1)
    print("TensorFlow result for variant 1:", tf_result_v1)
    print("Equal" if torch_result_v1["nanmedian_value"] == tf_result_v1["nanmedian_value"] else "Not equal")

    # Testing Variant 2
    torch_result_v2 = torch_nanmedian_variant2(input_data_variant2)
    tf_result_v2 = tf_nanmedian_variant2(input_data_variant2)
    print("Torch result for variant 2:", torch_result_v2)
    print("TensorFlow result for variant 2:", tf_result_v2)
    print("Equal" if np.array_equal(torch_result_v2["nanmedian_values"], tf_result_v2["nanmedian_values"]) and 
                 np.array_equal(torch_result_v2["nanmedian_indices"], tf_result_v2["nanmedian_indices"]) else "Not equal")


if __name__ == "__main__":
    main()
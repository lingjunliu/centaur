import numpy as np

# PyTorch Implementation
def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    running_mean = torch.tensor(input.get("running_mean", None)) if input.get("running_mean", None) is not None else None
    running_var = torch.tensor(input.get("running_var", None)) if input.get("running_var", None) is not None else None
    weight = torch.tensor(input.get("weight", None)) if input.get("weight", None) is not None else None
    bias = torch.tensor(input.get("bias", None)) if input.get("bias", None) is not None else None
    use_input_stats = input.get("use_input_stats", True)
    momentum = input.get("momentum", 0.1)
    eps = input.get("eps", 1e-05)

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.nn.functional.instance_norm(
        input_tensor, running_mean=running_mean, running_var=running_var, weight=weight, 
        bias=bias, use_input_stats=use_input_stats, momentum=momentum, eps=eps
    )

    if not cpu:
        result = result.cpu()

    return {"instance_norm_result": result.numpy()}

# TensorFlow Implementation
def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        running_mean = tf.constant(input.get("running_mean", None)) if input.get("running_mean", None) is not None else None
        running_var = tf.constant(input.get("running_var", None)) if input.get("running_var", None) is not None else None
        weight = tf.constant(input.get("weight", None)) if input.get("weight", None) is not None else None
        bias = tf.constant(input.get("bias", None)) if input.get("bias", None) is not None else None
        use_input_stats = input.get("use_input_stats", True)
        momentum = input.get("momentum", 0.1)
        eps = input.get("eps", 1e-05)

        # Simulate the instance normalization
        if use_input_stats:
            mean, variance = tf.nn.moments(input_tensor, axes=[2, 3], keepdims=True)
        else:
            mean, variance = running_mean, running_var

        normalized = (input_tensor - mean) / tf.math.sqrt(variance + eps)

        if weight is not None:
            normalized = normalized * weight

        if bias is not None:
            normalized = normalized + bias

        return {"instance_norm_result": normalized.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.random.randn(4, 3, 8, 8).astype(np.float32),
        "running_mean": None,
        "running_var": None,
        "weight": None,
        "bias": None,
        "use_input_stats": True,
        "momentum": 0.1,
        "eps": 1e-05
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result["instance_norm_result"])

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result["instance_norm_result"])

    # Assert and print result
    assert np.allclose(torch_result["instance_norm_result"], tf_result["instance_norm_result"], atol=1e-4), "Results differ!"
    print('equal')

if __name__ == "__main__":
    main()
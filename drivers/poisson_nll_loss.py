import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    target_tensor = torch.tensor(input["target"])

    if not cpu:
        input_tensor = input_tensor.cuda()
        target_tensor = target_tensor.cuda()
    
    log_input = input.get("log_input", True)
    full = input.get("full", False)
    size_average = input.get("size_average", None)
    eps = input.get("eps", 1e-08)
    reduce = input.get("reduce", None)
    reduction = input.get("reduction", 'mean')

    # Apply to torch.nn.functional.poisson_nll_loss
    loss = torch.nn.functional.poisson_nll_loss(
        input_tensor, target_tensor, log_input=log_input, full=full,
        size_average=size_average, eps=eps, reduce=reduce, reduction=reduction
    )

    if not cpu:
        loss = loss.cpu()

    return {"poisson_nll_loss": float(loss.item())}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        target_tensor = tf.constant(input["target"])

        log_input = input.get("log_input", True)
        full = input.get("full", False)
        eps = input.get("eps", 1e-08)
        reduction = input.get("reduction", 'mean')

        if log_input:
            log_input_tensor = input_tensor
            input_tensor = tf.exp(input_tensor)
        else:
            log_input_tensor = tf.math.log(input_tensor + eps)
        
        loss = input_tensor - target_tensor * log_input_tensor

        if full:
            stirling_approx = target_tensor * tf.math.log(target_tensor) - target_tensor + 0.5 * tf.math.log(2 * np.pi * target_tensor)
            zero_mask = tf.dtypes.cast(target_tensor == 0.0, tf.float32)
            stirling_approx = stirling_approx * (1 - zero_mask)
            loss += stirling_approx

        if reduction == 'mean':
            loss = tf.reduce_mean(tf.clip_by_value(loss, clip_value_min=-1000, clip_value_max=1000))
        elif reduction == 'sum':
            loss = tf.reduce_sum(tf.clip_by_value(loss, clip_value_min=-1000, clip_value_max=1000))

        return {"poisson_nll_loss": float(loss.numpy())}

def main():
    # Example input
    input_data = {
        "input": np.array([[0.5, 0.3, 0.8], [0.2, 0.6, 0.9]], dtype=np.float32),
        "target": np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=np.float32),
        "log_input": True,
        "full": False,
        "size_average": None,
        "eps": 1e-08,
        "reduce": None,
        "reduction": 'mean'
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Compare results
    torch_result_np = np.array(torch_result["poisson_nll_loss"])
    tf_result_np = np.array(tf_result["poisson_nll_loss"])

    assert np.isclose(torch_result_np, tf_result_np), "Results are not equal"
    print("equal")

if __name__ == "__main__":
    main()
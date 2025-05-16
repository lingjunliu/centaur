import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    target_tensor = torch.tensor(input["target"])
    log_input = input.get("log_input", True)
    full = input.get("full", False)
    eps = input.get("eps", 1e-08)
    reduction = input.get("reduction", 'mean')

    # Apply to torch.nn.PoissonNLLLoss
    loss_fn = torch.nn.PoissonNLLLoss(
        log_input=log_input, full=full, eps=eps, reduction=reduction
    )
    if not cpu:
        input_tensor = input_tensor.cuda()
        target_tensor = target_tensor.cuda()
        loss_fn = loss_fn.cuda()
    
    loss = loss_fn(input_tensor, target_tensor)

    if not cpu:
        loss = loss.cpu()

    return {"poisson_nll_loss": float(loss.item())}

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
        target_tensor = tf.constant(input["target"])
        log_input = input.get("log_input", True)
        full = input.get("full", False)
        eps = input.get("eps", 1e-08)
        reduction = input.get("reduction", 'mean')

        if log_input:
            loss = tf.math.exp(input_tensor) - target_tensor * input_tensor
        else:
            loss = input_tensor - target_tensor * tf.math.log(input_tensor + eps)
        
        if full:
            stirling_approx = target_tensor * tf.math.log(target_tensor) - target_tensor + 0.5 * tf.math.log(2 * np.pi * target_tensor)
            stirling_approx = tf.where(tf.math.is_nan(stirling_approx), tf.zeros_like(stirling_approx), stirling_approx)
            loss += stirling_approx

        if reduction == 'mean':
            loss = tf.reduce_mean(loss)
        elif reduction == 'sum':
            loss = tf.reduce_sum(loss)

        return {"poisson_nll_loss": float(loss.numpy())}

def main():
    # Example input
    input_data = {
        "input": np.array([[0.5, 0.3, 0.8], [0.2, 0.6, 0.9]], dtype=np.float32),
        "target": np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=np.float32),
        "log_input": True,
        "full": False,
        "eps": 1e-08,
        "reduction": 'mean'
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    if np.isclose(torch_result["poisson_nll_loss"], tf_result["poisson_nll_loss"], atol=1e-6):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()
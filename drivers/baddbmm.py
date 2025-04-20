import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack the input dictionary
    input_tensor = torch.tensor(input["input"])
    batch1_tensor = torch.tensor(input["batch1"])
    batch2_tensor = torch.tensor(input["batch2"])
    beta = input.get("beta", 1)
    alpha = input.get("alpha", 1)

    if not cpu:
        input_tensor = input_tensor.cuda()
        batch1_tensor = batch1_tensor.cuda()
        batch2_tensor = batch2_tensor.cuda()

    # Apply torch.baddbmm
    result = torch.baddbmm(input_tensor, batch1_tensor, batch2_tensor, beta=beta, alpha=alpha)

    if not cpu:
        result = result.cpu()

    return {"baddbmm_result": result.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack the input dictionary
        input_tensor = tf.constant(input["input"])
        batch1_tensor = tf.constant(input["batch1"])
        batch2_tensor = tf.constant(input["batch2"])
        beta = input.get("beta", 1)
        alpha = input.get("alpha", 1)

        # Manually perform the equivalent of torch.baddbmm
        matmul_result = tf.linalg.matmul(batch1_tensor, batch2_tensor)
        scaled_matmul = alpha * matmul_result
        scaled_input = beta * input_tensor
        result = scaled_input + scaled_matmul

    return {"baddbmm_result": result.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.random.randn(10, 3, 5).astype(np.float32),
        "batch1": np.random.randn(10, 3, 4).astype(np.float32),
        "batch2": np.random.randn(10, 4, 5).astype(np.float32),
        "beta": 1,
        "alpha": 1
    }

    # Torch version
    torch_result = torch_version(input_data, cpu=True)
    print("Torch result:", torch_result["baddbmm_result"])

    # TensorFlow version
    tensorflow_result = tensorflow_version(input_data, cpu=True)
    print("TensorFlow result:", tensorflow_result["baddbmm_result"])

    # Assert equality
    np.testing.assert_allclose(torch_result["baddbmm_result"], tensorflow_result["baddbmm_result"], atol=1e-2)
    print("equal")

if __name__ == "__main__":
    main()
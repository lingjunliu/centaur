import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    batch1_tensor = torch.tensor(input["batch1"])
    batch2_tensor = torch.tensor(input["batch2"])
    beta = input.get("beta", 1)
    alpha = input.get("alpha", 1)

    if not cpu:
        input_tensor = input_tensor.cuda()
        batch1_tensor = batch1_tensor.cuda()
        batch2_tensor = batch2_tensor.cuda()
    
    # Apply to torch.addbmm
    result = torch.addbmm(
        input=input_tensor,
        batch1=batch1_tensor,
        batch2=batch2_tensor,
        beta=beta,
        alpha=alpha
    )

    if not cpu:
        result = result.cpu()

    return {"addbmm_result": result.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    # Custom implementation to match the behavior of torch.addbmm

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        batch1_tensor = tf.constant(input["batch1"])
        batch2_tensor = tf.constant(input["batch2"])
        beta = input.get("beta", 1)
        alpha = input.get("alpha", 1)

        # Batch matrix multiply and aggregate
        batch_mmult = tf.einsum('bij,bjk->bik', batch1_tensor, batch2_tensor)
        sum_mmult = tf.reduce_sum(batch_mmult, axis=0)

        # Compute the result with alpha and beta
        result = beta * input_tensor + alpha * sum_mmult

        return {"addbmm_result": result.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.random.randn(3, 5).astype(np.float32),
        "batch1": np.random.randn(10, 3, 4).astype(np.float32),
        "batch2": np.random.randn(10, 4, 5).astype(np.float32),
        "beta": 1,
        "alpha": 1
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    torch_res_np = np.array(torch_result["addbmm_result"])
    tf_res_np = np.array(tf_result["addbmm_result"])

    if np.allclose(torch_res_np, tf_res_np, rtol=1e-5, atol=1e-8):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()
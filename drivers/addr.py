import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    vec1_tensor = torch.tensor(input["vec1"])
    vec2_tensor = torch.tensor(input["vec2"])
    beta = input.get("beta", 1)
    alpha = input.get("alpha", 1)

    if not cpu:
        input_tensor = input_tensor.cuda()
        vec1_tensor = vec1_tensor.cuda()
        vec2_tensor = vec2_tensor.cuda()
    
    # Apply torch.addr
    result = torch.addr(input_tensor, vec1_tensor, vec2_tensor, beta=beta, alpha=alpha)

    if not cpu:
        result = result.cpu()

    return {"addr_result": result.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = '/cpu:0'
    else:
        device_string = '/gpu:0'

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        vec1_tensor = tf.constant(input["vec1"])
        vec2_tensor = tf.constant(input["vec2"])
        beta = input.get("beta", 1)
        alpha = input.get("alpha", 1)
        
        # Perform the operation: out = beta * input + alpha * outer_product(vec1, vec2)
        outer_product = tf.tensordot(vec1_tensor, vec2_tensor, axes=0)
        result = beta * input_tensor + alpha * outer_product

    return {"addr_result": result.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.zeros((3, 2), dtype=np.float32),
        "vec1": np.arange(1., 4., dtype=np.float32),  # np.array([1.0, 2.0, 3.0])
        "vec2": np.arange(1., 3., dtype=np.float32),  # np.array([1.0, 2.0])
        "beta": 1,
        "alpha": 1
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assert equality
    assert np.allclose(torch_result["addr_result"], tf_result["addr_result"]), "Results are not equal"
    print("equal")

if __name__ == "__main__":
    main()
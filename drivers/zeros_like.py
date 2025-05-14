import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    dtype = input.get("dtype", None)
    layout = input.get("layout", None)
    device = input.get("device", 'cpu' if cpu else 'cuda')
    requires_grad = input.get("requires_grad", False)
    memory_format = input.get("memory_format", torch.preserve_format)

    try:
        device_tensor = torch.device(device)
        input_tensor = input_tensor.to(device_tensor)
    except Exception as e:
        device_tensor = torch.device('cpu')

    # Create a tensor with zeros, similar to input_tensor
    result = torch.zeros_like(input_tensor, dtype=dtype, layout=layout, device=device_tensor, 
                              requires_grad=requires_grad, memory_format=memory_format)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])

        # Create a tensor with zeros, similar to input_tensor
        result = tf.zeros_like(input_tensor)

        if not cpu:
            result = tf.identity(result)  # Ensure tensor is on GPU if needed

        return {"result": result.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[0.5, 0.3, 0.8], [0.2, 0.6, 0.9]], dtype=np.float32),  # Ensure target is float type
        "dtype": None,
        "layout": None,
        "device": 'cpu',
        "requires_grad": False,
        "memory_format": torch.preserve_format,
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Convert both results to numpy arrays for comparison
    assert np.array_equal(torch_result["result"], tf_result["result"]), "Results are not equal"
    print("equal")

if __name__ == "__main__":
    main()
import numpy as np

def torch_empty_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    size = input["size"]
    dtype = input.get("dtype", torch.float32)
    layout = input.get("layout", torch.strided)
    device = input.get("device", "cpu") if cpu else input.get("device", "cuda")
    requires_grad = input.get("requires_grad", False)
    pin_memory = input.get("pin_memory", False)
    memory_format = input.get("memory_format", torch.contiguous_format)

    tensor = torch.empty(size, dtype=dtype, layout=layout, device=device, 
                         requires_grad=requires_grad, pin_memory=pin_memory, 
                         memory_format=memory_format)

    return tensor


def tensorflow_empty_version(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    size = input["size"]
    dtype = input.get("dtype", tf.float32)
    device_string = "/cpu:0" if cpu else "/gpu:0"
    
    with tf.device(device_string):
        tensor = tf.Variable(initial_value=tf.zeros(size, dtype=dtype), trainable=input.get("requires_grad", False))

    return tensor


def main():
    # Example input for empty tensor creation
    input_data = {
        "size": (2, 3),
        "dtype": torch.float32,  # For PyTorch
        "layout": torch.strided,
        "device": None,
        "requires_grad": False,
        "pin_memory": False,
        "memory_format": torch.contiguous_format
    }

    # Torch example
    torch_tensor = torch_empty_version(input_data)
    print("Torch empty tensor:", torch_tensor)

    # TensorFlow example (adjust for equivalent dtype)
    input_data_tf = input_data.copy()
    input_data_tf["dtype"] = tf.float32
    tensorflow_tensor = tensorflow_empty_version(input_data_tf)
    print("TensorFlow empty tensor:", tensorflow_tensor)

    # Note: Actual values for uninitialized memory are undefined and can be different.
    # Here we ensure the shapes are same and dtype matches.
    assert torch_tensor.shape == tensorflow_tensor.shape
    assert torch_tensor.dtype == torch.float32
    assert tensorflow_tensor.dtype == tf.float32
    print("Shape and dtype are equal")

if __name__ == "__main__":
    main()
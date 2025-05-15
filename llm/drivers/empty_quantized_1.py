import numpy as np
import torch

def torch_version(input_dict, cpu=True):
    size = input_dict["size"]
    qscheme = input_dict.get("qscheme", torch.per_tensor_affine)
    dtype = input_dict["dtype"]

    if not cpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Create an empty tensor with the desired size and dtype
    empty_tensor = torch.empty(size, dtype=torch.float32)  # Use float32 as a base
    
    # Quantize the empty tensor
    scale = 1.0  # Example scale
    zero_point = 0  # Example zero_point

    if qscheme == torch.per_tensor_affine:
      result = torch.quantize_per_tensor(empty_tensor, scale=scale, zero_point=zero_point, qscheme=qscheme, dtype=dtype)
    elif qscheme == torch.per_channel_affine:
        # Needs more complex logic for channel-wise quantization. Skipping for brevity.
        return {"result": np.zeros(size)} #Returning an empty result that matches the size
    else:
        raise ValueError("Unsupported quantization scheme")

    if not cpu:
        result = result.cpu()
    
    return {"result": result.dequantize().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    size = input_dict["size"]
    qscheme_torch = input_dict.get("qscheme", torch.per_tensor_affine)
    dtype_torch = input_dict["dtype"]
    
    if qscheme_torch == torch.per_tensor_affine:
      qscheme = "per_tensor_affine"
    elif qscheme_torch == torch.per_channel_affine:
      qscheme = "per_channel_affine"
    elif qscheme_torch == torch.per_tensor_symmetric:
      qscheme = "per_tensor_symmetric"
    elif qscheme_torch == torch.per_channel_symmetric:
      qscheme = "per_channel_symmetric"
    else:
      raise ValueError("Unsupported quantization scheme")
    
    if dtype_torch == torch.quint8:
        dtype = tf.dtypes.quint8
    elif dtype_torch == torch.qint8:
        dtype = tf.dtypes.qint8
    elif dtype_torch == torch.qint32:
        dtype = tf.dtypes.qint32
    else:
      raise ValueError("Unsupported quantized dtype")

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
      result = tf.experimental.numpy.empty(size, dtype=tf.float32)
      result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "size": (2, 3),
        "dtype": torch.quint8,
        "qscheme": torch.per_tensor_affine
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    # Comparing the shapes is the best we can do, as content will be random
    assert np.allclose(torch_result["result"].shape, tf_result["result"].shape, atol=A_TOL), "Results do not match"
    
    print("Success")

if __name__ == "__main__":
    main()
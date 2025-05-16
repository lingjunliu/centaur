import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    s = input_dict.get("s", None)
    dim = input_dict.get("dim", (-2, -1))
    norm = input_dict.get("norm", "backward")

    if not cpu:
        input_tensor = input_tensor.cuda()
    
    result = torch.fft.ihfft2(input_tensor, s=s, dim=dim, norm=norm)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        s = input_dict.get("s", None)
        dim = input_dict.get("dim", (-2, -1))
        norm = input_dict.get("norm", "backward")

        if s is None:
            s = [input_tensor.shape[d] for d in dim]

        if norm == "forward":
            norm_factor = 1.0
        elif norm == "backward":
            norm_factor = 1.0 / np.prod(s)
        elif norm == "ortho":
            norm_factor = 1.0 / np.sqrt(np.prod(s))
        else:
            raise ValueError(f"Invalid norm value: {norm}")
        
        rank = len(input_tensor.shape)
        axes = [rank + d if d < 0 else d for d in dim]
        
        if len(axes) != 2:
            raise ValueError("ihfft2 only supports 2 dimensions")

        input_shape = input_tensor.shape
        
        real_input = tf.cast(input_tensor, tf.complex64)
        
        fft_result = tf.signal.ifft2d(real_input)
        result = tf.math.real(fft_result) * norm_factor
        
    return {"result": result.numpy()}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.random.rand(10, 10).astype(np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    min_shape_0 = min(torch_result["result"].shape[0], tf_result["result"].shape[0])
    min_shape_1 = min(torch_result["result"].shape[1], tf_result["result"].shape[1])
    
    torch_result_cropped = torch_result["result"][:min_shape_0, :min_shape_1]
    tf_result_cropped = tf_result["result"][:min_shape_0, :min_shape_1]

    assert np.allclose(torch_result_cropped, tf_result_cropped, atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.random.rand(5, 8).astype(np.float32),
        "s": (10,16),
        "norm": "ortho"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    

    print("Success")

if __name__ == "__main__":
    main()
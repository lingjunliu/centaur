import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    output_size = input_dict["output_size"]
    return_indices = input_dict.get("return_indices", False)

    if not cpu:
        input_tensor = input_tensor.cuda()

    m = torch.nn.AdaptiveMaxPool3d(output_size, return_indices=return_indices)
    result = m(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        output_size = input_dict["output_size"]
        return_indices = input_dict.get("return_indices", False)

        input_shape = tf.shape(input_tensor)
        batch_size = input_shape[0] if len(input_shape) == 5 else 1
        channels = input_shape[1] if len(input_shape) >= 4 else input_shape[0]
        din = input_shape[-3]
        hin = input_shape[-2]
        win = input_shape[-1]
        
        if isinstance(output_size, int):
            dout, hout, wout = output_size, output_size, output_size
        elif output_size is None:
            dout, hout, wout = din, hin, win
        else:
            dout = output_size[0] if output_size[0] is not None else din
            hout = output_size[1] if output_size[1] is not None else hin
            wout = output_size[2] if output_size[2] is not None else win

        if dout == 0 or dout is None:
            dout = 1
        if hout == 0 or hout is None:
            hout = 1
        if wout == 0 or wout is None:
            wout = 1

        stride_d = max(1, din // dout)
        stride_h = max(1, hin // hout)
        stride_w = max(1, win // wout)

        ksize_d = din - (dout - 1) * stride_d if dout > 0 else din
        ksize_h = hin - (hout - 1) * stride_h if hout > 0 else hin
        ksize_w = win - (wout - (win // stride_w)) if wout > 0 else win

        if len(input_shape) == 5:
            result = tf.nn.max_pool3d(
                input_tensor,
                ksize=[1, ksize_d, ksize_h, ksize_w, 1],
                strides=[1, stride_d, stride_h, stride_w, 1],
                padding='VALID',
            )
            target_shape = [batch_size, channels, dout, hout, wout]
            
        else:
            reshaped_input = tf.reshape(input_tensor, [1, channels, din, hin, win])
            result = tf.nn.max_pool3d(
                reshaped_input,
                ksize=[1, ksize_d, ksize_h, ksize_w, 1],
                strides=[1, stride_d, stride_h, stride_w, 1],
                padding='VALID',
            )
            target_shape = [1, channels, dout, hout, wout]

        try:
            result = tf.reshape(result, target_shape)
        except tf.errors.InvalidArgumentError as e:
            print(f"Reshape error: {e}")
            print(f"Input shape: {input_shape}")
            print(f"Target shape: {target_shape}")
            print(f"Result shape: {tf.shape(result)}")
            raise

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.random.rand(1, 3, 10, 10, 10).astype(np.float32),
        "output_size": (5, 5, 5),
        "return_indices": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.random.rand(2, 4, 8, 8, 8).astype(np.float32),
        "output_size": 4,
        "return_indices": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    input_data = {
        "input": np.random.rand(1, 16, 12, 16, 20).astype(np.float32),
        "output_size": (6, None, None),
        "return_indices": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    input_data = {
        "input": np.random.rand(3, 8, 10, 12, 14).astype(np.float32),
        "output_size": (None, 6, None),
        "return_indices": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()
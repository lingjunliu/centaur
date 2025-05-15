import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    size = input_dict.get("size", None)
    scale_factor = input_dict.get("scale_factor", None)

    if not cpu:
        input_tensor = input_tensor.cuda()

    upsample = torch.nn.UpsamplingNearest2d(size=size, scale_factor=scale_factor)
    result = upsample(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
    size = input_dict.get("size", None)
    scale_factor = input_dict.get("scale_factor", None)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_shape = tf.shape(input_tensor)
        height = input_shape[2]
        width = input_shape[3]

        if size is not None:
            new_height, new_width = size
        elif scale_factor is not None:
            if isinstance(scale_factor, float) or isinstance(scale_factor, int):
                scale_factor_height = float(scale_factor)
                scale_factor_width = float(scale_factor)
            else:
                scale_factor_height, scale_factor_width = scale_factor
            new_height = int(tf.cast(height, tf.float32) * scale_factor_height)
            new_width = int(tf.cast(width, tf.float32) * scale_factor_width)
        else:
            raise ValueError("Either size or scale_factor must be specified")
        
        result = tf.image.resize(input_tensor, [new_height, new_width], method='nearest')
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 1, 5, 5).astype(np.float32),
        "scale_factor": 2.0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    torch_result_reshaped = np.squeeze(torch_result["result"], axis=1)
    
    assert np.allclose(torch_result_reshaped, np.squeeze(tf_result["result"], axis=0), atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.random.rand(1, 1, 5, 5).astype(np.float32),
        "size": (10, 10)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    torch_result_reshaped = np.squeeze(torch_result["result"], axis=1)

    assert np.allclose(torch_result_reshaped, np.squeeze(tf_result["result"], axis=0), atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()
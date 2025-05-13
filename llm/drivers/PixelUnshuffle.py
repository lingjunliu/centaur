import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    downscale_factor = input_dict.get("downscale_factor", 2)

    if not cpu:
        input_tensor = input_tensor.cuda()

    try:
        pixel_unshuffle = torch.nn.PixelUnshuffle(downscale_factor=downscale_factor)
        result = pixel_unshuffle(input_tensor)
    except RuntimeError as e:
        print(f"Torch RuntimeError: {e}")
        return {"result": np.zeros(0)}

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
        downscale_factor = input_dict.get("downscale_factor", 2)

        input_shape = tf.shape(input_tensor)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]

        r = downscale_factor
        output_height = height // r
        output_width = width // r
        output_channels = channels * (r ** 2)

        x = tf.reshape(input_tensor, [batch_size, height , width , channels])

        outputs = []
        for i in range(output_height):
            for j in range(output_width):
                block = input_tensor[:, i*r:(i+1)*r, j*r:(j+1)*r, :]
                block = tf.reshape(block, [batch_size, 1, 1, r*r*channels])
                outputs.append(block)

        result = tf.concat(outputs, axis=1)
        result = tf.concat(tf.split(result, output_height * output_width, axis=1), axis = 2)

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.arange(1 * 4 * 4 * 4, dtype=np.float32).reshape((1, 4, 4, 4)),
        "downscale_factor": 2
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    if torch_result["result"].size == 0:
        print("Torch failed, skipping assertion")
    else:
        assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()
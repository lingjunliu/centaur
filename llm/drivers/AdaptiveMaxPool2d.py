import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    output_size = input_dict["output_size"]

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.nn.AdaptiveMaxPool2d(output_size)(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"])
    output_size = input_dict["output_size"]

    input_shape = input_tensor.shape
    if len(input_shape) == 3:
        input_tensor = tf.expand_dims(input_tensor, axis=0)
    
    input_shape = input_tensor.shape
    
    target_height = output_size[0]
    target_width = output_size[1]

    height = input_shape[2]
    width = input_shape[3]

    stride_height = height // target_height
    stride_width = width // target_width

    if stride_height * target_height < height:
      stride_height += 1
    if stride_width * target_width < width:
      stride_width += 1
      
    padding_height = max(0, (stride_height * target_height - height + 1) // 2)
    padding_width = max(0, (stride_width * target_width - width + 1) // 2)

    padded_tensor = tf.pad(input_tensor, [[0, 0], [0, 0], [padding_height, padding_height], [padding_width, padding_width]], "CONSTANT")

    if padded_tensor.shape[2] > 0 and padded_tensor.shape[3] > 0:
        result = tf.nn.pool(
            input=padded_tensor,
            window_shape=[stride_height, stride_width],
            pooling_type="MAX",
            padding="VALID",
            strides=[stride_height, stride_width]
        )
    
        result = tf.image.resize(result, [target_height, target_width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    else:
        result = tf.zeros((1, target_height, target_width, input_tensor.shape[1]))
    result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]], dtype=np.float32),
        "output_size": (2, 2)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()
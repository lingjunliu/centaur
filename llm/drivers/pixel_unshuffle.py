import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict['input'])
    downscale_factor = input_dict['downscale_factor']

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.pixel_unshuffle(input_tensor, downscale_factor)

    if not cpu:
        result = result.cpu()

    return {'result': result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict['input'])
    downscale_factor = input_dict['downscale_factor']

    input_shape = input_tensor.shape
    if len(input_shape) == 3:
        input_tensor = tf.expand_dims(input_tensor, axis=0)
    input_shape = input_tensor.shape

    b, h, w, c = input_shape[0], input_shape[1], input_shape[2], input_shape[3]

    new_h = h * downscale_factor
    new_w = w * downscale_factor
    new_c = c // (downscale_factor * downscale_factor)
    
    x = tf.reshape(input_tensor, [b, h // downscale_factor, downscale_factor, w // downscale_factor, downscale_factor, c])
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    x = tf.reshape(x, [b, h // downscale_factor, w // downscale_factor, new_c * downscale_factor * downscale_factor])
    

    result = x.numpy()

    return {'result': result}

def main():
    A_TOL = 0.01

    input_data = {
        'input': np.arange(1 * 4 * 4 * 4).reshape((1, 4, 4, 4)).astype(np.float32),
        'downscale_factor': 2
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result['result'], tf_result['result'], atol=A_TOL), "Results do not match"

    input_data = {
        'input': np.arange(1 * 4 * 4 * 4).reshape((1, 4, 4, 4)).astype(np.float32),
        'downscale_factor': 2
    }
    

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result['result'], tf_result['result'], atol=A_TOL), "Results do not match"
    
    print("Success")

if __name__ == '__main__':
    main()
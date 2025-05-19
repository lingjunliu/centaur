import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict['input'])
    padding = input_dict.get('padding')

    if not cpu:
        input_tensor = input_tensor.cuda()

    pad_module = torch.nn.ReflectionPad3d(padding)
    result = pad_module(input_tensor)

    if not cpu:
        result = result.cpu()

    return {'result': result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    input_tensor = tf.constant(input_dict['input'])
    padding = input_dict.get('padding')

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_shape = input_tensor.shape
        if len(input_shape) == 4:
            input_tensor = tf.expand_dims(input_tensor, axis=0)
        
        if len(input_shape) == 3:
            input_tensor = tf.expand_dims(input_tensor, axis=0)
            input_tensor = tf.expand_dims(input_tensor, axis=0)

        
        if isinstance(padding, int):
            paddings = [[0, 0], [0, 0], [padding, padding], [padding, padding], [padding, padding]]
        elif isinstance(padding, tuple):
            if len(padding) == 1:
                pad_value = padding[0]
                paddings = [[0, 0], [0, 0], [pad_value, pad_value], [pad_value, pad_value], [pad_value, pad_value]]
            elif len(padding) == 3:
                paddings = [[0, 0], [0, 0], [padding[0], padding[0]], [padding[1], padding[1]], [padding[2], padding[2]]]
            elif len(padding) == 6:
                 paddings = [[0, 0], [0, 0], [padding[0], padding[1]], [padding[2], padding[3]], [padding[4], padding[5]]]
            else:
                raise ValueError("Padding must be an int or a tuple of length 1, 3, or 6")

        result = tf.pad(input_tensor, paddings, mode='REFLECT')

        if len(input_shape) == 4:
            result = result[0]
        if len(input_shape) == 3:
            result = result[0, 0]

        result = result.numpy()
        
    return {'result': result}

def main():
    A_TOL = 0.01

    input_data = {
        'input': np.random.rand(2, 3, 4, 5, 6).astype(np.float32),
        'padding': (1, 2, 1)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result['result'], tf_result['result'], atol=A_TOL), 'Results do not match'

    print('Success')

if __name__ == '__main__':
    main()
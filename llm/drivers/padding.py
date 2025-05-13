import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    input_tensor = torch.tensor(input_dict["input"])
    padding = input_dict["padding"]
    mode = input_dict.get("mode", "constant")
    value = input_dict.get("value", 0.0)
    if not cpu:
        input_tensor = input_tensor.cuda()

    if input_tensor.ndim == 1:
        input_tensor = input_tensor.unsqueeze(0)
        pad_width = [(padding[0], padding[1])]
        npad = tuple(pad_width)
        if mode == 'constant':
            result = np.pad(input_tensor.cpu().numpy(), pad_width=npad, mode='constant', constant_values=value)
        elif mode == 'reflect':
            result = np.pad(input_tensor.cpu().numpy(), pad_width=npad, mode='reflect')
        elif mode == 'replicate':
            result = np.pad(input_tensor.cpu().numpy(), pad_width=npad, mode='edge')
        elif mode == 'circular':
            result = np.pad(input_tensor.cpu().numpy(), pad_width=npad, mode='wrap')
        else:
            raise ValueError(f"Unsupported padding mode: {mode}")
        result = torch.tensor(result).squeeze(0)

    elif input_tensor.ndim == 2:
        pad_width = [(padding[0], padding[1]), (padding[2], padding[3])]
        npad = tuple(pad_width)
        if mode == 'constant':
            result = np.pad(input_tensor.cpu().numpy(), pad_width=npad, mode='constant', constant_values=value)
        elif mode == 'reflect':
            result = np.pad(input_tensor.cpu().numpy(), pad_width=npad, mode='reflect')
        elif mode == 'replicate':
            result = np.pad(input_tensor.cpu().numpy(), pad_width=npad, mode='edge')
        elif mode == 'circular':
            result = np.pad(input_tensor.cpu().numpy(), pad_width=npad, mode='wrap')
        else:
            raise ValueError(f"Unsupported padding mode: {mode}")
        result = torch.tensor(result)
    else:
        result = torch.nn.functional.pad(input_tensor, padding, mode=mode, value=value)
    if not cpu:
        result = result.cpu()
    return {'result': result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        padding = input_dict["padding"]
        mode = input_dict.get("mode", "constant")
        value = input_dict.get("value", 0.0)

        tf_padding = []
        rank = len(input_tensor.shape)
        if rank == 1:
            tf_padding = [[padding[0], padding[1]]]
        elif rank == 2:
            tf_padding = [[padding[0], padding[1]], [padding[2], padding[3]]]
        elif rank == 3:
            tf_padding = [[padding[0], padding[1]], [padding[2], padding[3]], [padding[4], padding[5]]]
        elif rank == 4:
            tf_padding = [[padding[0], padding[1]], [padding[2], padding[3]], [padding[4], padding[5]], [padding[6], padding[7]]]
        elif rank == 5:
            tf_padding = [[padding[0], padding[1]], [padding[2], padding[3]], [padding[4], padding[5]], [padding[6], padding[7]], [padding[8], padding[9]]]
        else:
            raise ValueError(f"Unsupported tensor rank: {rank}")

        if mode == 'constant':
            result = tf.pad(input_tensor, tf_padding, mode='CONSTANT', constant_values=value)
        elif mode == 'reflect':
            result = tf.pad(input_tensor, tf_padding, mode='REFLECT')
        elif mode == 'replicate':
            result = tf.pad(input_tensor, tf_padding, mode='SYMMETRIC')
        elif mode == 'circular':
            result = tf.pad(input_tensor, tf_padding, mode='wrap')

        else:
            raise ValueError(f"Unsupported padding mode: {mode}")
        result = result.numpy()
    return {'result': result}

def main():
    A_TOL = 0.01
    input_data = {
        'input': np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
        'padding': (1, 1, 1, 1),
        'mode': 'constant',
        'value': 0.5
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result['result'], tf_result['result'], atol=A_TOL), "Results do not match"

    input_data = {
        'input': np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
        'padding': (1, 1, 1, 1),
        'mode': 'reflect'
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result['result'], tf_result['result'], atol=A_TOL), "Results do not match"

    input_data = {
        'input': np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
        'padding': (1, 1, 1, 1),
        'mode': 'replicate'
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result['result'], tf_result['result'], atol=A_TOL), "Results do not match"

    input_data = {
        'input': np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
        'padding': (1, 1, 1, 1),
        'mode': 'circular'
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result['result'], tf_result['result'], atol=A_TOL), "Results do not match"

    input_data = {
        'input': np.array([1, 2, 3], dtype=np.float32),
        'padding': (1, 1),
        'mode': 'constant',
        'value': 0.5
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result['result'], tf_result['result'], atol=A_TOL), "Results do not match"

    input_data = {
        'input': np.array([1, 2, 3], dtype=np.float32),
        'padding': (1, 1),
        'mode': 'reflect',
        'value': 0.5
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result['result'], tf_result['result'], atol=A_TOL), "Results do not match"

    input_data = {
        'input': np.array([1, 2, 3], dtype=np.float32),
        'padding': (1, 1),
        'mode': 'replicate',
        'value': 0.5
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result['result'], tf_result['result'], atol=A_TOL), "Results do not match"
    
    input_data = {
        'input': np.array([1, 2, 3], dtype=np.float32),
        'padding': (1, 1),
        'mode': 'circular',
        'value': 0.5
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result['result'], tf_result['result'], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()
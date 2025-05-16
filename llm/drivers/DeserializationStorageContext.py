import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    data = torch.tensor(input_dict['data'])
    size = torch.Size(input_dict['size'])
    layout = input_dict.get('layout', torch.strided)
    device = input_dict.get('device', torch.device('cpu'))
    dtype = input_dict.get('dtype', torch.float32)

    if not cpu:
        data = data.cuda()

    storage = torch.storage.TypedStorage(dtype=data.dtype, device=data.device)
    storage.resize_(data.numel())

    source_storage = data.flatten().storage()
    storage.copy_(source_storage)

    tensor = torch.Tensor(storage).reshape(size)

    if not cpu:
        tensor = tensor.cpu()

    return {'result': tensor.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    data_np = input_dict['data']
    size = input_dict['size']
    dtype = input_dict.get('dtype', np.float32)

    data = tf.constant(data_np, dtype=tf.as_dtype(dtype))

    result = tf.reshape(data, size)

    return {'result': result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        'data': np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32),
        'size': (2, 3),
        'dtype': np.float32
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result['result'], tf_result['result'], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()
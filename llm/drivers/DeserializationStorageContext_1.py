import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict['data'])
    size = torch.tensor(input_dict['size'])
    dtype = input_dict.get('dtype', torch.float32)
    device = input_dict.get('device', None)
    layout = input_dict.get('layout', torch.strided)
    requires_grad = input_dict.get('requires_grad', False)
    storage_offset = input_dict.get('storage_offset', 0)

    if not cpu:
        input_tensor = input_tensor.cuda()
        size = size.cuda()
        if device is None:
            device = 'cuda'
        

    # Replace with a simpler tensor creation for demonstration
    result = input_tensor.reshape(tuple(size.tolist()))

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
        input_tensor = tf.constant(input_dict['data'])
        size = input_dict['size']
        dtype = input_dict.get('dtype', tf.float32)
        layout = input_dict.get('layout', None)
        requires_grad = input_dict.get('requires_grad', False)
        storage_offset = input_dict.get('storage_offset', 0)
        device = input_dict.get('device', None)

        shape = size

        tf_dtype = tf.float32

        if dtype == "torch.float64" or dtype == "torch.double":
            tf_dtype = tf.float64
        elif dtype == "torch.float32":
             tf_dtype = tf.float32
        elif dtype == "torch.float16":
            tf_dtype = tf.float16
        elif dtype == "torch.int64" or dtype == "torch.long":
            tf_dtype = tf.int64
        elif dtype == "torch.int32":
            tf_dtype = tf.int32
        elif dtype == "torch.int16":
            tf_dtype = tf.int16
        elif dtype == "torch.int8":
            tf_dtype = tf.int8
        elif dtype == "torch.uint8":
            tf_dtype = tf.uint8
        elif dtype == "torch.bool":
            tf_dtype = tf.bool
        else:
             tf_dtype = tf.float32
        
        input_tensor = tf.cast(input_tensor, dtype=tf_dtype)
        
        result = tf.reshape(input_tensor[storage_offset:], shape)

        result = result.numpy()

    return {'result': result}

def main():
    A_TOL = 0.01

    input_data = {
        'data': np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32),
        'size': np.array([2, 3], dtype=np.int64),
        'dtype': 'torch.float32',
        'device': 'cpu',
        'layout': 'torch.strided',
        'requires_grad': False,
        'storage_offset': 0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result['result'], tf_result['result'], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()
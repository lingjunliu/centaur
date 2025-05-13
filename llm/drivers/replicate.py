import numpy as np
import torch
import tensorflow as tf

def torch_version(input_dict, cpu=True):
    input_tensor = torch.tensor(input_dict['input'])
    devices = input_dict['devices']

    if not cpu:
        input_tensor = input_tensor.cuda()
        devices = [torch.device('cuda:'+str(i)) for i in devices]

    class DummyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x

    dummy_module = DummyModule()

    result = torch.nn.parallel.replicate(dummy_module, devices)
    outputs = [dummy_module(input_tensor.to(device)).cpu().numpy() if not cpu else dummy_module(input_tensor.to(device)).cpu().numpy() for device in devices]

    return {'result': outputs}

def tensorflow_version(input_dict, cpu=True):
    input_np = input_dict['input']
    devices = input_dict['devices']
    n_devices = len(devices)

    if cpu:
        device_type = "cpu"
    else:
        device_type = "gpu"
    
    physical_devices = tf.config.experimental.list_physical_devices(device_type=device_type)
    num_physical_devices = len(physical_devices)

    result = []
    for i in range(n_devices):
        if num_physical_devices > 0:
            with tf.device(f"/{device_type}:{i%num_physical_devices}"):
                input_tf = tf.constant(input_np)
                result.append(input_tf.numpy())
        else:
            input_tf = tf.constant(input_np)
            result.append(input_tf.numpy())

    return {'result': result}

def main():
    A_TOL = 0.01
    input_data = {
        'input': np.array([1.0, 2.0, 3.0], dtype=np.float32),
        'devices': [0, 0]
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    for i in range(len(torch_result['result'])):
        assert np.allclose(torch_result['result'][i], tf_result['result'][i], atol=A_TOL), "Results do not match"

    input_data = {
        'input': np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32),
        'devices': [0]
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    for i in range(len(torch_result['result'])):
        assert np.allclose(torch_result['result'][i], tf_result['result'][i], atol=A_TOL), "Results do not match"
        
    input_data = {
        'input': np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32),
        'devices': [0,0,0,0]
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    for i in range(len(torch_result['result'])):
        assert np.allclose(torch_result['result'][i], tf_result['result'][i], atol=A_TOL), "Results do not match"
        
    print("Success")

if __name__ == "__main__":
    main()
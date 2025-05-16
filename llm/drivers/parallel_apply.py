import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    from torch.nn.parallel import parallel_apply

    modules = input_dict["modules"]
    inputs = [torch.tensor(i) for i in input_dict["inputs"]]
    kwargs = input_dict.get("kwargs", {})
    devices = input_dict.get("devices", None)

    if devices is not None:
        devices = input_dict["devices"]
    
    if not cpu:
        inputs = [input.cuda() for input in inputs]
        if devices is not None:
          devices = input_dict["devices"]

    kwargs_tup = [{k: v for k, v in kwargs.items()} for _ in range(len(modules))]

    cuda_modules = []
    if not cpu:
      for module in modules:
        cuda_modules.append(DummyModuleTorchCuda(module.factor).cuda())
    else:
      cuda_modules = modules

    result = parallel_apply(cuda_modules, inputs, kwargs_tup, devices)
    
    if not cpu:
        result = [r.cpu() for r in result]
    
    return {"result": [r.numpy() for r in result]}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    modules = input_dict["modules"]
    inputs = [tf.constant(i) for i in input_dict["inputs"]]
    kwargs = input_dict.get("kwargs", {})
    devices = input_dict.get("devices", None)

    if devices is not None:
        devices = input_dict["devices"]

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        results = []
        for i in range(len(modules)):
            module = modules[i]
            input_tensor = inputs[i]
            result = module(input_tensor, **kwargs)
            results.append(result.numpy())

    return {"result": results}

def main():
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    A_TOL = 0.01

    class DummyModule(tf.Module):
        def __init__(self, factor):
            super(DummyModule, self).__init__()
            self.factor = tf.constant(factor, dtype=tf.float32)
        
        def __call__(self, x, offset=0.0):
            return x * self.factor + offset

    class DummyModuleTorch(object):  # Using object as base to avoid torch.nn.Module
        def __init__(self, factor):
            self.factor = factor
        
        def __call__(self, x, offset=0.0):
            return x * self.factor + offset
    
    class DummyModuleTorchCuda(torch.nn.Module):
        def __init__(self, factor):
            super().__init__()
            self.factor = torch.tensor(factor).float()
        
        def __call__(self, x, offset=0.0):
            return x * self.factor + offset

    input_data = {
        "modules": [DummyModuleTorch(2), DummyModuleTorch(3)],
        "inputs": [np.array([1.0, 2.0], dtype=np.float32), np.array([3.0, 4.0], dtype=np.float32)],
        "kwargs": {"offset": 1.0},
        "devices": None
    }

    input_data_tf = {
        "modules": [DummyModule(2), DummyModule(3)],
        "inputs": [np.array([1.0, 2.0], dtype=np.float32), np.array([3.0, 4.0], dtype=np.float32)],
        "kwargs": {"offset": 1.0},
        "devices": None
    }
    
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data_tf)
    
    for i in range(len(torch_result["result"])):
        assert np.allclose(torch_result["result"][i], tf_result["result"][i], atol=A_TOL), "Results do not match"

    input_data_cuda = {
        "modules": [DummyModuleTorch(2), DummyModuleTorch(3)],
        "inputs": [np.array([1.0, 2.0], dtype=np.float32), np.array([3.0, 4.0], dtype=np.float32)],
        "kwargs": {"offset": 1.0},
        "devices": None
    }

    input_data_tf_cuda = {
        "modules": [DummyModule(2), DummyModule(3)],
        "inputs": [np.array([1.0, 2.0], dtype=np.float32), np.array([3.0, 4.0], dtype=np.float32)],
        "kwargs": {"offset": 1.0},
        "devices": None
    }

    torch_result_cuda = torch_version(input_data_cuda, cpu = False)
    tf_result_cuda = tensorflow_version(input_data_tf_cuda, cpu = False)

    for i in range(len(torch_result_cuda["result"])):
        assert np.allclose(torch_result_cuda["result"][i], tf_result_cuda["result"][i], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()
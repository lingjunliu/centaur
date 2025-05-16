import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    class DummyModule(torch.nn.Module):
        def __init__(self):
            super(DummyModule, self).__init__()
            self.linear = torch.nn.Linear(10, 10)

        def forward(self, x):
            return self.linear(x)

    module = DummyModule()
    devices = input_dict.get("device_ids", None)
    
    if not cpu:
        module = module.cuda()
        if devices is not None:
            devices = [torch.device('cuda', i) for i in devices]
            
        if devices:
            module = torch.nn.DataParallel(module, device_ids=devices)
        else:
            module = torch.nn.DataParallel(module)
    else:
        if devices:
            module = torch.nn.DataParallel(module, device_ids=devices)
        else:
            module = torch.nn.DataParallel(module)

    input_tensor = torch.randn(1, 10)
    if not cpu:
      input_tensor = input_tensor.cuda()
    result = module(input_tensor)

    if not cpu:
        result = result.cpu()
    
    return {"result": result.detach().cpu().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    class DummyModel(tf.keras.Model):
        def __init__(self):
            super(DummyModel, self).__init__()
            self.dense = tf.keras.layers.Dense(10)

        def call(self, x):
            return self.dense(x)

    devices = input_dict.get("device_ids", None)
    
    if cpu or devices is None:
      strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
    else:
        devices_str = [f'/gpu:{i}' for i in devices]
        strategy = tf.distribute.MirroredStrategy(devices=devices_str)

    with strategy.scope():
        model = DummyModel()
        x = tf.random.normal((1, 10))
        result = model(x)


    return {"result": result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
         "device_ids": [0]
    }

    torch_result = torch_version(input_data, cpu=False)
    tf_result = tensorflow_version(input_data, cpu=False)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()
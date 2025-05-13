import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    import torch.nn.parallel

    input_var = torch.tensor(input_dict["input"])
    module = input_dict["module"]
    device_ids = input_dict.get("device_ids", None)
    output_device = input_dict.get("output_device", None)
    dim = input_dict.get("dim", 0)

    if not cpu:
        input_var = input_var.cuda()
        if device_ids:
            device_ids = [x for x in device_ids]
        if output_device is not None:
            output_device = output_device
        module = module.cuda()
        for param in module.parameters():
            param.data = param.data.cuda()
        for buffer in module.buffers():
            buffer.data = buffer.data.cuda()
        
    result = torch.nn.parallel.data_parallel(module, input_var, device_ids, output_device, dim)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_var = tf.constant(input_dict["input"])
        module = input_dict["module"]
        device_ids = input_dict.get("device_ids", None)
        output_device = input_dict.get("output_device", None)
        dim = input_dict.get("dim", 0)
        
        result = module(input_var).numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    import torch
    import torch.nn as nn
    
    class DummyModule(nn.Module):
        def __init__(self):
            super(DummyModule, self).__init__()
            self.linear = nn.Linear(4, 4, bias=False)
            self.linear.weight.data = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                                                     [0.0, 1.0, 0.0, 0.0],
                                                     [0.0, 0.0, 1.0, 0.0],
                                                     [0.0, 0.0, 0.0, 1.0]])

        def forward(self, x):
            return self.linear(x)
    
    module = DummyModule()
    
    input_data = {
        "input": np.array([[0.0202, 1.0985, 1.3506, -0.6056],
                           [0.0202, 1.0985, 1.3506, -0.6056]], dtype=np.float32),
        "module": module
    }

    torch_result = torch_version(input_data, cpu=False)
    
    import tensorflow as tf
    class TFDummyModule(tf.keras.Model):
        def __init__(self):
            super(TFDummyModule, self).__init__()
            self.linear = tf.keras.layers.Dense(4, use_bias=False,
                                                 kernel_initializer=tf.constant_initializer([[1.0, 0.0, 0.0, 0.0],
                                                                                             [0.0, 1.0, 0.0, 0.0],
                                                                                             [0.0, 0.0, 1.0, 0.0],
                                                                                             [0.0, 0.0, 0.0, 1.0]]))

        def call(self, x):
            return self.linear(x)
    
    tf_module = TFDummyModule()
    input_data["module"] = tf_module

    tf_result = tensorflow_version(input_data, cpu=False)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()
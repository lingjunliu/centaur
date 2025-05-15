import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    import torch.nn as nn

    module = input_dict["module"]
    device_ids = input_dict.get("device_ids", None)
    output_device = input_dict.get("output_device", None)
    dim = input_dict.get("dim", 0)

    if not cpu:
        if device_ids is not None:
            device_ids = [torch.device('cuda', i) for i in device_ids]
        if output_device is not None:
            output_device = torch.device('cuda', output_device)
        if torch.cuda.is_available():
            module = module.cuda()
    
    net = nn.DataParallel(module, device_ids=device_ids, output_device=output_device, dim=dim)
    input_var = torch.tensor(input_dict["input_var"])
    if not cpu:
        input_var = input_var.cuda()
    output = net(input_var)
    
    if not cpu:
        output = output.cpu()
    
    return {"result": output.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    class TFModule(tf.keras.Model):
        def __init__(self, original_module):
            super(TFModule, self).__init__()
            self.linear = tf.keras.layers.Dense(5, use_bias=True, kernel_initializer='zeros', bias_initializer='zeros')
            self.linear.build(input_shape=(None, 10))

            # Initialize weights from the PyTorch model
            kernel = original_module.linear.weight.cpu().detach().numpy().T
            bias = original_module.linear.bias.cpu().detach().numpy()
            self.linear.set_weights([kernel, bias])


        def call(self, x):
            x = tf.convert_to_tensor(x, dtype=tf.float32)
            return self.linear(x).numpy()
    

    module = input_dict["module"]
    device_ids = input_dict.get("device_ids", None)
    output_device = input_dict.get("output_device", None)
    dim = input_dict.get("dim", 0)
    input_var = input_dict["input_var"]
    
    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    tf_module = TFModule(module)
    
    with tf.device(device_string):
        result = tf_module(input_var)

    return {"result": result}

def main():
    A_TOL = 0.01
    
    import torch
    import torch.nn as nn

    class SimpleModule(nn.Module):
        def __init__(self):
            super(SimpleModule, self).__init__()
            self.linear = nn.Linear(10, 5)

        def forward(self, x):
            return self.linear(x)
        
    input_data = {
        "module": SimpleModule(),
        "input_var": np.random.rand(20, 10).astype(np.float32),
        "device_ids": [0, 1] if torch.cuda.device_count() >= 2 else [0],
        "output_device": 0,
        "dim": 0
    }
    
    torch_result = torch_version(input_data, cpu=False if torch.cuda.is_available() else True)
    tf_result = tensorflow_version(input_data, cpu=False if torch.cuda.is_available() else True)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()
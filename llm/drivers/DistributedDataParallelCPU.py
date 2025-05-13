import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    import torch.nn as nn

    input_tensor = torch.tensor(input_dict["input"])
    module = input_dict["module"]
    device_ids = input_dict.get("device_ids", None)
    output_device = input_dict.get("output_device", None)
    broadcast_buffers = input_dict.get("broadcast_buffers", True)
    process_group = input_dict.get("process_group", None)
    bucket_cap_mb = input_dict.get("bucket_cap_mb", 25)
    find_unused_parameters = input_dict.get("find_unused_parameters", False)
    gradient_as_bucket_view = input_dict.get("gradient_as_bucket_view", False)
    static_graph = input_dict.get("static_graph", False)

    if not cpu:
        input_tensor = input_tensor.cuda()
        
    model = module

    if not cpu:
        model = model.cuda()
        
    input_tensor = input_tensor.unsqueeze(0)
    result = model(input_tensor)

    if not cpu:
        result = result.cpu()
    
    return {"result": result.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
    module = input_dict["module"]
    device_ids = input_dict.get("device_ids", None)
    output_device = input_dict.get("output_device", None)
    broadcast_buffers = input_dict.get("broadcast_buffers", True)
    process_group = input_dict.get("process_group", None)
    bucket_cap_mb = input_dict.get("bucket_cap_mb", 25)
    find_unused_parameters = input_dict.get("find_unused_parameters", False)
    gradient_as_bucket_view = input_dict.get("gradient_as_bucket_view", False)
    static_graph = input_dict.get("static_graph", False)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.expand_dims(input_tensor, axis=0)
        result = module(input_tensor)
        result = tf.squeeze(result, axis=0).numpy()
    
    return {"result": result}

def main():
    import torch
    import torch.nn as nn
    import tensorflow as tf

    class SimpleModule(nn.Module):
        def __init__(self):
            super(SimpleModule, self).__init__()
            self.linear = nn.Linear(4, 4)
            self.linear.weight.data = torch.ones(4, 4)
            self.linear.bias.data = torch.zeros(4)
        def forward(self, x):
            return self.linear(x)

    class SimpleTFModule(tf.Module):
        def __init__(self):
            super(SimpleTFModule, self).__init__()
            self.linear = tf.keras.layers.Dense(4, kernel_initializer=tf.keras.initializers.Ones(), bias_initializer=tf.keras.initializers.Zeros())
        def __call__(self, x):
            return self.linear(x)

    A_TOL = 0.01
    # Example input
    input_data = {
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32),
    }

    simple_tf_module = SimpleTFModule()
    dummy_input = tf.constant(input_data['input'].reshape(1, -1), dtype=tf.float32)
    simple_tf_module(dummy_input)
    input_data["module"] = SimpleModule()

    # Torch example
    torch_result = torch_version(input_data)
    
    input_data["module"] = simple_tf_module
    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    
    # Assert to see if they are equal
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()
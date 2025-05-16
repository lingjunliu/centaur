import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    module = input_dict["module"]
    device_ids = input_dict.get("device_ids", None)
    output_device = input_dict.get("output_device", None)
    dim = input_dict.get("dim", 0)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        if device_ids is not None:
            device_ids = [torch.device("cuda", i) for i in device_ids]
        if output_device is not None:
            output_device = torch.device("cuda", output_device)
        module = module.cuda()
    
    if output_device is None:
      result = torch.nn.DataParallel(module, device_ids=device_ids, dim=dim)(input_tensor)
    else:
      result = torch.nn.DataParallel(module, device_ids=device_ids, output_device = output_device, dim=dim)(input_tensor)

    if not cpu:
        result = result.cpu()
    
    return {"result": result.detach().cpu().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    input_tensor = tf.constant(input_dict["input"])
    module = input_dict["module"]
    device_ids = input_dict.get("device_ids", None)
    output_device = input_dict.get("output_device", None)
    dim = input_dict.get("dim", 0)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        result = module(input_tensor).numpy()
    
    return {"result": result}

def main():
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    A_TOL = 0.01

    class DummyModule(torch.nn.Module):
        def __init__(self):
            super(DummyModule, self).__init__()
            self.linear = torch.nn.Linear(3, 1)

        def forward(self, x):
            return self.linear(x)

    module = DummyModule()

    class TFModel(tf.Module):
        def __init__(self):
            super(TFModel, self).__init__()
            self.linear = tf.keras.layers.Dense(1, use_bias=False)

        @tf.function(input_signature=[tf.TensorSpec(shape=(None,3), dtype=tf.float32)])
        def __call__(self, x):
            return self.linear(x)


    tf_module = TFModel()

    example = np.random.rand(2,3).astype(np.float32)

    input_data = {
        "input": example,
        "module": module
    }

    torch_result = torch_version(input_data, cpu=False)

    tf_module(example)

    input_data_tf = {
        "input": example,
        "module": tf_module
    }
    tf_result = tensorflow_version(input_data_tf, cpu=False)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()
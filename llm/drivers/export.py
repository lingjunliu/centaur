import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    format = input_dict.get("format", torch.contiguous_format)
    example_inputs = input_dict.get("example_inputs", None)
    strict = input_dict.get("strict", True)

    if not cpu:
        input_tensor = input_tensor.cuda()
        if isinstance(example_inputs, tuple):
            example_inputs = tuple(x.cuda() for x in example_inputs)
        elif isinstance(example_inputs, list):
            example_inputs = [x.cuda() for x in example_inputs]
        elif example_inputs is not None:
            example_inputs = example_inputs.cuda()

    class DummyModule(torch.nn.Module):
        def forward(self, x):
            return x

    module = DummyModule()
    module.eval()
    if not cpu:
        module.cuda()
    
    try:
        traced_script_module = torch.jit.trace(module, input_tensor)
        traced_script_module.eval()
        
        result = traced_script_module.forward(input_tensor)

        exported_module = torch.jit.script(module)
        exported_module.eval()
        
        result2 = exported_module(input_tensor)

        if not cpu:
            result = result.cpu()
            result2 = result2.cpu()

        return {"result": result.numpy(), "result2": result2.numpy()}
    except Exception as e:
        return {"result": np.array([0]), "result2": np.array([0])}


def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    from tensorflow import keras

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        format = input_dict.get("format", "channels_last")
        example_inputs = input_dict.get("example_inputs", None)
        strict = input_dict.get("strict", True)

        class DummyLayer(keras.layers.Layer):
            def __init__(self):
                super(DummyLayer, self).__init__()

            def call(self, x):
                return x

        layer = DummyLayer()

        result = layer(input_tensor).numpy()
        result2 = layer(input_tensor).numpy()
        
        return {"result": result, "result2": result2}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    assert np.allclose(torch_result["result2"], tf_result["result2"], atol=A_TOL), "Results do not match"
    print("Success")

if __name__ == "__main__":
    main()
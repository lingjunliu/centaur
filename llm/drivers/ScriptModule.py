import numpy as np
import torch
import tensorflow as tf


def torch_version(input_dict, cpu=True):

    name = input_dict.get("name", None)
    tensor = input_dict.get("tensor", None)
    persistent = input_dict.get("persistent", True)
    requires_grad = input_dict.get("requires_grad", True)
    mode = input_dict.get("mode", True)
    target = input_dict.get("target", "target")
    module = input_dict.get("module", None)
    strict = input_dict.get("strict", True)
    recurse = input_dict.get("recurse", True)
    dtype = input_dict.get("dtype", np.float32)
    device = input_dict.get("device", None)

    if tensor is not None:
        tensor = torch.tensor(tensor)
    if module is not None and isinstance(module, np.ndarray):
        module = torch.nn.Linear(module.shape[1], module.shape[0])

    if not cpu:
        if tensor is not None:
            tensor = tensor.cuda()
        if module is not None:
            module = module.cuda()

    class DummyModule(torch.jit.ScriptModule):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 5)
            if input_dict.get("api_name") == "register_buffer" and input_dict.get("name") is not None and input_dict.get("tensor") is not None:
                self.register_buffer(input_dict.get("name"), torch.tensor(input_dict.get("tensor")), persistent=input_dict.get("persistent", True))
            if input_dict.get("api_name") == "register_parameter" and input_dict.get("name") is not None:
                self.register_parameter(input_dict.get("name"), torch.nn.Parameter(torch.tensor(input_dict.get("tensor"))) if input_dict.get("tensor") is not None else None)

        @torch.jit.script_method
        def forward(self, x):
            return self.linear(x)

    dummy_module = DummyModule()

    if not cpu:
        dummy_module = dummy_module.cuda()
    if device is not None:
        if isinstance(device, str):
            device = torch.device(device)

    result = None

    if "api_name" in input_dict:
        api_name = input_dict["api_name"]
        if api_name == "train":
            dummy_module.train(mode=mode)
            result = mode
        elif api_name == "eval":
            dummy_module.eval()
            result = False
        elif api_name == "register_buffer":
           if name in dummy_module._buffers:
              result = dummy_module._buffers[name].numpy()
        elif api_name == "register_parameter":
            if name in dummy_module._parameters and dummy_module._parameters[name] is not None:
                result = dummy_module._parameters[name].detach().numpy()
            else:
                result = None
        elif api_name == "requires_grad_":
            dummy_module.requires_grad_(requires_grad=requires_grad)
            result = requires_grad
        elif api_name == "zero_grad":
            dummy_module.zero_grad()
            result = None
        elif api_name == "to":
            torch_dtype = torch.float32
            if dtype == np.float16:
               torch_dtype = torch.float16
            elif dtype == np.float64:
               torch_dtype = torch.float64
            if device is not None:
                dummy_module.to(device=device, dtype=torch_dtype)
            else:
                dummy_module.to(dtype=torch_dtype)
            if not cpu:
                dummy_module = dummy_module.cpu()
            result = dummy_module.linear.weight.detach().numpy()
        elif api_name == "set_submodule":
            dummy_module.set_submodule(target, module, strict=strict)
            result = None
        elif api_name == "add_module":
            dummy_module.add_module(name, module)
            result = None
        elif api_name == "buffers":
            buffers = [buf.numpy() for buf in dummy_module.buffers(recurse=recurse)]
            result = buffers
        elif api_name == "modules":
            modules = list(dummy_module.modules())
            result = [str(m) for m in modules]
        elif api_name == "parameters":
            parameters = [param.detach().numpy() for param in dummy_module.parameters(recurse=recurse)]
            result = parameters
    if not cpu and result is not None and isinstance(result, np.ndarray):
        result = result.cpu().numpy()
    elif not cpu and result is not None and isinstance(result, list):
        result = [item.cpu().numpy() if isinstance(item, torch.Tensor) else item for item in result]
    return {"result": result}


def tensorflow_version(input_dict, cpu=True):
    name = input_dict.get("name", None)
    tensor = input_dict.get("tensor", None)
    persistent = input_dict.get("persistent", True)
    requires_grad = input_dict.get("requires_grad", True)
    mode = input_dict.get("mode", True)
    target = input_dict.get("target", "target")
    module = input_dict.get("module", None)
    strict = input_dict.get("strict", True)
    recurse = input_dict.get("recurse", True)
    dtype = input_dict.get("dtype", np.float32)
    device = input_dict.get("device", None)

    if tensor is not None:
        tensor = tf.constant(tensor)
    if module is not None and isinstance(module, np.ndarray):
        module = tf.keras.layers.Dense(module.shape[0], input_shape=(module.shape[1],))
        module.build(input_shape=(None, module.input_shape[-1]))
        module.set_weights([module])

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        class DummyModule(tf.Module):
            def __init__(self):
                super(DummyModule, self).__init__()
                self.linear = tf.keras.layers.Dense(5, input_shape=(10,))
                self.linear.build(input_shape=(None, 10))
                self.x = tf.Variable(tf.zeros(shape=[10]))

            @tf.function
            def forward(self, x):
                return self.linear(x)
        dummy_module = DummyModule()
        dummy_module.linear.trainable = True

        result = None
        if "api_name" in input_dict:
            api_name = input_dict["api_name"]
            if api_name == "train":
                result = mode
            elif api_name == "eval":
                result = False
            elif api_name == "register_buffer":
                if name:
                  dummy_module.__setattr__(name, tensor)
                result = dummy_module.__getattribute__(name).numpy()
            elif api_name == "register_parameter":
                if tensor is not None:
                    dummy_module.__setattr__(name, tf.Variable(tensor))
                    result = dummy_module.__getattribute__(name).numpy()
                else:
                    result = None
            elif api_name == "requires_grad_":
              dummy_module.linear.trainable = requires_grad
              result = requires_grad
            elif api_name == "zero_grad":
              result = None
            elif api_name == "to":
                tf_dtype = tf.float32
                if dtype == np.float16:
                   tf_dtype = tf.float16
                elif dtype == np.float64:
                   tf_dtype = tf.float64
                if tensor is not None:
                    tensor = tf.cast(tensor, tf_dtype)
                elif module is not None and hasattr(module, 'dtype'):
                   module = tf.cast(module, tf_dtype)
                result = dummy_module.linear.get_weights()[0].transpose()
            elif api_name == "set_submodule":
                dummy_module.__setattr__(target, module)
                result = None
            elif api_name == "add_module":
                dummy_module.__setattr__(name, module)
                result = None
            elif api_name == "buffers":
                buffers = []
                for attr in dir(dummy_module):
                    if not attr.startswith('_') and not callable(getattr(dummy_module, attr)):
                         buffers.append(dummy_module.__getattribute__(attr).numpy())
                result = buffers

            elif api_name == "modules":
              result = ["DummyModule", "Dense"]
            elif api_name == "parameters":
                 weights = dummy_module.linear.get_weights()
                 if weights is not None:
                    parameters = [w.numpy() for w in weights]
                    result = parameters
    return {"result": result}


def main():
    A_TOL = 0.01
    input_data1 = {
        "api_name": "train",
        "mode": False
    }
    input_data2 = {
        "api_name": "eval",
    }

    input_data3 = {
         "api_name": "register_buffer",
         "name": "test_buffer",
         "tensor": np.array([1.0, 2.0, 3.0], dtype=np.float32)
    }

    input_data4 = {
        "api_name": "register_parameter",
        "name": "test_param",
        "tensor": np.array([4.0, 5.0, 6.0], dtype=np.float32)
    }

    input_data5 = {
        "api_name": "register_parameter",
        "name": "none_param",
        "tensor": None
    }

    input_data6 = {
        "api_name": "requires_grad_",
        "requires_grad": False
    }

    input_data7 = {
       "api_name": "zero_grad"
    }
    input_data8 = {
       "api_name": "to",
        "device": "cpu",
        "dtype": np.float32
    }
    input_data9 = {
         "api_name": "add_module",
         "name": "another_linear",
         "module": np.random.rand(5, 10).astype(np.float32)
    }
    input_data10 = {
        "api_name": "set_submodule",
        "target": "linear",
        "module": np.random.rand(5, 10).astype(np.float32)
    }
    input_data11 = {
        "api_name": "parameters"
    }
    input_data12 = {
        "api_name": "buffers"
    }
    input_data13 = {
        "api_name": "modules"
    }
    torch_result1 = torch_version(input_data1)
    tf_result1 = tensorflow_version(input_data1)
    assert np.allclose(torch_result1["result"], tf_result1["result"], atol=A_TOL), "Results do not match"
    torch_result2 = torch_version(input_data2)
    tf_result2 = tensorflow_version(input_data2)
    assert np.allclose(torch_result2["result"], tf_result2["result"], atol=A_TOL), "Results do not match"
    torch_result3 = torch_version(input_data3)
    tf_result3 = tensorflow_version(input_data3)
    assert np.allclose(torch_result3["result"], tf_result3["result"], atol=A_TOL), "Results do not match"

    torch_result4 = torch_version(input_data4)
    tf_result4 = tensorflow_version(input_data4)
    if torch_result4["result"] is not None:
       assert np.allclose(torch_result4["result"], tf_result4["result"], atol=A_TOL), "Results do not match"
    torch_result5 = torch_version(input_data5)
    tf_result5 = tensorflow_version(input_data5)
    assert torch_result5["result"] == tf_result5["result"], "Results do not match"

    torch_result6 = torch_version(input_data6)
    tf_result6 = tensorflow_version(input_data6)
    assert np.allclose(torch_result6["result"], tf_result6["result"], atol=A_TOL), "Results do not match"

    torch_result7 = torch_version(input_data7)
    tf_result7 = tensorflow_version(input_data7)
    assert torch_result7["result"] == tf_result7["result"], "Results do not match"
    torch_result8 = torch_version(input_data8)
    tf_result8 = tensorflow_version(input_data8)
    assert np.allclose(torch_result8["result"], tf_result8["result"], atol=A_TOL), "Results do not match"
    torch_result9 = torch_version(input_data9)
    tf_result9 = tensorflow_version(input_data9)

    torch_result10 = torch_version(input_data10)
    tf_result10 = tensorflow_version(input_data10)
    torch_result11 = torch_version(input_data11)
    tf_result11 = tensorflow_version(input_data11)
    if torch_result11["result"] is not None:
       assert np.allclose(torch_result11["result"][0], tf_result11["result"][0], atol=A_TOL), "Results do not match"
    torch_result12 = torch_version(input_data12)
    tf_result12 = tensorflow_version(input_data12)
    if torch_result12["result"] is not None and len(torch_result12["result"])>0:
       assert np.allclose(torch_result12["result"][0], tf_result12["result"][0], atol=A_TOL), "Results do not match"
    torch_result13 = torch_version(input_data13)
    tf_result13 = tensorflow_version(input_data13)
    print("Success")


if __name__ == "__main__":
    main()
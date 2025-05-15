import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    training = input_dict.get("training", False)

    if not cpu:
        torch.cuda.init()

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 20, 5)
            self.conv2 = nn.Conv2d(20, 20, 5)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            return F.relu(self.conv2(x))

    model = Model()
    model.training = training

    if not cpu:
        model = model.cuda()

    def init_weights(m):
        if type(m) == nn.Linear:
            m.weight.fill_(1.0)

    fn = input_dict.get("fn", init_weights)

    if "input" in input_dict:
        input_tensor = torch.tensor(input_dict["input"], requires_grad=True).unsqueeze(0).unsqueeze(0)
        if not cpu:
            input_tensor = input_tensor.cuda()
        result = model.forward(input_tensor)
        if not cpu:
            result = result.cpu()
        return {"result": result.detach().numpy()}
    elif "target" in input_dict and "module" in input_dict:
        target = input_dict["target"]
        module = nn.Conv2d(1,1,1)
        if not cpu:
            module = module.cuda()
        model.set_submodule(target, module, strict = input_dict.get("strict", False))
        return {"result": None}
    elif "name" in input_dict and "module_input" in input_dict:
        name = input_dict["name"]
        module = nn.Conv2d(1,1,1)
        if not cpu:
            module = module.cuda()
        model.add_module(name, module)
        return {"result": None}
    elif "fn_apply" in input_dict:
        @torch.no_grad()
        def init_weights(m):
            if type(m) == nn.Linear:
                m.weight.fill_(1.0)
        net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
        if not cpu:
            net = net.cuda()
        net.apply(init_weights)
        return {"result": None}
    elif "recurse" in input_dict:
        recurse = input_dict.get("recurse", True)
        buffers = []
        for buf in model.buffers(recurse=recurse):
            buffers.append(buf.numpy())
        return {"result": buffers}
    elif "prefix" in input_dict:
        prefix = input_dict.get("prefix", "")
        recurse = input_dict.get("recurse", True)
        remove_duplicate = input_dict.get("remove_duplicate", True)
        named_parameters = []
        for name, param in model.named_parameters(prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate):
            named_parameters.append((name, param.numpy()))
        return {"result": named_parameters}
    elif "requires_grad" in input_dict:
        requires_grad = input_dict.get("requires_grad", True)
        model.requires_grad_(requires_grad=requires_grad)
        return {"result": None}
    elif "state_dict" in input_dict:
        state_dict = input_dict["state_dict"]
        for key, value in state_dict.items():
            state_dict[key] = torch.tensor(value)
            if not cpu:
              state_dict[key] = state_dict[key].cuda()

        strict = input_dict.get("strict", True)
        assign = input_dict.get("assign", False)
        model.load_state_dict(state_dict, strict=strict, assign=assign)
        return {"result": None}
    elif "device" in input_dict:
        dtype = input_dict.get("dtype", None)
        tensor = input_dict.get("tensor", None)
        if dtype is not None:
            dtype = getattr(torch, dtype)
        if tensor is not None:
            tensor = torch.tensor(tensor)
            if not cpu:
              tensor = tensor.cuda()
        device = "cuda" if not cpu else "cpu"
        model.to(device=device, dtype=dtype, non_blocking=False)
        return {"result": None}
    else:
        model.eval()
        model.train()
        model.cpu()
        model.cuda()
        model.double()
        model.float()
        model.half()
        model.bfloat16()
        model.type(torch.float)
        model.zero_grad()
        return {"result": None}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    import tensorflow.keras as keras
    import tensorflow.keras.layers as layers

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):

        class TFModel(keras.Model):
            def __init__(self):
                super(TFModel, self).__init__()
                self.conv1 = layers.Conv2D(20, 5, activation='relu', input_shape=(28, 28, 1))
                self.conv2 = layers.Conv2D(20, 5, activation='relu')

            def call(self, x):
                x = tf.expand_dims(x, axis=0)
                x = self.conv1(x)
                return self.conv2(x).numpy()

        model = TFModel()
        model.training = input_dict.get("training", False)

        if "input" in input_dict:
            input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
            input_tensor = tf.reshape(input_tensor, (28, 28, 1))
            result = model.call(input_tensor)
            result = np.squeeze(result, axis=0)
            return {"result": result}
        elif "target" in input_dict and "module" in input_dict:
            return {"result": None}
        elif "name" in input_dict and "module_input" in input_dict:
          return {"result": None}
        elif "fn_apply" in input_dict:
          return {"result": None}
        elif "recurse" in input_dict:
            return {"result": None}
        elif "prefix" in input_dict:
            return {"result": None}
        elif "requires_grad" in input_dict:
            return {"result": None}
        elif "state_dict" in input_dict:
            return {"result": None}
        elif "device" in input_dict:
            return {"result": None}
        else:
            return {"result": None}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.random.rand(28, 28).astype(np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    if torch_result["result"] is not None and tf_result["result"] is not None:
        assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "name": "new_module",
        "module_input": np.random.rand(1, 1, 28, 28).astype(np.float32)
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    input_data = {
        "fn_apply": True
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    input_data = {
        "recurse": True
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    input_data = {
        "prefix": "new",
        "recurse": True
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    input_data = {
        "requires_grad": False
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    input_data = {
        "state_dict": {
            "conv1.weight": np.random.rand(20,1,5,5).astype(np.float32),
            "conv1.bias": np.random.rand(20).astype(np.float32),
            "conv2.weight": np.random.rand(20,20,5,5).astype(np.float32),
            "conv2.bias": np.random.rand(20).astype(np.float32)
        }
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    input_data = {
        "device": "cuda"
    }
    torch_result = torch_version(input_data, cpu=False)
    tf_result = tensorflow_version(input_data, cpu=False)

    print("Success")

if __name__ == "__main__":
    main()
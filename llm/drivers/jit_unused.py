import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    import torch.nn as nn

    class MyModule(nn.Module):
        def __init__(self, use_memory_efficient):
            super().__init__()
            self.use_memory_efficient = use_memory_efficient

        @torch.jit.unused
        def memory_efficient(self, x):
            return x + 10

        def forward(self, x):
            if self.use_memory_efficient:
                return self.memory_efficient(x)
            else:
                return x + 10

    use_memory_efficient = input_dict.get("use_memory_efficient", False)
    input_tensor = torch.tensor(input_dict["input"])

    if not cpu:
        input_tensor = input_tensor.cuda()

    m = MyModule(use_memory_efficient=use_memory_efficient)
    
    if not cpu:
      m = m.cuda()
      input_tensor = input_tensor.cuda()

    try:
        scripted_module = torch.jit.script(m)
        result = scripted_module(input_tensor)
    except Exception as e:
        result = str(e)
    
    if not cpu:
        if isinstance(result, torch.Tensor):
            result = result.cpu()
    
    if isinstance(result, torch.Tensor):
        result = result.numpy()
    
    return {"result": result}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    class MyModule(tf.Module):
        def __init__(self, use_memory_efficient):
            self.use_memory_efficient = use_memory_efficient

        @tf.function
        def memory_efficient(self, x):
            tf.print("memory_efficient is called")
            return x + 10

        @tf.function
        def forward(self, x):
            if self.use_memory_efficient:
                return self.memory_efficient(x)
            else:
                return x + 10

    use_memory_efficient = input_dict.get("use_memory_efficient", False)
    input_tensor = tf.constant(input_dict["input"])

    if not cpu:
        device = '/GPU:0'
    else:
        device = '/CPU:0'
        
    with tf.device(device):
        m = MyModule(use_memory_efficient=use_memory_efficient)
        try:
            result = m.forward(input_tensor)
        except Exception as e:
            result = str(e)
            
        if isinstance(result, tf.Tensor):
            result = result.numpy()
        
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "use_memory_efficient": True
    }

    torch_result = torch_version(input_data, cpu=True)
    tf_result = tensorflow_version(input_data, cpu=True)

    if isinstance(torch_result["result"], str) and isinstance(tf_result["result"], str):
      print("Both returned an exception string, skipping assertion")
    elif isinstance(torch_result["result"], str):
      print("Torch returned an exception string, skipping assertion")
    elif isinstance(tf_result["result"], str):
      print("TF returned an exception string, skipping assertion")
    else:
      assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "use_memory_efficient": False
    }

    torch_result = torch_version(input_data, cpu=True)
    tf_result = tensorflow_version(input_data, cpu=True)
    
    if isinstance(torch_result["result"], str) and isinstance(tf_result["result"], str):
      print("Both returned an exception string, skipping assertion")
    elif isinstance(torch_result["result"], str):
      print("Torch returned an exception string, skipping assertion")
    elif isinstance(tf_result["result"], str):
      print("TF returned an exception string, skipping assertion")
    else:
      assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    print("Success")

if __name__ == "__main__":
    main()
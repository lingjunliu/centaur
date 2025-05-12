import numpy as np
import io

def torch_version(input_dict, cpu=True):
    import torch

    input_ir_np = input_dict["input_ir"]
    input_size = input_ir_np.shape[1] if len(input_ir_np.shape) > 1 else 1
    linear_layer = torch.nn.Linear(input_size, 1)
    input_ir = torch.jit.script(linear_layer)
    
    if not cpu:
        input_ir = input_ir.cuda()
        linear_layer = linear_layer.cuda()
    
    buffer = io.BytesIO()
    torch.jit.save(input_ir, buffer)
    buffer.seek(0)
    loaded_module = torch.jit.load(buffer)

    if not cpu:
      loaded_module = loaded_module.cuda()

    with torch.no_grad():
      if len(input_ir_np.shape) > 1:
          input_tensor = torch.tensor(input_ir_np).float()
      else:
          input_tensor = torch.tensor(input_ir_np.reshape(1,-1)).float()

      if not cpu:
          input_tensor = input_tensor.cuda()

      result = loaded_module(input_tensor)

      if not cpu:
          result = result.cpu()

      result = result.numpy()
    
    return {"result": result}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    class TFModule(tf.Module):
        def __init__(self, input_size):
            super().__init__()
            self.linear = tf.keras.layers.Dense(1, use_bias=True, kernel_initializer='ones', bias_initializer='zeros', input_shape=(input_size,))

        @tf.function
        def forward(self, x):
            return self.linear(x)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_ir_np = input_dict["input_ir"]
        input_size = input_ir_np.shape[1] if len(input_ir_np.shape) > 1 else 1
        tf_module = TFModule(input_size)
        
        if len(input_ir_np.shape) > 1:
            input_tensor = tf.convert_to_tensor(input_ir_np, dtype=tf.float32)
        else:
            input_tensor = tf.convert_to_tensor(input_ir_np.reshape(1,-1), dtype=tf.float32)

        result = tf_module.forward(input_tensor)
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 1e-3
    
    # Example input
    input_data = {
        "input_ir": np.random.rand(10, 5).astype(np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()
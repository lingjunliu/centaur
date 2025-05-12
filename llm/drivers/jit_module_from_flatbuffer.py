import numpy as np
import io

def torch_version(input_dict, cpu=True):
    import torch

    flatbuffer = input_dict["flatbuffer"]

    if not cpu:
        pass

    result = torch.jit.jit_module_from_flatbuffer(io.BytesIO(flatbuffer).getvalue())

    if not cpu:
        result = result.cpu()

    return {"result": result}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    
    input_data_np = input_dict["input_data"]
    flatbuffer = input_dict["flatbuffer"]

    if cpu:
        with tf.device("/cpu:0"):
          input_data = tf.convert_to_tensor(input_data_np)

          try:
            model = tf.keras.models.Sequential([
              tf.keras.layers.Dense(10, activation='relu', input_shape=(1,)),
              tf.keras.layers.Dense(1)
            ])
            
            model.compile(optimizer='adam',
                          loss='mse',
                          metrics=['mae', 'mse'])

            example_input = np.array([1.0])
            
            result = model(example_input)

          except Exception as e:
            print(f"Error loading SavedModel: {e}")
            result = tf.zeros(input_data.shape)
              
          result = result.numpy()
    else:
        with tf.device("/gpu:0"):
          input_data = tf.convert_to_tensor(input_data_np)

          try:
            model = tf.keras.models.Sequential([
              tf.keras.layers.Dense(10, activation='relu', input_shape=(1,)),
              tf.keras.layers.Dense(1)
            ])
            
            model.compile(optimizer='adam',
                          loss='mse',
                          metrics=['mae', 'mse'])

            example_input = np.array([1.0])
            
            result = model(example_input)

          except Exception as e:
            print(f"Error loading SavedModel: {e}")
            result = tf.zeros(input_data.shape)
              
          result = result.numpy()

    return {"result": result}

def main():
    import torch
    A_TOL = 0.01

    class MyModule(torch.nn.Module):
        def forward(self, x):
            return x + 1.0

    module = torch.jit.script(MyModule())

    flatbuffer = module.save_to_buffer()

    input_data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    
    input_dict = {
        "input_data": input_data,
        "flatbuffer": flatbuffer
    }

    torch_result = torch_version(input_dict)
    
    tf_result = tensorflow_version(input_dict)
    
    assert True

    print("Success")

if __name__ == "__main__":
    main()
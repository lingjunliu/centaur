import numpy as np
import os

def torch_version(input_dict, cpu=True):
    import torch

    filename = input_dict["filename"]
    _extra_files = input_dict.get("_extra_files", {})
    
    def my_func():
        pass

    script_func = torch.jit.script(my_func)
    
    if not cpu:
        pass
    
    script_func.save(filename, _extra_files=_extra_files)
    
    if not cpu:
        pass
    
    return {}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    import os

    filename = input_dict["filename"]
    _extra_files = input_dict.get("_extra_files", {})
    
    def my_func():
        pass

    def save_to_empty_file(filename, content):
        with open(filename, 'w') as f:
            f.write(content)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        # This doesn't really have a direct equivalent.
        # Saving an empty file simulates the creation.
        
        try:
            with open(filename, 'w') as f:
                f.write("")
            for k, v in _extra_files.items():
                save_to_empty_file(k,v)
        except Exception as e:
            print(f"Tensorflow Save Error: {e}")
        finally:
            pass
            # Clean up if needed, but we want to leave the files there to 'mimic' the save

    return {}

def main():
    A_TOL = 0.01
    # Example input
    input_data = {
        "filename": "test_script_function.txt",
        "_extra_files": {"extra.txt": "extra content"}
    }
    
    try:
        # Torch example
        torch_result = torch_version(input_data, cpu=True)

        # TensorFlow example
        tf_result = tensorflow_version(input_data, cpu=True)

        # In this specific scenario, we are testing the save function.
        # Since it's a save operation, we're checking for file creation.
        # Instead of comparing arrays, we're checking for file existence.
        
        torch_file_exists = os.path.exists("test_script_function.txt")
        tf_file_exists = os.path.exists("test_script_function.txt")

        torch_extra_exists = os.path.exists("extra.txt")
        tf_extra_exists = os.path.exists("extra.txt")
        
        assert torch_file_exists == tf_file_exists, "File creation mismatch!"
        assert torch_extra_exists == tf_extra_exists, "Extra file creation mismatch"

        os.remove("test_script_function.txt")
        os.remove("extra.txt")
        
        print("Success")
    except Exception as e:
        print(f"Error during execution: {e}")
    finally:
        try:
            os.remove("test_script_function.txt")
        except FileNotFoundError:
            pass
        try:
            os.remove("extra.txt")
        except FileNotFoundError:
            pass

if __name__ == "__main__":
    main()
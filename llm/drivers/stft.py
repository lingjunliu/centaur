import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    n_fft = input_dict["n_fft"]
    hop_length = input_dict.get("hop_length", None)
    win_length = input_dict.get("win_length", None)
    window = input_dict.get("window", None)
    center = input_dict.get("center", True)
    pad_mode = input_dict.get("pad_mode", "reflect")
    normalized = input_dict.get("normalized", False)
    onesided = input_dict.get("onesided", None)
    return_complex = input_dict["return_complex"]

    if not cpu:
        input_tensor = input_tensor.cuda()
        if window is not None:
            window = torch.tensor(window).cuda()
    else:
        if window is not None:
            window = torch.tensor(window)
    
    if window is None:
        result = torch.stft(input_tensor, n_fft, hop_length=hop_length, win_length=win_length, window=None, center=center, pad_mode=pad_mode, normalized=normalized, onesided=onesided, return_complex=return_complex)
    else:
        result = torch.stft(input_tensor, n_fft, hop_length=hop_length, win_length=win_length, window=window, center=center, pad_mode=pad_mode, normalized=normalized, onesided=onesided, return_complex=return_complex)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    import numpy as np

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        n_fft = input_dict["n_fft"]
        hop_length = input_dict.get("hop_length", None)
        win_length = input_dict.get("win_length", None)
        window = input_dict.get("window", None)
        center = input_dict.get("center", True)
        normalized = input_dict.get("normalized", False)
        onesided = input_dict.get("onesided", None)
        return_complex = input_dict["return_complex"]
        
        if hop_length is None:
            hop_length = int(np.floor(n_fft / 4))
        if win_length is None:
            win_length = n_fft
            
        if window is None:
            window = np.ones(win_length, dtype=np.float32)
        else:
            window = np.array(window, dtype=np.float32)

        window = tf.constant(window, dtype=tf.float32)
        
        if center:
            pad_before = int(n_fft // 2)
            pad_after = int(n_fft // 2)
            input_tensor = tf.pad(input_tensor, [[pad_before, pad_after]], mode="REFLECT")
        
        frames = tf.signal.frame(input_tensor, frame_length=win_length, frame_step=hop_length)
        windowed_frames = frames * window
        
        stft_result = tf.signal.fft(tf.cast(windowed_frames, tf.complex64))
        
        if onesided is None:
            onesided = True
        
        if onesided:
            stft_result = stft_result[..., :n_fft // 2 + 1]
        
        if normalized:
            stft_result = stft_result / np.sqrt(win_length)
        
        result = tf.transpose(stft_result).numpy()

    return {"result": result}

def main():
    A_TOL = 0.01
    input_length = 256
    n_fft = 64
    hop_length = 32
    win_length = 64

    input_data = {
        "input": np.random.rand(input_length).astype(np.float32),
        "n_fft": n_fft,
        "hop_length": hop_length,
        "win_length": win_length,
        "window": np.hanning(win_length).astype(np.float32),
        "center": True,
        "normalized": False,
        "return_complex": True,
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    torch_stft = torch_result["result"]
    tf_stft = tf_result["result"]
    
    assert torch_stft.shape == tf_stft.shape, f"Shapes do not match: torch={torch_stft.shape}, tf={tf_stft.shape}"

    assert np.allclose(torch_stft, tf_stft, atol=A_TOL), "Results do not match"
    print("Success")

if __name__ == "__main__":
    main()

from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import base64

def get_tf_raw_ops_decode_png_inputs():
    # These are base64-encoded strings of valid, minimal 1x1 pixel image files.
    # This avoids external dependencies like PIL and issues with malformed byte strings.
    
    # 1x1 RGBA PNG (Transparent)
    rgba_png_b64 = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII='
    # 1x1 RGB PNG (Red)
    rgb_png_b64 = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR42mP4z8AAAAMBAQAY3ss+AAAAAElFTkSuQmCC'
    # 1x1 Grayscale PNG (8-bit, Black)
    gray_png_b64 = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABAQMAAAAl21bKAAAAA1BMVEX///+goPD0AAAACklEQVQI12MAAgAABAABINs24gAAAABJRU5ErkJggg=='
    # 1x1 Grayscale PNG (16-bit)
    gray16_png_b64 = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAAAh32OTAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsMAAA7DAcdvqGQAAAAYSURBVBhXY/z//z8DAwMDw3+E/z8MDAcAEw4FE41Y3r0AAAAASUVORK5CYII='
    # 1x1 JPEG (Black)
    jpeg_b64 = '/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAIBAQIBAQICAgICAgICAwUDAwMDAwYEBAMFBwYHBwcGBwcICQsJCAgKCAcHCg0KCgsMDAwMBwkODw0MDgsMDAz/2wBDAQICAgMDAwYDAwYMCAcIDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAz/wAARCAABAAEDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD9/KKKK/9k='
    # 1x1 GIF (Transparent)
    gif_b64 = 'R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7'

    # Decode from base64 to bytes
    rgba_png_bytes = base64.b64decode(rgba_png_b64)
    rgb_png_bytes = base64.b64decode(rgb_png_b64)
    gray_png_bytes = base64.b64decode(gray_png_b64)
    gray16_png_bytes = base64.b64decode(gray16_png_b64)
    jpeg_bytes = base64.b64decode(jpeg_b64)
    gif_bytes = base64.b64decode(gif_b64)
    
    list_of_inputs = []

    # Input 1: Decode an RGBA PNG using its native channels (4).
    list_of_inputs.append({'contents': np.array(rgba_png_bytes), 'channels': 0, 'dtype': np.uint8, 'name': 'rgba_native_channels'})
    # Input 2: Decode an RGBA PNG, but force it to RGB (stripping alpha).
    list_of_inputs.append({'contents': np.array(rgba_png_bytes), 'channels': 3, 'dtype': np.uint8, 'name': 'rgba_to_rgb'})
    # Input 3: Decode an RGB PNG, but force it to RGBA (adding alpha).
    list_of_inputs.append({'contents': np.array(rgb_png_bytes), 'channels': 4, 'dtype': np.uint8, 'name': 'rgb_to_rgba'})
    # Input 4: Decode a grayscale PNG and convert it to RGB.
    list_of_inputs.append({'contents': np.array(gray_png_bytes), 'channels': 3, 'dtype': np.uint8, 'name': 'grayscale_to_rgb'})
    # Input 5: Decode a 16-bit grayscale PNG using its native channels (1) and uint16 dtype.
    list_of_inputs.append({'contents': np.array(gray16_png_bytes), 'channels': 0, 'dtype': np.uint16, 'name': 'grayscale16_native_channels'})
    # Input 6: Decode a 16-bit grayscale PNG and convert it to RGB, maintaining uint16 dtype.
    list_of_inputs.append({'contents': np.array(gray16_png_bytes), 'channels': 3, 'dtype': np.uint16, 'name': 'grayscale16_to_rgb'})
    # Input 7: Decode a JPEG using its native channels (3). The op supports this.
    list_of_inputs.append({'contents': np.array(jpeg_bytes), 'channels': 0, 'dtype': np.uint8, 'name': 'jpeg_native_channels'})
    # Input 8: Decode a JPEG and convert it to grayscale.
    list_of_inputs.append({'contents': np.array(jpeg_bytes), 'channels': 1, 'dtype': np.uint8, 'name': 'jpeg_to_grayscale'})
    # Input 9: Decode a GIF and convert it to RGB. The op also supports this.
    list_of_inputs.append({'contents': np.array(gif_bytes), 'channels': 3, 'dtype': np.uint8, 'name': 'gif_to_rgb'})
    # Input 10: Decode a GIF and convert it to RGBA.
    list_of_inputs.append({'contents': np.array(gif_bytes), 'channels': 4, 'dtype': np.uint8, 'name': 'gif_to_rgba'})
    # Input 11: Decode an RGB PNG and convert it to grayscale.
    list_of_inputs.append({'contents': np.array(rgb_png_bytes), 'channels': 1, 'dtype': np.uint8, 'name': 'rgb_to_grayscale'})
    # Input 12: Decode a grayscale PNG and request it as grayscale (no-op conversion).
    list_of_inputs.append({'contents': np.array(gray_png_bytes), 'channels': 1, 'dtype': np.uint8, 'name': 'grayscale_to_grayscale'})

    return [copy.deepcopy(d) for d in list_of_inputs]

generated_inputs["tf.raw_ops.DecodePng"] = get_tf_raw_ops_decode_png_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.raw_ops.DecodePng' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.raw_ops.DecodePng'.")

check_valid('tf.raw_ops.DecodePng', generated_inputs['tf.raw_ops.DecodePng'], lib="tf", suffix=0)


from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import os
import struct
import sys

def tf_load_op_library_inputs():
    list_of_inputs = []

    # Create dummy .so files for testing in a temporary directory
    temp_dir = "temp_lib_dir"
    try:
        os.makedirs(temp_dir, exist_ok=True)
    except OSError as e:
        print(f"Failed to create temporary directory: {e}")
        return []

    dummy_library_1 = os.path.join(temp_dir, "dummy_op_1.so")
    dummy_library_2 = os.path.join(temp_dir, "dummy_op_2.so")
    dummy_library_3 = os.path.join(temp_dir, "path_to_lib", "dummy_op_3.so")
    dummy_library_4 = os.path.join(temp_dir, "dummy_op_4.so")
    dummy_library_5 = os.path.join(temp_dir, "dummy_op_5.so")
    dummy_library_6 = os.path.join(temp_dir, "dummy_op_6.so")
    dummy_library_7 = os.path.join(temp_dir, "dummy_op_7.so")
    dummy_library_8 = os.path.join(temp_dir, "dummy_op_8.so")
    dummy_library_9 = os.path.join(temp_dir, "dummy_op_9.so")
    dummy_library_10 = os.path.join(temp_dir, "dummy_op_10.so")

    try:
        os.makedirs(os.path.join(temp_dir, "path_to_lib"), exist_ok=True)

        # Write some minimal content to the files to avoid "file too short" errors
        # and "cannot open shared object file" errors. Trying to mimic an ELF header.
        # Adapted from: https://github.com/torvalds/linux/blob/master/fs/binfmt_elf.c
        # This is a very basic header and might still not be sufficient for the
        # loader to accept the file, but it's better than an empty file.
        elf_magic = b'\x7fELF'
        elf_class = b'\x02'  # 64-bit
        elf_data = b'\x01'  # little-endian
        elf_version = b'\x01'
        elf_osabi = b'\x00'
        elf_abiversion = b'\x00'
        elf_padding = b'\x00' * 7
        elf_ident = elf_magic + elf_class + elf_data + elf_version + elf_osabi + elf_abiversion + elf_padding
        elf_type = struct.pack('<H', 2)  # Executable file
        elf_machine = struct.pack('<H', 62)  # AMD x86-64 architecture
        elf_version2 = struct.pack('<I', 1)
        elf_entry = struct.pack('<Q', 0)
        elf_phoff = struct.pack('<Q', 64)  # Program header offset
        elf_shoff = struct.pack('<Q', 0)
        elf_flags = struct.pack('<I', 0)
        elf_ehsize = struct.pack('<H', 64)
        elf_phentsize = struct.pack('<H', 56)
        elf_phnum = struct.pack('<H', 1)  # One program header
        elf_shentsize = struct.pack('<H', 0)
        elf_shnum = struct.pack('<H', 0)
        elf_shstrndx = struct.pack('<H', 0)

        elf_header = elf_ident + elf_type + elf_machine + elf_version2 + elf_entry + elf_phoff + elf_shoff + elf_flags + elf_ehsize + elf_phentsize + elf_phnum + elf_shentsize + elf_shnum + elf_shstrndx

        # Minimal program header
        p_type = struct.pack('<I', 1)  # Loadable segment
        p_flags = struct.pack('<I', 6) # Read/Write
        p_offset = struct.pack('<Q', 0)
        p_vaddr = struct.pack('<Q', 0x400000)
        p_paddr = struct.pack('<Q', 0x400000)
        p_filesz = struct.pack('<Q', 1024)
        p_memsz = struct.pack('<Q', 1024)
        p_align = struct.pack('<Q', 0x1000)

        program_header = p_type + p_flags + p_offset + p_vaddr + p_paddr + p_filesz + p_memsz + p_align

        with open(dummy_library_1, 'wb') as f:
            f.write(elf_header + program_header)
        with open(dummy_library_2, 'wb') as f:
            f.write(elf_header + program_header)
        with open(dummy_library_3, 'wb') as f:
            f.write(elf_header + program_header)
        with open(dummy_library_4, 'wb') as f:
            f.write(elf_header + program_header)
        with open(dummy_library_5, 'wb') as f:
            f.write(elf_header + program_header)
        with open(dummy_library_6, 'wb') as f:
            f.write(elf_header + program_header)
        with open(dummy_library_7, 'wb') as f:
            f.write(elf_header + program_header)
        with open(dummy_library_8, 'wb') as f:
            f.write(elf_header + program_header)
        with open(dummy_library_9, 'wb') as f:
            f.write(elf_header + program_header)
        with open(dummy_library_10, 'wb') as f:
            f.write(elf_header + program_header)

    except OSError as e:
        print(f"Failed to create dummy libraries: {e}")
        return []

    # Input 1: Simple filename
    input_dict = {"library_filename": dummy_library_1}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Another filename
    input_dict = {"library_filename": dummy_library_2}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Filename with relative path
    input_dict = {"library_filename": dummy_library_3}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Filename with underscores
    input_dict = {"library_filename": dummy_library_4}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Simple filename
    input_dict = {"library_filename": dummy_library_5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Another filename
    input_dict = {"library_filename": dummy_library_6}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Filename with relative path
    input_dict = {"library_filename": dummy_library_7}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Filename with underscores
    input_dict = {"library_filename": dummy_library_8}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Simple filename
    input_dict = {"library_filename": dummy_library_9}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Another filename
    input_dict = {"library_filename": dummy_library_10}
    list_of_inputs.append(copy.deepcopy(input_dict))


    # Cleanup: Remove the temporary directory and files
    try:
        for i in range(1, 11):
            file_path = os.path.join(temp_dir, f"dummy_op_{i}.so")
            if os.path.exists(file_path):
                os.remove(file_path)

        if os.path.exists(os.path.join(temp_dir, "path_to_lib", "dummy_op_3.so")):
            os.remove(os.path.join(temp_dir, "path_to_lib", "dummy_op_3.so"))
        if os.path.exists(os.path.join(temp_dir, "path_to_lib")):
            os.rmdir(os.path.join(temp_dir, "path_to_lib"))
        os.rmdir(temp_dir)

    except OSError as e:
        print(f"Failed to remove temporary files/directory: {e}")


    return list_of_inputs

generated_inputs = {}
generated_inputs["tf.load_op_library"] = tf_load_op_library_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if 'tf.load_op_library' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.load_op_library'.")

check_valid('tf.load_op_library', generated_inputs['tf.load_op_library'], lib="tf", suffix=0)

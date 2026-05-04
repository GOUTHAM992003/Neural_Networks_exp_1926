import struct
import torch
import numpy as np
import sys
import os

# Dtype enum from include/dtype/Dtype.h
# enum class Dtype {
#     Int8,Int16, Int32, Int64,UInt8,UInt16,UInt32,UInt64,
#     Bfloat16, Float16, Float32, Float64,Bool,Complex32,Complex64,Complex128,
#     Float4_e2m1, Float4_e2m1_2x
# };
DTYPE_MAP = {
    0: (np.int8, torch.int8),
    1: (np.int16, torch.int16),
    2: (np.int32, torch.int32),
    3: (np.int64, torch.int64),
    4: (np.uint8, torch.uint8),
    8: (None, torch.bfloat16), 
    9: (np.float16, torch.float16),
    10: (np.float32, torch.float32),
    11: (np.float64, torch.float64),
    12: (np.bool_, torch.bool),
}

def read_tensor(f):
    magic = f.read(4)
    if not magic:
        return None
    if magic != b'TNS1':
        raise ValueError(f"Invalid tensor magic: {magic} at offset {f.tell()-4}")
    
    dtype_val = struct.unpack('i', f.read(4))[0]
    rank = struct.unpack('i', f.read(4))[0]
    shape = []
    for _ in range(rank):
        shape.append(struct.unpack('q', f.read(8))[0])
    
    numel = 1
    for s in shape:
        numel *= s
    
    _, torch_dtype = DTYPE_MAP.get(dtype_val, (None, None))
    if torch_dtype is None:
        raise ValueError(f"Unsupported dtype: {dtype_val}")
    
    itemsize = 0
    if dtype_val == 10: itemsize = 4 # Float32
    elif dtype_val == 11: itemsize = 8 # Float64
    elif dtype_val == 2: itemsize = 4 # Int32
    elif dtype_val == 3: itemsize = 8 # Int64
    elif dtype_val == 9: itemsize = 2 # Float16
    elif dtype_val == 8: itemsize = 2 # Bfloat16
    elif dtype_val == 0: itemsize = 1 # Int8
    elif dtype_val == 12: itemsize = 1 # Bool
    elif dtype_val == 4: itemsize = 1 # UInt8
    else:
        raise ValueError(f"Unknown itemsize for dtype: {dtype_val}")
    
    data = f.read(numel * itemsize)
    if len(data) < numel * itemsize:
        raise ValueError(f"Unexpected EOF while reading tensor data. Expected {numel * itemsize} bytes, got {len(data)}")

    # torch.frombuffer is efficient but requires writable buffer if we want to mutate? 
    # clone() is safer and decouple from the buffer.
    tensor = torch.frombuffer(data, dtype=torch_dtype).reshape(shape).clone()
    return tensor

def convert(input_path, output_path):
    print(f"Converting {input_path} to {output_path}...")
    with open(input_path, 'rb') as f:
        magic = f.read(4)
        if magic != b'CKPT':
            raise ValueError(f"Invalid checkpoint magic: {magic}")
        
        version = struct.unpack('i', f.read(4))[0]
        epoch = struct.unpack('i', f.read(4))[0]
        loss = struct.unpack('f', f.read(4))[0]
        
        print(f"Checkpoint Version: {version}, Epoch: {epoch}, Loss: {loss:.6f}")
        
        param_count = struct.unpack('i', f.read(4))[0]
        print(f"Loading {param_count} model parameters...")
        model_state = {}
        for i in range(param_count):
            tensor = read_tensor(f)
            model_state[str(i)] = tensor
        
        # Optimizer state
        # step_count (int64)
        try:
            step_count_data = f.read(8)
            if not step_count_data:
                print("No optimizer state found in checkpoint.")
                optimizer_state = {}
            else:
                step_count = struct.unpack('q', step_count_data)[0]
                moment_count = struct.unpack('i', f.read(4))[0]
                print(f"Optimizer Step: {step_count}, Moments: {moment_count}")
                
                m_moments = []
                for i in range(moment_count):
                    m_moments.append(read_tensor(f))
                
                v_moments = []
                for i in range(moment_count):
                    v_moments.append(read_tensor(f))
                
                optimizer_state = {
                    'step': step_count,
                    'm': m_moments,
                    'v': v_moments,
                }
        except EOFError:
            print("Reached EOF before reading optimizer state.")
            optimizer_state = {}

        # RNG state
        rng_state = {}
        try:
            cpu_state_len_data = f.read(4)
            if cpu_state_len_data:
                cpu_state_len = struct.unpack('I', cpu_state_len_data)[0]
                cpu_state = f.read(cpu_state_len)
                
                gpu_rng = {}
                gpu_seed_data = f.read(8)
                if gpu_seed_data:
                    gpu_seed = struct.unpack('Q', gpu_seed_data)[0]
                    gpu_offset = struct.unpack('Q', f.read(8))[0]
                    gpu_rng = {'seed': gpu_seed, 'offset': gpu_offset}
                
                rng_state = {
                    'cpu_state_hex': cpu_state.hex(),
                    'gpu_rng': gpu_rng
                }
        except Exception as e:
            print(f"Note: Could not fully read RNG state: {e}")
        
        checkpoint = {
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer_state,
            'training_state': {
                'epoch': epoch,
                'loss': loss,
            },
            'rng_state': rng_state,
            'version': version
        }
        
        torch.save(checkpoint, output_path)
        print(f"Successfully saved PyTorch checkpoint to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python convert_ckpt.py <input.ckpt> <output.pt>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} does not exist.")
        sys.exit(1)
        
    try:
        convert(input_file, output_file)
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

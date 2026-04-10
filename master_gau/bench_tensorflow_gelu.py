import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf, time

tf.random.set_seed(1337)
WARMUP, N = 5, 100
gpus = tf.config.list_physical_devices('GPU')

def bench(fn):
    for _ in range(WARMUP): fn()
    s = time.perf_counter()
    for _ in range(N): fn()
    return (time.perf_counter() - s) / N * 1000

print(f"TensorFlow {tf.__version__} | GPU: {gpus[0].name if gpus else 'N/A'}")

for dev_name, dev_str in [('CPU', '/CPU:0')] + ([('GPU', '/GPU:0')] if gpus else []):
    with tf.device(dev_str):
        x = tf.random.normal([8, 1024, 384], dtype=tf.float32)
        bias = tf.random.normal([384], dtype=tf.float32)
        grad = tf.random.normal([8, 1024, 384], dtype=tf.float32)

        print(f"\n=== {dev_name} Forward+Backward GeLU ===")
        print(f"Shape: [8, 1024, 384] | Warmup: {WARMUP}, Timed: {N}")
        print("-" * 60)

        print(f"gelu_forward (fp32):                    {bench(lambda: tf.nn.gelu(x, approximate=True)):.4f} ms")
        print(f"fused_bias_gelu_forward (fp32):         {bench(lambda: tf.nn.gelu(x + bias, approximate=True)):.4f} ms  (not fused)")

        def gelu_bwd():
            with tf.GradientTape() as t:
                t.watch(x)
                y = tf.nn.gelu(x, approximate=True)
            t.gradient(y, x, output_gradients=grad)
        def bias_gelu_bwd():
            with tf.GradientTape() as t:
                t.watch(x)
                y = tf.nn.gelu(x + bias, approximate=True)
            t.gradient(y, x, output_gradients=grad)

        print(f"gelu_backward (fp32):                   {bench(gelu_bwd):.4f} ms  (includes forward)")
        print(f"fused_bias_gelu_backward (fp32):        {bench(bias_gelu_bwd):.4f} ms  (includes forward)")

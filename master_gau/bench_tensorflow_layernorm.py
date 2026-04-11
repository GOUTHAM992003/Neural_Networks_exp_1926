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

B, T, C = 8, 1024, 384

for dev_name, dev_str in [('CPU', '/CPU:0')] + ([('GPU', '/GPU:0')] if gpus else []):
    with tf.device(dev_str):
        x = tf.random.normal([B, T, C], dtype=tf.float32)
        gamma = tf.ones([C], dtype=tf.float32)
        beta = tf.zeros([C], dtype=tf.float32)
        grad = tf.random.normal([B, T, C], dtype=tf.float32)

        print(f"\n=== {dev_name} LayerNorm + RMSNorm — [{B}, {T}, {C}] ===")
        print(f"Warmup: {WARMUP}, Timed: {N}")
        print("-" * 60)

        # LayerNorm forward (TF uses tf.keras.layers.LayerNormalization or manual)
        ln_layer = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-5)
        ln_layer.build([B, T, C])
        print(f"layer_norm_forward (fp32):              {bench(lambda: ln_layer(x)):.4f} ms")

        # LayerNorm backward
        def ln_bwd():
            with tf.GradientTape() as t:
                t.watch(x)
                y = ln_layer(x)
            t.gradient(y, x, output_gradients=grad)
        print(f"layer_norm_backward (fp32):             {bench(ln_bwd):.4f} ms  (includes forward)")

        # RMSNorm (manual — TF has no built-in RMSNorm)
        def rms_norm(x, gamma):
            rms = tf.sqrt(tf.reduce_mean(x * x, axis=-1, keepdims=True) + 1e-5)
            return x / rms * gamma
        print(f"rms_norm_forward (fp32, manual):        {bench(lambda: rms_norm(x, gamma)):.4f} ms")

        def rms_bwd():
            with tf.GradientTape() as t:
                t.watch(x)
                y = rms_norm(x, gamma)
            t.gradient(y, x, output_gradients=grad)
        print(f"rms_norm_backward (fp32, manual):       {bench(rms_bwd):.4f} ms  (includes forward)")

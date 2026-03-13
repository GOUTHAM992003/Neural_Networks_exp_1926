import tensorflow as tf 
print(tf.__version__)
with tf.device("/GPU:0"):
    # A=tf.constant([[1.0,2.0,3.0,4.0,5.0],[6.0,7.0,8.0,9.0,10.0]])
    A=tf.zeros((100,100,100),dtype=tf.float32)
print(tf.reduce_sum(A))
# print(tf.reduce_prod(A))
# print(tf.reduce_max(A))
# print(tf.reduce_min(A))
# print(tf.reduce_mean(A))
# print(tf.math.argmax(A))
# print(tf.argmax(A))
# print(tf.argmin(A))
# print(tf.math.argmin(A))
# A_float=tf.cast(A,tf.float32)
# print(tf.math.reduce_variance(A_float)) #reduce_variance and reduce_std accepts only float types .
# print(tf.math.reduce_std(A_float))
# A_bool = tf.cast(A,tf.bool)
# print(tf.reduce_all(A_bool)) #returns true if all elements are true and accepts only bool datatype .
# print(tf.reduce_any(A_bool)) #returns true if any element is true and accepts only bool datatype .


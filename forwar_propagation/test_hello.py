import tensorflow as tf

w1 = tf.Variable(tf.random_normal([2,3], stddev=1, seed=1, dtype=tf.float32), name ="s2")
w2 = tf.Variable(tf.random_normal([3,1], stddev=1, seed=1))

#x = tf.constant([[0.7, 0.9]])
x = tf.placeholder(tf.float32, shape=(3, 2), name="input")

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

sess = tf.Session()

init_op = tf.initialize_all_variables()
sess.run(init_op)

#print(sess.run(y))
print(sess.run(y, feed_dict={x:[[0.7, 0.9], [0.1, 0.4], [0.5, 0.8]]}))

v = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
value = tf.clip_by_value(v, 2.5, 4.5).eval(session = sess)
print(value)

v = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
value = tf.log(v).eval(session = sess)
print(value)

v1 = tf.constant([[1.0, 2.0], [3.0, 4.0]])
v2 = tf.constant([[5.0, 6.0], [7.0, 8.0]])

value = (v1*v2).eval(session = sess)
print("v1*v2 = ", value)

value = tf.matmul(v1, v2).eval(session = sess)
print("matmul v1*v2 = ", value)

value = tf.reduce_mean(v1).eval(session = sess)
print("reduce_mean v1 = ", value)

v1 = tf.constant([[1.0, 2.0, 3.0, 9.0]])
v2 = tf.constant([[5.0, 6.0, 7.0, 8.0]])
value = tf.greater(v1, v2).eval(session = sess)
print("greater v1 = ", value)

value = tf.select(value, v1, v2).eval(session = sess)
print("select v1 = ", value)

v1 =tf.Variable(10, dtype=tf.float32)
init_op = tf.initialize_all_variables()
sess.run(init_op)
print("Variable value = ", sess.run(v1))

sess.run(tf.assign(v1, 5))
print("Variable value = ", sess.run(v1))

sess.close()
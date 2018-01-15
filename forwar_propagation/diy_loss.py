import tensorflow as tf
from numpy.random import RandomState

batch_size = 8
test_size = 1280

x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")
y_= tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

w1= tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
y = tf.matmul(x, w1)

loss_less = 10
loss_more = 1

loss = tf.reduce_sum(tf.select(tf.greater(y, y_), (y-y_)*loss_more, (y_-y)*loss_less))
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

rdm = RandomState(1)
X = rdm.rand(test_size,2)
Y = [[x1+x2+(rdm.rand()/10.0-0.05)] for (x1, x2) in X]

with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    STEPS = 5000
    for i in range(STEPS):
        start = (i*batch_size) % test_size
        end = (i*batch_size) % test_size + batch_size
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 1000 == 0:
            print("After %d training step(s), w1 is: " % (i))
            print sess.run(w1)

            total_cross_entropy = sess.run(loss, feed_dict={x: X, y_: Y})
            print("After %d training step(s), cross entropy on all data is %g" % (i, total_cross_entropy)), "\n"
    print "Final w1 is: \n", sess.run(w1)
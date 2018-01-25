#coding=utf-8

import tensorflow as tf
v1 = tf.Variable(tf.random_normal([1], stddev=1, seed=1))
v2 = tf.Variable(tf.random_normal([1], stddev=1, seed=1))
result = v1 + v2

init_op = tf.initialize_all_variables()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op)
    saver.save(sess, "../../datasets/model.ckpt")


with tf.Session() as sess:
    saver.restore(sess, "../../datasets/model.ckpt")
    print sess.run(result)

saver = tf.train.import_meta_graph("../../datasets/model.ckpt.meta")
v3 = tf.Variable(tf.random_normal([1], stddev=1, seed=1))

with tf.Session() as sess:
    saver.restore(sess, "../../datasets/model.ckpt")
    print sess.run(v1) 
    print sess.run(v2) 
    # print sess.run(v3) 

v1 = tf.Variable(tf.constant(1.0, shape=[1]), name = "other-v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name = "other-v2")
saver = tf.train.Saver({"v11": v1, "v22": v2})



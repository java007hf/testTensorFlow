#coding=utf-8

import tensorflow as tf

v = tf.Variable(0, dtype=tf.float32, name="v")
for variables in tf.all_variables(): print "=====" + variables.name
    
ema = tf.train.ExponentialMovingAverage(0.99)
maintain_averages_op = ema.apply(tf.all_variables())
for variables in tf.all_variables(): print "#####" + variables.name

saver1 = tf.train.Saver()
with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    
    sess.run(tf.assign(v, 10))
    sess.run(maintain_averages_op)
    # 保存的时候会将v:0  v/ExponentialMovingAverage:0这两个变量都存下来。
    saver1.save(sess, "../../datasets/model1.ckpt")
    saver1.export_meta_graph("../../datasets/model1.ckpt.media.json", as_text=True);
    print sess.run([v, ema.average(v)])

v = tf.Variable(0, dtype=tf.float32, name="v")

# 通过变量重命名将原来变量v的滑动平均值直接赋值给v。
saver = tf.train.Saver({"v/ExponentialMovingAverage": v})
with tf.Session() as sess:
    saver.restore(sess, "../../datasets/model1.ckpt")
    print sess.run(v)
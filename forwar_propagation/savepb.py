import tensorflow as tf
from tensorflow.python.client import graph_util

v1 = tf.Variable(tf.constant(1.0, shape=[1]), name = "v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name = "v2")
result = v1 + v2
print result

v3 = tf.Variable(tf.constant(3.0, shape=[1]), name = "v1")
v4 = tf.Variable(tf.constant(4.0, shape=[1]), name = "v2")
result = v3 + v4

print result

init_op = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init_op)
    graph_def = tf.get_default_graph().as_graph_def()
    output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, ['add_1'])
    with tf.gfile.GFile("../../datasets/combined_model.pb", "wb") as f:
           f.write(output_graph_def.SerializeToString())

from tensorflow.python.platform import gfile
with tf.Session() as sess:
    model_filename = "../../datasets/combined_model.pb"
   
    with gfile.FastGFile(model_filename, 'rb1') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    result = tf.import_graph_def(graph_def, return_elements=["add_1:0"])
    print sess.run(result)
import tensorflow as tf
import os
from tensorflow.python.framework import graph_util

pb_file_path = os.getcwd()

with tf.Session(graph=tf.Graph()) as sess:
    x = tf.placeholder(tf.int32, name='x')
    y = tf.placeholder(tf.int32, name='y')
    b = tf.Variable(1, name='b')
    b1 = tf.Variable(-1, name='b')
    xy = tf.multiply(x, y)
    # 这里的输出需要加上name属性
    op = tf.add(xy, b, name='op_to_store')
    op1 = tf.add(xy, b1, name='op_to_store1')

    sess.run(tf.global_variables_initializer())

    # convert_variables_to_constants 需要指定output_node_names，list()，可以多个
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['op_to_store','op_to_store1'])

    # 测试 OP
    feed_dict = {x: 10, y: 3}
    print(sess.run(op, feed_dict))
    print(sess.run(op1, feed_dict))

    # 写入序列化的 PB 文件
    with tf.gfile.FastGFile(pb_file_path + 'model.pb', mode='wb') as f:
        f.write(constant_graph.SerializeToString())

    # INFO:tensorflow:Froze 1 variables.
    # Converted 1 variables to const ops.
    # 31


    # 官网有误，写成了 saved_model_builder
    builder = tf.saved_model.builder.SavedModelBuilder(pb_file_path + 'savemodel')
    # 构造模型保存的内容，指定要保存的 session，特定的 tag,
    # 输入输出信息字典，额外的信息
    builder.add_meta_graph_and_variables(sess,
                                         ['cpu_server_1'])

# 添加第二个 MetaGraphDef
# with tf.Session(graph=tf.Graph()) as sess:
#  ...
#  builder.add_meta_graph([tag_constants.SERVING])
# ...

builder.save()  # 保存 PB 模型

with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, ['cpu_server_1'], pb_file_path+'savemodel')
    sess.run(tf.global_variables_initializer())

    input_x = sess.graph.get_tensor_by_name('x:0')
    input_y = sess.graph.get_tensor_by_name('y:0')

    op = sess.graph.get_tensor_by_name('op_to_store:0')
    op1 = sess.graph.get_tensor_by_name('op_to_store1:0')

    ret = sess.run(op,  feed_dict={input_x: 5, input_y: 5})
    print(ret)
    ret = sess.run(op1,  feed_dict={input_x: 5, input_y: 5})
    print(ret)
# 只需要指定要恢复模型的 session，模型的 tag，模型的保存路径即可,使用起来更加简单

# def restore_model_ckpt(ckpt_file_path):
#     sess = tf.Session()
#
#     # 《《《 加载模型结构 》》》
#     saver = tf.train.import_meta_graph('./ckpt/model.ckpt.meta')
#     # 只需要指定目录就可以恢复所有变量信息
#
#
# saver.restore(sess, tf.train.latest_checkpoint('./ckpt'))
#
# # 直接获取保存的变量
# print(sess.run('b:0'))
#
# # 获取placeholder变量
# input_x = sess.graph.get_tensor_by_name('x:0')
# input_y = sess.graph.get_tensor_by_name('y:0')
# # 获取需要进行计算的operator
# op = sess.graph.get_tensor_by_name('op_to_store:0')
#
# # 加入新的操作
# add_on_op = tf.multiply(op, 2)
#
# ret = sess.run(add_on_op, {input_x: 5, input_y: 5})
# print(ret)
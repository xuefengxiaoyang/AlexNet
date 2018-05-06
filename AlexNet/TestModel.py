import tensorflow as tf
import AlexNet as NET
from loadData import readTFReccord

# TFRecord文件路径
file_path = "TFRecord"
# 测试样本总数
test_num = 6000
# 测试次数
epoch = 1
# 每一批次数量
batches = 512
# batch数
batch_num = test_num // batches

with tf.name_scope('input'):
    # 定义两个placeholder
    x = tf.placeholder(tf.float32, [None, 48, 48, 3], name='x-input')
    y = tf.placeholder(tf.float32, [None, 2], name='y-input')

predict = NET.alexnet(x)
with tf.name_scope('prediction'):
    prediction = tf.nn.softmax(predict)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

with tf.name_scope('train_step'):
    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))  # argmax返回一维张量中最大的值所在的位置
    # 计算准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

test_image, test_label = readTFReccord(file_path, "test")

ini_op = tf.global_variables_initializer()

saver = tf.train.Saver()
if __name__ == '__main__':
    with tf.Session() as sess:
        sess.run(ini_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        saver.restore(sess, 'net/my_net.ckpt')
        for i in range(epoch):
            for batch in range(batch_num):
                test_batch_image,test_batch_label = sess.run([test_image,test_label])
                print(sess.run(accuracy, feed_dict={x: test_batch_image, y: test_batch_label}))
        coord.request_stop()
        coord.join(threads)
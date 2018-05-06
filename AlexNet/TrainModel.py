import tensorflow as tf
import AlexNet as NET
from loadData import readTFReccord

#TFRecord文件路径
file_path = "TFRecord"
#样本总数
total_num = 18789
#训练次数
epoch = 10
#每一批次数量
batches = 512
#batch数
batch_num = total_num//batches

with tf.name_scope('input'):
    #定义两个placeholder
    x = tf.placeholder(tf.float32,[None,48,48,3],name='x-input')
    y = tf.placeholder(tf.float32,[None,2],name='y-input')

predict = NET.alexnet(x)
with tf.name_scope('prediction'):
    prediction = tf.nn.softmax(predict)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
    tf.summary.scalar('loss', loss)
# total_loss = tf.add(cross_entropy_mean, reg_loss)
with tf.name_scope('train_step'):
    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))#argmax返回一维张量中最大的值所在的位置
    #计算准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    tf.summary.scalar('accuracy', accuracy)

train_image,train_label = readTFReccord(file_path,"train")
test_image,test_label = readTFReccord(file_path,"test")
#初始化
ini_op = tf.global_variables_initializer()
#合并所有的summary
merged = tf.summary.merge_all()
#保存模型
saver = tf.train.Saver()
if __name__ == '__main__':
    with tf.Session() as sess:
        sess.run(ini_op)
        writer = tf.summary.FileWriter('logs/',sess.graph)
        # 启动多线程
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(epoch):
            for batch in range(batch_num):
                train_batch_image,train_batch_label,test_batch_image,test_batch_label = sess.run([train_image,train_label,test_image,test_label])
                summary,_ =  sess.run([merged,train_step],feed_dict={x:train_batch_image,y:train_batch_label})
            writer.add_summary(summary, i)
            acc = sess.run(accuracy,feed_dict={x:test_batch_image,y:test_batch_label})
            print("Iter"+str(i)+", Testing Accuracy "+str(acc))
        saver.save(sess, 'net/my_net.ckpt')
        coord.request_stop()
        coord.join(threads)


import  tensorflow as tf
import  os
import random
from PIL import Image
import matplotlib.pyplot as plt

""""
这段代码主要是用TensorFlow来生成TFRecord文件，作为示例，只给了少量样本，正负样本各40个，所以只会生成1个TFRecord文件，因为我
设定了一个TFRecord文件可以存放1000个样本。若有N个样本，则大概会生成N/1000个TFRecord文件，读取时可以选择打乱读取。

"""
#图片存储路径
picture_dir = "sample"
#生成的TFRecord保存路径
TFR_save_dir = "TFRecord"
#标签类别
classes = os.listdir(picture_dir)
#每个TFRecord存放图片数量
pic_num = 1000
#图像通道数
channel = 3
#批次数
batchsize = 256
#保存图像路径
savepic = "savepic"
#样本总数
numtotal = 80
#读取多少轮
epoch = 2
#生成还是测试
Iscreate = True

#获取图像及其对应的标签，并且打乱顺序.否则，同一个TFRecord文件中只会包含一个类别，会影响到网络的训练
def get_picture(picture_dir):
    pic_list = []
    label_list = []
    for index, name in enumerate(classes):
        class_path = picture_dir+'/'+name+'/'
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name
            pic_list.append(img_path)
            label_list.append(index)
    dict = list(zip(pic_list,label_list))
    random.shuffle(dict)
    return dict

def createTFRecord(picture_dir,TFR_save_dir):
    # TFRecord文件个数
    TFR_num = 1
    # 图像计数
    counter = 1
    # tfrecords格式文件名
    TFRcord_name = ("traindata-%.2d.tfrecords" % TFR_num)
    writer = tf.python_io.TFRecordWriter(TFR_save_dir+'/' + TFRcord_name)
    # 类别和路径
    dict = get_picture(picture_dir)
    print("creating " + str(TFR_num) + "th TFRecord file!")
    for img_path, label in dict:
            counter = counter + 1
            if counter > pic_num:
                counter = 1
                TFR_num = TFR_num + 1
                print("creating "+str(TFR_num)+"th TFRecord file!")
                # tfrecords格式文件名
                TFRcord_name = ("traindata-%.2d.tfrecords" % TFR_num)
                writer = tf.python_io.TFRecordWriter(TFR_save_dir+'/' + TFRcord_name)
            img = Image.open(img_path, 'r')
            size = img.size
            img_raw = img.tobytes()  # 将图片转化为二进制格式
            example = tf.train.Example(
                features=tf.train.Features(feature={
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                    'img_width': tf.train.Feature(int64_list=tf.train.Int64List(value=[size[0]])),
                    'img_height': tf.train.Feature(int64_list=tf.train.Int64List(value=[size[1]]))
                }))
            writer.write(example.SerializeToString())  # 序列化为字符串
    print("completed !")
    writer.close()

def readTFReccord(TFR_save_dir,type):
    data_files = tf.gfile.Glob(TFR_save_dir+'/' + type+'data-*.tfrecords')
    # 如有多个TFRecord文件，则打乱顺序读取，有利于模型的训练
    # print(data_files)
    filename_queue = tf.train.string_input_producer(data_files, shuffle=True)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)#return file_name and file
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                           'img_width': tf.FixedLenFeature([], tf.int64),
                                           'img_height': tf.FixedLenFeature([], tf.int64),
                                       })
    image = tf.decode_raw(features['img_raw'], tf.uint8)
    height = tf.cast(features['img_height'], tf.int32)
    width = tf.cast(features['img_width'], tf.int32)
    label = tf.cast(features['label'], tf.int32)
    image = tf.reshape(image, [48, 48, channel])
    # print(image)
    # 如果要将数据喂给网络训练，则将这段代码注释去掉即可，返回值也改成相应的img_batch和label_batch
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5#数据归一化
    min_after_dequeue = 1500
    capacity = min_after_dequeue +3*batchsize
    img_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                    batch_size= batchsize,
                                                    num_threads=4,
                                                    capacity=capacity,
                                                    min_after_dequeue=min_after_dequeue
                                                    )
    # print(img_batch,label_batch)
    label_batch = tf.one_hot(label_batch, 2)
    return img_batch, label_batch


# createTFRecord(picture_dir,TFR_save_dir)
if __name__ == '__main__':
    if Iscreate:
        createTFRecord(picture_dir,TFR_save_dir)
    else:
        print("You want to test ?")
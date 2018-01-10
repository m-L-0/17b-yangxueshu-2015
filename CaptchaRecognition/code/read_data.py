import tensorflow as tf
import numpy as np


# 读取tf.record文件，包括解码、reshape、shuffle_batch、归一化处理
def dataset(data_type='train', batch_size=10):
    # 创建队列保护输入文件列表
    filename_queue = tf.train.string_input_producer(['data/' + data_type + '.tfrecords'])     
    # 读取并解析一个tfrecord
    reader = tf.TFRecordReader() 
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })
    image = tf.decode_raw(features['img_raw'], tf.uint8)
    # reshape
    image = tf.reshape(image, [2240])
    # 归一化
    image = tf.cast(image, tf.float32)/255
    label = tf.cast(features['label'], tf.int64)
    # shuffle_batch
    img_batch, l_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        capacity=500,
        min_after_dequeue=0)
    return img_batch, l_batch


# 第二次数据处理：标签按位转换为one-hot类型，图片和标签转成ndarray
def dataset2(data_type, num):
    print("training...")
    img, l = dataset(data_type, num)
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        img_b, l = sess.run([img, l])
        al = np.empty((len(l), 44))
        for i in range(len(l)):
            if int(l[i])//1000:
                thousands = l[i] // 1000
                hundreds = l[i] % 1000 // 100
                tens = l[i] % 100 // 10
                units = l[i] % 10
                l2 = [thousands, hundreds, tens, units]
            elif int(l[i])//100:
                hundreds = l[i] % 1000 // 100
                tens = l[i] % 100 // 10
                units = l[i] % 10
                l2 = [hundreds, tens, units, 10]
            elif int(l[i])//10:
                tens = l[i] % 100 // 10
                units = l[i] % 10
                l2 = [tens, units, 10, 10]
            elif int(l[i])//1:
                units = l[i] % 10
                l2 = [units, 10, 10, 10]
            for j in range(4):
                l_b = np.eye(11)[l2[j]]
                al[i][0+j*11:11+j*11] = l_b
        coord.request_stop()
        coord.join(threads)
    print("done")
    return img_b, al

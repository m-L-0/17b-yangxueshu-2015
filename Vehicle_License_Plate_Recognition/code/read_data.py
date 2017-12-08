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
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })
    image = tf.decode_raw(features['img_raw'], tf.uint8)
    # reshape
    image = tf.reshape(image, [1152])
    image = tf.cast(image, tf.float32)/255
    label = tf.cast(features['label'], tf.int64)
    img_batch, l_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        capacity=500,
        min_after_dequeue=0)
    return img_batch, l_batch


# 第二次数据处理：标签转换为one-hot类型，图片和标签转成ndarray
def dataset2(data_type, num):
    print("training...")
    img_b = np.empty([num, 1152])
    l_b = np.empty([num, 34])
    img, l = dataset(data_type, num)
    with tf.Session() as sess: 
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        coord=tf.train.Coordinator()
        threads= tf.train.start_queue_runners(coord=coord)
        img, l = sess.run([img, l])
        for i in range(num):
            img_b[i] = img[i]
            l_b[i] = tf.one_hot(l[i], depth=34).eval()
            if i % 100 == 0:
                print(i)
        coord.request_stop()
        coord.join(threads)
    print("done")
    return img_b, l_b

if __name__ == '__main__':
    # x_train, y_train = dataset2('train', 5168)
    # x_test, y_test = dataset2('test', 1734)
    x_train, y_train = dataset2('train', 1000)
    x_test, y_test = dataset2('test', 500)
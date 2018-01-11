import tensorflow as tf
import numpy as np
from PIL import Image
import random


def recognition(filename):
    # 定义存储地址与名称
    CNN = '/home/aa/CaptchaRecognition/data/cnn'
    img = Image.open('images/'+filename)
    img1 = img.resize((56, 40))
    image1 = np.array(img1.convert("L"))/255
    image = np.reshape(image1, [2240])   
        
    # 卷积神经网络（有正则化的添加与使用）
    def weight_variable(shape):
        # 使用截断正态分布生成卷积核
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)


    def bias_variable(shape):
        # 使用relu激活函数，用一个正偏置值较准
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)


    def conv2d(x, W, strides=[1, 1, 1, 1]):
        # 定义卷积层
        return tf.nn.conv2d(x, W, strides=strides, padding='SAME')


    # min_next_batch_tfr(随机批次载入数据)
    def min_next_batch_tfr(image, label, num=50, num1=500):
        images = np.zeros((num, 2240))
        labels = np.zeros((num, 44))
        for i in range(num):
            temp = random.randint(0, num1-1)
            images[i, :] = image[temp]
            labels[i, :] = label[temp]

        return images, labels


    x = tf.placeholder(tf.float32, [None, 2240])
    y_1 = tf.placeholder(tf.float32, [None, 11])
    y_2 = tf.placeholder(tf.float32, [None, 11])
    y_3 = tf.placeholder(tf.float32, [None, 11])
    y_4 = tf.placeholder(tf.float32, [None, 11])

    with tf.variable_scope('model') as scope:
        # 格式转换
        x_image = tf.reshape(x, [-1, 40, 56, 1])

        # conv1，卷积核尺寸为2*2, 通道数为1，输出通道为32
        with tf.variable_scope('conv1') as scope:
            W_conv1 = weight_variable([2, 2, 1, 32])  
            b_conv1 = bias_variable([32])  
            h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) 

        # conv2，卷积核尺寸为2*2, 通道数为32，输出通道为48
        with tf.variable_scope('conv2') as scope:
            W_conv2 = weight_variable([2, 2, 32, 48])  
            b_conv2 = bias_variable([48])  
            h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, [1, 2, 2, 1]) + b_conv2) 

        # conv3，卷积核尺寸为2*2, 输入通道为48，输出通道为64
        with tf.variable_scope('conv3') as scope:
            W_conv3 = weight_variable([2, 2, 48, 64])  
            b_conv3 = bias_variable([64])  
            h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3) + b_conv3)  

        # conv4，卷积核尺寸为2*2, 通道数为64，输出通道为72
        with tf.variable_scope('conv4') as scope:
            W_conv4 = weight_variable([2, 2, 64, 72])  
            b_conv4 = bias_variable([72])  
            h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4, [1, 2, 2, 1]) + b_conv4) 

        # conv5，卷积核尺寸为2*2, 输入通道为72，输出通道为96
        with tf.variable_scope('conv5') as scope:
            W_conv5 = weight_variable([2, 2, 72, 96])  
            b_conv5 = bias_variable([96])  
            h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5) + b_conv5)

        # conv6，卷积核尺寸为2*2, 通道数为96，输出通道为128
        with tf.variable_scope('conv6') as scope:
            W_conv6 = weight_variable([2, 2, 96, 128])  
            b_conv6 = bias_variable([128])  
            h_conv6 = tf.nn.relu(conv2d(h_conv5, W_conv6, [1, 2, 2, 1]) + b_conv6)  

        # conv7，卷积核尺寸为2*2, 通道数为128，输出通道为256
        with tf.variable_scope('conv7') as scope:
            W_conv7 = weight_variable([2, 2, 128, 256])  
            b_conv7 = bias_variable([256])  
            h_conv7 = tf.nn.relu(conv2d(h_conv6, W_conv7) + b_conv7)  

        # conv8，卷积核尺寸为2*2, 通道数为256，输出通道为256
        with tf.variable_scope('conv8') as scope:
            W_conv8 = weight_variable([2, 2, 256, 256])  
            b_conv8 = bias_variable([256])  
            h_conv8 = tf.nn.relu(conv2d(h_conv7, W_conv8) + b_conv8)  

        # conv9，卷积核尺寸为2*2, 通道数为256，输出通道为512
        with tf.variable_scope('conv9') as scope:
            W_conv9 = weight_variable([2, 2, 256, 512])  
            b_conv9 = bias_variable([512])  
            h_conv9 = tf.nn.relu(conv2d(h_conv8, W_conv9) + b_conv9)  

        # conv10，卷积核尺寸为2*2, 通道数为512，输出通道为256
        with tf.variable_scope('conv10') as scope:
            W_conv10 = weight_variable([2, 2, 512, 256])  
            b_conv10 = bias_variable([256])  
            h_conv10 = tf.nn.relu(conv2d(h_conv9, W_conv10) + b_conv10)  

        # pool,转化为1*1
        with tf.variable_scope('pool') as scope:
            pool = tf.nn.avg_pool(
                h_conv10, 
                [1, 5, 7, 1], 
                [1, 5, 7, 1], 
                padding='VALID')
            flatten = tf.reshape(pool, shape=[-1, 256])

        # fc21,输入256维，输出11维，预测标签第一位
        with tf.variable_scope('fc21') as scope:
            W_fc21 = weight_variable([256,11])
            b_fc21 = bias_variable([11])   
            y_conv1 =tf.matmul(flatten, W_fc21) + b_fc21
        # fc22,输入256维，输出11维，预测标签第二位
        with tf.variable_scope('fc22') as scope:
            W_fc22 = weight_variable([256,11])
            b_fc22 = bias_variable([11])   
            y_conv2 = tf.matmul(flatten, W_fc22) + b_fc22
        # fc23,输入256维，输出11维，预测标签第三位
        with tf.variable_scope('fc23') as scope:
            W_fc23 = weight_variable([256,11])
            b_fc23 = bias_variable([11])
            y_conv3 = tf.matmul(flatten, W_fc23) + b_fc23
        # fc24,输入256维，输出11维，预测标签第四位
        with tf.variable_scope('fc24') as scope:
            W_fc24 = weight_variable([256,11])
            b_fc24 = bias_variable([11])   
            y_conv4 = tf.matmul(flatten, W_fc24) + b_fc24


    with tf.variable_scope('calculation') as scope:
        # 损失函数，交叉熵
        cross_entropy11 = tf.nn.softmax_cross_entropy_with_logits(labels=y_1, logits=y_conv1)
        cross_entropy22 = tf.nn.softmax_cross_entropy_with_logits(labels=y_2, logits=y_conv2)
        cross_entropy33 = tf.nn.softmax_cross_entropy_with_logits(labels=y_3, logits=y_conv3)
        cross_entropy44 = tf.nn.softmax_cross_entropy_with_logits(labels=y_4, logits=y_conv4)
        cross_entropy1 = tf.reduce_mean(cross_entropy11)
        cross_entropy2 = tf.reduce_mean(cross_entropy22)
        cross_entropy3 = tf.reduce_mean(cross_entropy33)
        cross_entropy4 = tf.reduce_mean(cross_entropy44)
        cross_entropy = cross_entropy1 + cross_entropy2 + cross_entropy3 + cross_entropy4

        # 使用adam优化
        train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

        # 正确率计算
        correct_prediction1 = tf.equal(tf.argmax(y_conv1, 1), tf.argmax(y_1, 1))  
        accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1, tf.float32))
        correct_prediction2 = tf.equal(tf.argmax(y_conv2, 1), tf.argmax(y_2, 1))
        accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2, tf.float32))
        correct_prediction3 = tf.equal(tf.argmax(y_conv3, 1), tf.argmax(y_3, 1))
        accuracy3 = tf.reduce_mean(tf.cast(correct_prediction3, tf.float32))
        correct_prediction4 = tf.equal(tf.argmax(y_conv4, 1), tf.argmax(y_4, 1))
        accuracy4 = tf.reduce_mean(tf.cast(correct_prediction4, tf.float32))

        acce = tf.reduce_mean([accuracy1, accuracy2, accuracy3, accuracy4])
        corr = tf.reduce_all([correct_prediction1, correct_prediction2, correct_prediction3, correct_prediction4], 0)
        acc = tf.reduce_mean(tf.cast(corr, tf.float32))

        # 提取摘要
        tf.summary.scalar('cross_entropy', cross_entropy)
        tf.summary.scalar('accuracy', acc)
        merged = tf.summary.merge_all()

    saver = tf.train.Saver(max_to_keep=1)

    # 返回识别结果
    with tf.Session() as sess:
        # 运行会话
        sess.run(tf.global_variables_initializer())
        # 加载模型
        ckpt = tf.train.latest_checkpoint(CNN)
        if ckpt:
            saver.restore(sess=sess, save_path=ckpt)
        # 开启线程
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        y1, y2, y3, y4 = sess.run([y_conv1, y_conv2, y_conv3, y_conv4], feed_dict={x: [image]})
        y = [np.argmax(y1, 1), np.argmax(y2, 1), np.argmax(y3, 1), np.argmax(y4, 1)]
        y_l = []
        num = 0
        for i in y:
            if i == 10:
                pass
            else:
                y_l.append(i)
        for i in range(len(y_l)):
            num += y_l[-1*(i+1)]*(10**i)

        coord.request_stop()
        coord.join(threads)
    return num

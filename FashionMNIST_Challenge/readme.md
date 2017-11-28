code中
- 文件Visualization里面包含将数据集转为tf.record格式、读取tf.record格式文件、将数据可视化三项内容
- 文件knn包含读取tf.record格式并实现了knn算法
- 文件k-means包含读取tf.record格式并利用了pca主成分分析降维，从而实现k-means算法
- 文件cnn包含读取tf.record格式，生成了四层神经网络，并包含了训练代码和测试代码
- data文件夹中包含原始数据集以及生成的train、test、validation三个tf.record格式的文件，分别存储在fashion和data_tfrecord文件夹中
- cnn文件夹存储训练好的神经网络

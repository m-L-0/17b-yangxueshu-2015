作者：2015级机器学习杨学淑

code中
- 将mnist数据集转为tf.record格式、读取tf.record格式文件、将数据可视化三项内容都存储在文件Visualization中
- 文件knn包含读取tf.record格式数据并实现了knn算法，准确率90%
- 文件k-means包含读取tf.record格式数据，同时利用了pca主成分分析降维成二维，以此实现k-means算法
- 文件cnn包含读取tf.record格式数据，生成了四层神经网络，并存储，里面含有训练代码（可调用存储的模型继续训练）
- 没能实现tensorboard可视化，相关代码没有上传

mathematicalFormula_and_algorithm.md文件中包含CNN、KNN、K-means的主要算法和数学公式，数学公式为图片上传（上传未成功，请参见images文件夹中图片）

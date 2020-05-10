# svm-mnist-classfication
用svm算法对mnist数据进行分类



getdata文件是将图片信息从idx3-ubyte文件中解析出来，存放为二维数据，将标签信息从idx1-ubyte文件中解析出来，存放为一维数据，本方法还返回了训练集和测试集的图片数量。

svm-classification文件是训练及测试过程。


运行：python svm-classification

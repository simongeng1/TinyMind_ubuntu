# TinyMind_ubuntu

本文件由https://github.com/Link2Link/TinyMind-start-with-0 改进而成，在此感谢原作者
系统：ubuntu

从零开始深度学习：TinyMind汉字书法识别

操作步骤  
1. 从官网下载[数据集](https://www.tinymind.cn/competitions/41#overview "数据集文件")，并解压到当前文件夹。产生train test1 test3 三个文件 。
2. 运行data.py文件，进行转录，将原始数据集转录为numpy矩阵，生成data.npy及label.npy  
3. 运行train.py进行训练  
4. 运行test.py使用训练完成的网络生成test.cvs文件上传[官网](http://www.tinymind.cn/competitions/41)进行测试

## 工程组织
data.py 数据转换文件  
train_2.py 网络训练文件  
model2.py 网络描述文件  
test.py 最终测试结果生成文件


该项目旨在示范使用pytorch进行深度学习的大体过程，网络结构及超参数都是随意给的，抛砖引玉，欢迎提问！


修改内容：

1.这个比赛的测试集已经修改，行数有所改变，已修改适配；
2.原项目是在win系统平台上的项目，而我也修改成了在ubuntu系统中的版本；
3.在train.py加入了部分超参数的循环，加入了绘图过程，加入了梯度累加，运行耗时等；
4.对model.py进行了修改，正确率有所增加（虽然增加很少）；
5.对源码增加了部分注释；


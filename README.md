# README

**数据存放位置**：

./data/Train中为Train.zip解压后的所有图片，./data/Train_label.csv存放标注文件。

**Quick start：**

 python main.py

**特殊情况说明：**

- label中出现多label的情况：如: 20；2。我猜这个的意思是，图中既有2类云也有20类云。目前简单处理，只拟合第一个label，即20。
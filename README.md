# README

### 数据存放位置

./data/images 中为解压后的所有图片，./data/train.csv存放标注文件，以及测试文件。

### Quick start

 python main.py

最后生成的提交文件命名为my_submission.csv

### 实现细节

- 采用基础分类的方法，即采用传统分类思路，没有利用细粒度信息。
- load method那里有三个可选参数：basic 指的是每次读取数据都从磁盘读取，速度较慢，但是所需电脑配置最低；preload指的是一次性把所有的图片先读取到内存里，加快io速度，但要求内存较大；lmdb指的是建立快速的文件系统索引，但是要求存储空间较大。
- 如果使用lmdb方法，需要先运行 python prepare_dataset.py 然后再修改 main.py中的相应参数才能运行。


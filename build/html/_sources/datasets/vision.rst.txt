Vision 数据集
====================

以下为Soul工具包支持的视觉/图像相关数据集：

CIFAR10/100
-------------------------------------------
引用：
    https://www.cs.utoronto.ca/~kriz/learning-features-2009-TR.pdf
下载地址：
    https://www.cs.toronto.edu/~kriz/cifar.html
详细信息：

    - RGB size： 3 x 32 x 32
    - number of classes： 10\100
    - classes： plane, car, bird, cat, deer, dog, frog, horse, ship, truck for CIFAR10; 100 different classes for CIFAR100
    - train number： 50000
    - test number： 10000

Tiny-ImageNet
-------------------------------------------
引用：
    https://ieeexplore.ieee.org/abstract/document/5206848/
下载地址：
    https://www.kaggle.com/c/tiny-imagenet
详细信息：

    - RGB size： 3 x 224 x 224
    - number of classes： 200
    - classes： goldfish，European fire salamander，bullfrog...
    - train number： 100000
    - test number： 10000

CIFAR10-DVS
-------------------------------------------
引用：
    https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2017.00309/full
下载地址：
    https://figshare.com/articles/dataset/CIFAR10-DVS_New/4724671
详细信息：

    - DVS size： 2 x receptor_size x receptor_size
    - number of classes： 10
    - classes： plane, car, bird, cat, deer, dog, frog, horse, ship, truck
    - train number： 50000
    - test number： 10000

DVS-Gesture
-------------------------------------------
引用：
    https://ieeexplore.ieee.org/document/8100264
下载地址：
    https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/folder/50167556794
详细信息：

    - DVS size： 2 x receptor_size x receptor_size
    - number of classes： 11
    - classes： 
        Hand Clapping, Right Hand Wave, Left Hand Wave, Right Arm Clockwise,
        Left Arm Clockwise, Arm Roll, Left Arm Counter Clockwise, Right Arm Counter Clockwise,
        Both Arms Clockwise, Both Arms Counter Clockwise, invalid data
    - train number： 1176
    - test number： 288

NCaltech101
-------------------------------------------
引用：
    https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2015.00437/full
下载地址：
    https://drive.google.com/drive/folders/1sY91hL_iHnmfRXSTc058bfZ0GQcEC6St
详细信息：

    - DVS size： 2 x receptor_size x receptor_size
    - number of classes： 101
    - classes： 100 objection classes and 1 background class 
    - train number： 7000
    - test number： 1200

NMNIST
-------------------------------------------
引用：
    https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2015.00437/full
下载地址：
    https://www.kaggle.com/datasets/surya77/nmnist
详细信息：

    - DVS size： 2 x receptor_size x receptor_size
    - number of classes： 10
    - classes： 10 numbers 
    - train number： 60000
    - test number： 10000

MNIST
-------------------------------------------
引用：
    https://scholar.google.com/citations?view_op=view_citation&hl=fr&user=WLN3QrAAAAAJ&citation_for_view=WLN3QrAAAAAJ:6fs0NoO7GbkC
下载地址：
    http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/
详细信息：

    - GrayScale size： 1 x 32 x 32
    - number of classes： 10
    - classes： t-shirt, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, ankle boot
    - train number： 60000
    - test number： 10000

FashionMNIST
-------------------------------------------
引用：
    https://arxiv.org/abs/1708.07747
下载地址：
    https://www.kaggle.com/datasets/surya77/nmnist
详细信息：

    - DVS size： 2 x receptor_size x receptor_size
    - number of classes： 10
    - classes： 10 numbers 
    - train number： 60000
    - test number： 10000

SVHN
-------------------------------------------
引用：
    https://experimentationground.wordpress.com/2016/09/26/digit-recognition-from-google-street-view-images/
下载地址：
    https://www.kaggle.com/datasets/stanfordu/street-view-house-numbers
详细信息：

    - GrayScale size： 1 x 32 x 32
    - number of classes： 10
    - classes： 10 numbers 
    - train number : 73257
    - test number : 26032 
    - extra number : 531131
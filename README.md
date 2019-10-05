# skin-classification
所选项目：皮肤感染情况检测

文件说明：data文件夹下保存的是所有训练及测试图片；model.pkl是训练好的模型；test.py是测试文件；cnn_classify.py是训练模型所用的文件

使用框架：pytorch，resnet18

测试代码需要库支持：import torch
from torchvision import datasets, models, transforms
import os, sys
import copy
import shutil
import csv

模型算法：使用了pytorch的迁移学习，导入resnet18，同时将皮肤的数据分成训练集和测试集。训练集10000张未感染寄生虫图片和10000张感染寄生虫图片，测试集1000张未感染寄生虫图片和1000张感染寄生虫图片。cnn_classify就是训练模型所使用文件。

使用的神经网络：resnet18，残差神经网络。使用的卷积层为3*3，卷积核大小为1。激活函数为relu。

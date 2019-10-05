from __future__ import print_function, division

import torch
from torchvision import datasets, models, transforms
import os, sys
import copy
import shutil
import csv

def test(filepath):
    isExists = os.path.exists('./classifier')
    if (isExists == False):
        os.makedirs('./classifier')
    else:
        shutil.rmtree('./classifier')
    shutil.copytree(filepath, './classifier/test')

    # 对数据格式进行转换，变成torch格式的数据
    test_data=datasets.ImageFolder(os.path.join('','./classifier'),transforms.Compose([
            transforms.Resize(128),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]))
    test_loader=torch.utils.data.DataLoader(test_data,shuffle=False,num_workers=4)

    #获取测试文件名
    file_name=[]
    path = "./classifier/test"  # 待读取的文件夹
    path_list = os.listdir(path)
    path_list.sort()  # 对读取的路径进行排序
    for filename in path_list:
        if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.JPG'):
            file_name.append(filename)

    test_model=torch.load('model.pkl') #加载已经训练好的模型
    test_model.eval()

    csvfile = open("result.csv", "w+")
    try:
        writer = csv.writer(csvfile)
        writer.writerow(('file_name', 'class'))
    except IOError as e:
        print(e)

    #对图片进行分类
    for batch_idx,(data, target) in enumerate(test_loader):
        outputs = test_model(data)
        _, preds = torch.min(outputs, 1)
        if preds.data.numpy()==[1]:
            Class=1
        else:
            Class=0
        #写入结果
        try:
            writer.writerow((file_name[batch_idx],Class))
        except IOError as e:
            print(e)

    csvfile.close()

    print('测试完成')


# print(sys.argv)
if __name__ == '__main__':
    tmp = sys.argv
    if len(tmp) < 2:
        print('输入测试路径名')
    else:
        filepath = tmp[1] # filepath为测试路径名（该路径名下有很多图片）
        print(filepath)
        test(filepath)

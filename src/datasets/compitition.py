import os
import torch
import torchvision
import wilds
from torch.utils.data import Dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from PIL import Image

class CustomDataset(Dataset):
    "自定义比赛提交要用的dataser,id就是图片的名字"
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = os.listdir(root_dir)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        file_name=self.image_paths[idx]
        img_path = os.path.join(self.root_dir,file_name)
        image = Image.open(img_path).convert("RGB")
        
        label = int(file_name.split("_")[1].split(".")[0])

        if self.transform:
            image = self.transform(image)

        return image, label
    

def ret_class_name_dic()->dict:
    """返回动物名字到数字和数字映射到动物名的字典"""
    classes = open('datasets/data/compitition/classname.txt').read().splitlines()#这是一个包含所有类的列表
    class_name_dic_num={}
    class_name_dic_name={}
    for i in classes:
        name,idx = i.split(' ')
        c = name
        if c.startswith('Animal'):
            c = c[7:]
        if c.startswith('Thu-dog'):
            c = c[8:]
        if c.startswith('Caltech-101'):
            c = c[12:]
        if c.startswith('Food-101'):
            c = c[9:]
        if c not in class_name_dic_name:
            class_name_dic_name[c]=idx
        else:
            print(name,"already exist!!")
    return class_name_dic_name



class Compitition:
    test_subset = None

    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=4,
                 subset='test',
                 classnames=None,
                 **kwargs):

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_location = os.path.join(location, 'compitition', 'train')
        self.train_dataset = torchvision.datasets.ImageFolder(
            root=self.train_location, transform=preprocess)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers)
        if self.test_subset=="up":
            self.test_location = os.path.join(location, 'compitition',
                                            'TestSetA')
            self.test_dataset=CustomDataset(self.test_location, transform=preprocess)
            self.test_loader = torch.utils.data.DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            pass
        else:
            self.test_location = os.path.join(location, 'compitition',
                                            self.test_subset)
            print("Loading Test Data from ", self.test_location)
            self.test_dataset = torchvision.datasets.ImageFolder(
                root=self.test_location, transform=preprocess)
            self.test_loader = torch.utils.data.DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)
        #简历类名列表
        names=[]
        for dir_name in self.train_dataset.classes:
            dir_name,set_name=dir_name.split("&")
            dir_name=dir_name.replace("_"," ")
            if set_name=="Thu-dog":
                dir_name=dir_name.title()
            elif set_name=="Food-101":
                dir_name.lower()
            elif set_name=="Animal":
                dir_name.lower()
            elif set_name=="Caltech-101":
                pass
            names.append(dir_name)
            
        self.classnames=names
        self.ret_class_name_dic=ret_class_name_dic()
    
    def convert_id_to_name(self,prob_list):
        "将默认的类默认数字变为文件名,再将文件名改为比赛对应的数字"
        ret_list=[]
        for i in prob_list:
            dir_name=self.train_dataset.classes[i]
            # print('dir_name',dir_name)

            class_name=dir_name.split("&")[0]
            class_id=self.ret_class_name_dic[class_name]
            ret_list.append(class_id)
            # print('class_id',class_id)
        # print('ret_list',ret_list)
        return ret_list


class CompititionVal(Compitition):
    def __init__(self, *args, **kwargs):
        self.test_subset = 'val'
        super().__init__(*args, **kwargs)


class CompititionTest(Compitition):
    def __init__(self, *args, **kwargs):
        self.test_subset = 'test'
        super().__init__(*args, **kwargs)

class CompititionUpload(Compitition):
    def __init__(self, *args, **kwargs):
        self.test_subset = 'up'
        super().__init__(*args, **kwargs)
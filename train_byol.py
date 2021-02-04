import paddle
import paddle.nn as nn
import os
import numpy as np
import matplotlib.pyplot as plt
import byol.transforms 
import paddle.vision.transforms as transforms
from PIL import Image
import BYOL
from resnet import ResNet50
from paddle.vision.datasets import ImageFolder
import argparse
import paddle.fluid as fluid
import time
from paddle.vision.datasets import Cifar10



parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')

parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--bs', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')





def reader_decorator(reader):
    def __reader__():
        for item in reader():
            img = np.array(item[0]).astype('float32').reshape(3, 224, 224)
            label = np.array(item[1]).astype('int64').reshape(1)
            yield img, label

    return __reader__





def main():
    args = parser.parse_args()

    # model

    resnet = ResNet50()
    model = BYOL.BYOL(resnet)
    optimizer = paddle.optimizer.Adam(learning_rate=args.lr,
        parameters=model.parameters())

    # data                    

    root = os.getcwd()
    traindir = os.path.join(root, 'testimg')


    # augmentation utils

    normalize = transforms.Normalize(mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                     std=[0.229 * 255, 0.224 * 255, 0.225 * 255], data_format='HWC')

    augmentation = [
        byol.transforms.RandomApply([
            transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
        ],p = 0.3),
        byol.transforms.RandomGrayscale(p=0.2),
        byol.transforms.RandomApply([byol.transforms.GaussianBlur((1.0, 2.0))],p=0.2),
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        normalize
    ]



    
    byoltransforms = byol.transforms.TwoCropsTransform(transforms.Compose(augmentation))

    cifar10_train = Cifar10(mode='train', transform=byoltransforms)
    train_loader = paddle.io.DataLoader(cifar10_train,
                                        shuffle=True,
                                        batch_size=args.bs)


    for epoch in range(args.epochs):
        train(train_loader, model, optimizer, epoch, args)
            
    
def train(train_loader, model, optimizer, epoch, args):
    model.train()
    f = open('out.txt', 'a')
    for batch_id, images in enumerate(train_loader()):
        img1, img2 = paddle.split(images[0], num_or_sections=2, axis=1)
        img1 = paddle.transpose(img1, perm=[0,3,1,2])
        img2 = paddle.transpose(img2, perm=[0,3,1,2])
        loss = model(img_one = img1, img_two = img2)
        dy_out = loss.numpy()
        # compute gradient and do Adam step
        optimizer.clear_grad()
        loss.backward()
        optimizer.step()
        model.update_moving_average() # update moving average of target encoder
        if batch_id % 50 == 0:
            item = "[Epoch %d, batch %d] loss: %.5f" % (epoch, batch_id, dy_out)
            print(item)
            f.write(str(item)+'\n')
    f.close()



        





if __name__ == '__main__':
    main()
    # train()

















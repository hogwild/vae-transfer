import os
from PIL import Image
import numpy as np
# import cv2
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def default_loader(path):
    return Image.open(path).convert('RGB')

class Dataset(data.Dataset):
    def __init__(self,dataPath,loadSize,fineSize,test=False,video=False):
        super(Dataset,self).__init__()
        self.dataPath = dataPath
        self.image_list = [x for x in os.listdir(dataPath) if is_image_file(x)]
        self.image_list = sorted(self.image_list)
        if(video):
            self.image_list = sorted(self.image_list)
        # if not test:
        #     self.transform = transforms.Compose([
        #     		         transforms.Resize(fineSize),
        #     		         transforms.RandomCrop(fineSize),
        #                      transforms.RandomHorizontalFlip(),
        #     		         transforms.ToTensor()])
        # else:
        #     self.transform = transforms.Compose([
        #     		         transforms.Resize(fineSize),
        #     		         transforms.ToTensor()])

        self.test = test
        self.fineSize = fineSize

    def transform(self, img):
        # img = cv2.resize(img, (self.fineSize, self.fineSize))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.resize((self.fineSize, self.fineSize))
        x = torch.from_numpy(np.asarray(img, dtype=np.float32).transpose((2, 0, 1))) / 127.5 - 1.
        # x = torch.unsqueeze(x, 0)
        return x


    def __getitem__(self,index):
        dataPath = os.path.join(self.dataPath,self.image_list[index])

        Img = default_loader(dataPath)
        ImgA = self.transform(Img)

        imgName = self.image_list[index]
        imgName = imgName.split('.')[0]
        return ImgA,imgName

    def __len__(self):
        return len(self.image_list)


if __name__=="__main__":

    content_path = "../dataset/tsushima_yoshiko/"
    loadSize = 128
    fineSize = 128
    batchSize = 32

    content_dataset = Dataset(content_path, loadSize, fineSize)
    content_loader_ = data.DataLoader(dataset = content_dataset,
                                      batch_size = batchSize,
                                      shuffle = True,
                                      num_workers = 0,
                                      drop_last = True)
    content_loader = iter(content_loader_)
    for i in range(10):
        # content, name = content_loader.next()
        # print(content.shape, name)
        try: 
            content, _ = content_loader.next()
        except IOError:
            content, _ = content_loader.next()
        except StopIteration:
            content_loader = iter(content_loader_)
            content, _ = content_loader.next()
        except:
            continue


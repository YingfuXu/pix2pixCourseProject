import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms
from os import listdir
from os.path import join, splitext


# list1 = ["这", "是", "一个", "测试"]
# for index, item in enumerate(list1, 1): # 返回list中数据的索引和数据， 索引从1开始
#     print(index, item)

def tensor2image(input_image, imtype=np.uint8, return_numpy=False, save_image_dir='None', tensor_normalized=True):
    """"Converts a Tensor [C, H, W] into a numpy image array [H, W, C].
    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray): # if the input_image is not a numpy array

        if isinstance(input_image, torch.Tensor):  # if the input_image is a Tensor
            image_tensor = input_image.data # get the Tensor data from a variable
        else: # if the input_image is a not a Tensor and not a numpy array.
            print('Error! The input image is a neither a Tensor nor a numpy array! Returning input...')
            return input_image

        image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array, [C, H, W] 通道，图像高，图像宽

        # print('shape: ', image_numpy.shape)
        if image_numpy.shape[0] == 1:  # if this is a grayscale image
            # image_numpy = np.tile(image_numpy, (3, 1, 1)) # 通过重复image_numpy (3, 1, 1)次来构造出一个新数组
            if tensor_normalized: # 
                image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0 
                # transpose the image_numpy from [C, H, W] to [H, W, C]
                # post-processing: tranpose and scaling, because of the pixel data was normalized to -1 -> 1 in tensor
            else:    
                image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0 
            
            image_numpy = image_numpy[:, :, 0]
            # print('image_numpy shape: ', image_numpy.shape)
        elif image_numpy.shape[0] == 3: # if this is a 3-channel image
            if tensor_normalized: # 
                image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0 
                # transpose the image_numpy from [C, H, W] to [H, W, C]
                # post-processing: tranpose and scaling, because of the pixel data was normalized to -1 -> 1 in tensor
            else:    
                image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0 
            # print('image_numpy shape: ', image_numpy.shape)
        else:
            print('Error! The input image has neither 1 or 3 channels! Returning input...')
            return input_image

    else:  # if it is a numpy array, do nothing
        image_numpy = input_image

    image_numpy = image_numpy.astype(imtype)
    if return_numpy:
        return image_numpy.astype(imtype)

    image_pil = Image.fromarray(image_numpy) # Image.fromarray accept numpy array image in shape: [H, W, C]
    if save_image_dir != 'None':
        image_pil.save(save_image_dir)
    return image_pil

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])

# 返回的 transform_list 是对图像进行的全部transform的集合，然后调用 new_img = transform_list(params)(img)
# input PIL image RGB
def img_transform_list(target_size, convert2grayscale=False, convert2tensor=True, Normalize=True):
    transform_list = []
    transform_list.append(transforms.Resize(target_size))
    if convert2grayscale:
        transform_list.append(transforms.Grayscale(num_output_channels=1)) # 单通道灰度图

    if convert2tensor:
        transform_list += [transforms.ToTensor()] # [0,255] -> [0,1] # tensor: [C, H, W] 通道，图像高，图像宽
    else: # transfer to numpy array
        transform_list += [np.array]

    if Normalize:
        if convert2grayscale:
            transform_list += [transforms.Normalize(mean = (0.5,), std = (0.5,))]
        else:
            transform_list += [transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))]
            # channel=（channel-mean）/std -> [-1,1]
    return transforms.Compose(transform_list)

# Inherit the data.Dataset and create a new dataset for getting each item easily
# 继承 data.Dataset 并且实现两个成员方法 __getitem__ 和 __len__
class Image_Dataset(Dataset):
    
    def __init__(self, dataset_name, target_img_size, data_dir, input_grayscale=False, target_grayscale=False):
        super(Image_Dataset, self).__init__()

        self.input_grayscale = input_grayscale
        self.target_grayscale = target_grayscale
        self.dataset_name = dataset_name
        self.input_dir = join(data_dir, "input")
        self.target_dir = join(data_dir, "target")
        self.target_img_size = target_img_size

        self.input_fullnames = listdir(self.input_dir) # os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表
        self.target_fullnames = listdir(self.target_dir)
        
        # print(self.input_fullnames, self.target_fullnames)
        self.input_image_fullnames = [x for x in self.input_fullnames if is_image_file(x)] 
        self.target_image_fullnames = [x for x in self.target_fullnames if is_image_file(x)] 
        
        # get the image numbers os.path.splitext; sort()的reverse = False 升序（默认）
        # self.input_image_fullnames.sort(key = lambda fullname: splitext(fullname)[0]) 
        # self.target_image_fullnames.sort(key = lambda fullname: splitext(fullname)[0]) 
        self.input_image_fullnames.sort(key = lambda fullname: int(fullname.split('_')[0])) # for the deblur dataset
        self.target_image_fullnames.sort(key = lambda fullname: int(fullname.split('_')[0]))
        # print(self.input_image_fullnames, self.target_image_fullnames)

        # if self.input_image_fullnames != self.target_image_fullnames:
        #     print('Error! Image pairs names do not match!')
        # else:
        #     self.image_fullnames = self.input_image_fullnames

        
    def __getitem__(self, index):
        if self.input_grayscale:
            input_image_pil = Image.open(join(self.input_dir, self.input_image_fullnames[index])).convert('L')
            input_img_tensor = img_transform_list(self.target_img_size, Normalize=True, convert2grayscale=True)(input_image_pil)
        else:
            input_image_pil = Image.open(join(self.input_dir, self.input_image_fullnames[index])).convert('RGB')
            input_img_tensor = img_transform_list(self.target_img_size, Normalize=True)(input_image_pil)

        if self.target_grayscale:
            target_image_pil = Image.open(join(self.target_dir, self.target_image_fullnames[index])).convert('L')
            target_img_tensor = img_transform_list(self.target_img_size, Normalize=True, convert2grayscale=True)(target_image_pil)
        else:
            target_image_pil = Image.open(join(self.target_dir, self.target_image_fullnames[index])).convert('RGB')
            target_img_tensor = img_transform_list(self.target_img_size, Normalize=True)(target_image_pil)
        # print(img.getbands(),img.size) # ('L',) ('R', 'G', 'B')
        
        filename = self.input_image_fullnames[index]
        # print(filename)

        return input_img_tensor, target_img_tensor, filename

    def __len__(self):
        return len(self.input_image_fullnames)


# if __name__ == '__main__':
#     dataset_direction = './images/'
#     dataset_name = 'city'
#     output_direction = './output/'
#     target_image_size = (256, 256)
#     threads = 4
#     batch_size = 1

#     dataset = Image_Dataset(dataset_name, target_image_size, dataset_direction, target_grayscale=True)
#     training_data_loader = DataLoader(dataset, num_workers=threads, batch_size=batch_size, shuffle=False) # shuffle: 是否重新排序

#     for index, item in enumerate(training_data_loader, 1): # 返回list中数据的索引和数据， 索引从1开始
#         # print('for loop, %d', item)
#         image_name = item[2]
#         print(item[1].size()) # (1, 3, 256, 256)
#         input_img_tensor = item[0].squeeze(0) # 将输入张量形状中的1 去除并返回。如果输入是形如(A×1×B×1×C×1×D)，那么输出形状就为： (A×B×C×D) 当给定dim时，那么挤压操作只在给定维度上。
#         output_img_tensor = item[1].squeeze(0)
#         # print('index: %d, %s' % (index, image_name)) # item[1] is the file name here
#         if index == 8:
#             # print(image_name) # __getitem__ returns a tuple(元组)
#             print(output_img_tensor.size())
#             input_img = tensor2image(input_img_tensor, imtype=np.uint8, return_numpy=False, save_image_dir=output_direction+image_name[0], tensor_normalized=True)
#             output_img = tensor2image(output_img_tensor, imtype=np.uint8, return_numpy=False, save_image_dir=output_direction+image_name[0], tensor_normalized=True)
#             input_img.show()
#             output_img.show()

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
import glob
from os import listdir


def plot_PSNRs_SSIMs(data_dirs):

    PSNR_epochs = []
    SSIMs_epochs = []
    PSNRs = []
    SSIMs = []

    for data_dir in data_dirs:
        PSNR_log_name = data_dir + 'PSNR_log.txt'
        PSNR_log = np.loadtxt(PSNR_log_name)
        PSNR_epochs_list = list(PSNR_log[:, 0])
        PSNRs_list = list(PSNR_log[:, 1])
        PSNR_epochs += [PSNR_epochs_list]
        PSNRs += [PSNRs_list]


        SSIM_log_name = data_dir + 'SSIM_log.txt'
        SSIM_log = np.loadtxt(SSIM_log_name)
        SSIMs_epochs_list = list(SSIM_log[:, 0])
        SSIMs_list = list(SSIM_log[:, 1])
        SSIMs_epochs += [SSIMs_epochs_list]
        SSIMs += [SSIMs_list]

    plt.figure()

    plt.subplots_adjust(wspace =0.08, hspace =0.5)

    plt.subplot(2, 1, 1)
    plt.title('PSNR Results')
    PSNR_unet = plt.plot(PSNR_epochs[0], PSNRs[0], color='green', label='unet')
    PSNR_resnet = plt.plot(PSNR_epochs[1], PSNRs[1], color='blue', label='resnet_9blocks')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('PSNR')

    plt.subplot(2, 1, 2)
    plt.title('SSIM Results')
    SSIM_unet = plt.plot(SSIMs_epochs[0], SSIMs[0], color='green', label='unet')
    SSIM_resnet = plt.plot(SSIMs_epochs[1], SSIMs[1], color='blue', label='resnet_9blocks')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('SSIM')

    plt.show()

def plot_Loss(data_dirs):

    Loss_epochs = []
    iterations = []
    D_Losses = []
    GAN_Losses = []
    L1_Losses = []


    for data_dir in data_dirs:
        Loss_log_name = data_dir + 'Loss_log.txt'
        Loss_log = np.loadtxt(Loss_log_name)

        Loss_epochs_list = list(Loss_log[:, 0])
        Losses_iterations_list = list(Loss_log[:, 1])
        Losses_D_list = list(Loss_log[:, 2])
        Losses_GAN_list = list(Loss_log[:, 3])
        Losses_L1_list = list(Loss_log[:, 4])

        total_iterations_array = Loss_log[:, 0:2]
        # print(total_iterations_array.shape[0])
        # print(Loss_epochs_list)
        # calculate iterations
        total_train_data = 22509
        Losses_iterations = np.zeros(total_iterations_array.shape[0])
        for i in range(total_iterations_array.shape[0]):
            Losses_iterations[i] = (total_iterations_array[i, 0] - 1) * total_train_data + total_iterations_array[i, 1]
        
        # print(Losses_iterations)

        # print(Losses_iterations.shape)
        Losses_iterations_list = list(Losses_iterations)
        # print(len(Losses_iterations_list))
        # a
        # PSNR_epochs += [PSNR_epochs_list]
        # PSNRs += [PSNRs_list]
        iterations += [Losses_iterations_list]
        D_Losses += [Losses_D_list]
        GAN_Losses += [Losses_GAN_list]
        L1_Losses += [Losses_L1_list]


    # print(iterations[0])
    # print(D_Losses[0])


    plt.figure()

    plt.subplots_adjust(wspace =0.08, hspace =0.5)

    plt.subplot(2, 1, 1)
    plt.title('Loss Results')
    plt.plot(iterations[0], L1_Losses[0], color='red', label='L1_Loss')
    plt.plot(iterations[0], GAN_Losses[0], color='blue', label='GAN_Loss')
    plt.plot(iterations[0], D_Losses[0], color='green', label='D_Loss')
    plt.xlim(0, 50000)
    plt.legend()
    # plt.xlabel('iteration')
    plt.ylabel('Loss')

    plt.subplot(2, 1, 2)
    # plt.title('Loss Results')
    # plt.plot(iterations[1], D_Losses[1], color='green', label='D_Losses')
    # plt.plot(iterations[1], GAN_Losses[1], color='blue', label='GAN_Losses')
    # plt.plot(iterations[1], L1_Losses[1], color='red', label='L1_Losses')
    plt.plot(iterations[0], D_Losses[0], color='green', label='D_Loss')
    plt.plot(iterations[0], GAN_Losses[0], color='blue', label='GAN_Loss')
    # plt.plot(iterations[0], L1_Losses[0], color='red', label='L1_Losses')
    plt.xlim(0, 50000)
    plt.ylim(0, 1.5)
    # plt.legend()
    plt.xlabel('iteration')
    plt.ylabel('Loss')


    plt.show()

def splice_epoch_images(image_dir):

    # plt.figure(figsize=(16,8))

    # plt.subplots_adjust(wspace =0.08, hspace =0.002)
    # plt.axis('off')

    # for num in range(1,10):
    #     filenum = num + 18
    #     img_name = "epoch_{}_predict_{}_A.png".format(2, filenum)
    #     # print(img_name)
    #     img = Image.open(image_dir+img_name)
    #     plt.subplot(1,9,num)
    #     plt.axis('off')
    #     # plt.title('img')

    #     plt.rcParams['savefig.dpi'] = (256, 256)
    #     plt.imshow(img)
    # plt.show()

    UNIT_SIZE = 369
    TARGET_WIDTH = 369*9 # 拼接完后的横向长度
    target = Image.new('RGB', (TARGET_WIDTH, UNIT_SIZE))
    left = 0
    right = UNIT_SIZE
    epoch = 0
    for num in range(1,10): 
        filenum = num + 18
        # img_name = "epoch_{}_predict_{}_A.png".format(epoch, filenum)
        img_name = "{}_A.png".format(filenum)
        # print(img_name)
        img = Image.open(image_dir+img_name)
        img = img.resize((UNIT_SIZE, UNIT_SIZE),Image.ANTIALIAS)
        print(img.size)
        target.paste(img, (left, 0, right, UNIT_SIZE))# 将image复制到target的指定位置中
        left += UNIT_SIZE # left是左上角的横坐标，依次递增
        right += UNIT_SIZE # right是右下的横坐标，依次递增
        quality_value = 100 # quality来指定生成图片的质量，范围是0～100

    target.save(image_dir+"epoch_{}.png".format(epoch), quality = quality_value)
    target.show()

    # for n in range(1,11):

    #     epoch = n * 2

    #     for num in range(1,10):
    #         filenum = num + 18
    #         img_name = "epoch_{}_predict_{}_A.png".format(epoch, filenum)
    #         print(img_name)

    #         apple = Image.open(image_dir+img_name)
    #         plt.subplot(10,9,num)
    #         plt.axis('off')
    #         # plt.title('apple')
    #         plt.imshow(apple)

    # plt.show()

def splice_test_images(image_dir, number_of_images, imageNameSave):

    input_fullnames = listdir(image_dir)

    # self.input_fullnames.sort(key = lambda fullname: int(splitext(fullname)[0])) 
    # input_fullnames.sort(key = lambda fullname: int(fullname.split('_')[3]))

    # print(input_fullnames)

    UNIT_SIZE = 369
    TARGET_WIDTH = 369*number_of_images # 拼接完后的横向长度
    target = Image.new('RGB', (TARGET_WIDTH, UNIT_SIZE))
    left = 0
    right = UNIT_SIZE
    epoch = 0
    for image_name in input_fullnames: 

        print(image_name)

        img = Image.open(image_dir+image_name)
        img = img.resize((UNIT_SIZE, UNIT_SIZE),Image.ANTIALIAS)
        # print(img.size)
        target.paste(img, (left, 0, right, UNIT_SIZE))# 将image复制到target的指定位置中
        left += UNIT_SIZE # left是左上角的横坐标，依次递增
        right += UNIT_SIZE # right是右下的横坐标，依次递增
        
    quality_value = 100 # quality来指定生成图片的质量，范围是0～100
    target.save(imageNameSave)
    target.show()    


if __name__ == '__main__':
    data_dirs = []
    data_dirs += ['./GPU_result/cGAN_deblur_simplified_unet/checkpoint/']
    data_dirs += ['./GPU_result/cGAN_deblur_simplified_resnet/checkpoint/']

    # plot_PSNRs_SSIMs(data_dirs)
    # plot_Loss(data_dirs)

    splice_epoch_images_dir = './GPU_result/cGAN_deblur_simplified_unet/checkpoint/image_for_comparison/'
    # splice_epoch_images(splice_epoch_images_dir)

    
    splice_test_images_dir = './GPU_result/cGAN_depth_simplified_resnet/checkpoint/samples_predict_images/YOLO/'
    # splice_test_images_dir = './test/input/YOLO/'
    splice_test_images_dir = './depth_New_test_sample/netG_unet_256_epoch_80_80_lr_0002/YOLO/'
    splice_test_images_dir = './GPU_result/cGAN_deblur_simplified_0005/checkpoint/visualSetSamples/YOLO/'
    splice_test_images_dir = './GPU_result/cGAN_deblur_simplified_unet/checkpoint/image_for_comparison/originBlur/'
    splice_test_images_dir = './GPU_result/cGAN_deblur_simplified_0005/checkpoint/visualSetSamples/YOLO/'

    splice_test_images_dir = './test/netG_unet_256_epoch_10_10_lr_0005/YOLO/'
    splice_test_images_dir = './depth_New_test_sample/netG_resnet_9blocks_epoch_80_80_lr_0002/YOLO/'
    
    splice_test_images(splice_test_images_dir, 8, 'depth_New_test_sample_netG_resnet_9blocks_epoch_80_80_lr_0002.png')
    
    

    



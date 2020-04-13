from libraries import *

# change the dataset in the current directory
dataset_dir = '../../../data/yingfu/blur/' # ../../../data/yingfu/blur/  ./datasets/
project = 'cGAN_deblur_simplified'
data_train_dir = dataset_dir + 'train'
data_test_dir = dataset_dir + 'test' 
data_visual_dir = dataset_dir + 'visualization'
project_dir = dataset_dir + project
checkpoints_dir = dataset_dir + project + '/checkpoint'

PSNR_log_name = checkpoints_dir + '/PSNR_log.txt'
SSIM_log_name = checkpoints_dir + '/SSIM_log.txt'
loss_log_name = checkpoints_dir + '/Loss_log.txt'
timer_log_name = checkpoints_dir + '/Timer_log.txt'
lr_log_name = checkpoints_dir + '/LR_log.txt'

netG_type = 'unet_256' # 'resnet_9blocks' 'resnet_6blocks' 'unet_256'

save_model_number_epoch = 10
print_loss_number_iteration = 100

batch_size = 1
test_batch_size = 1

# input image size to networks
target_img_size = (256, 256)

# number of channels
input_nc = 3
output_nc = 3
# the number of filters in the first conv layer
ngf = 64
ndf = 64

epoch_count = 1
n_epoch = 10 # for deblur dataset 22,509 pairs
n_epoch_decay = 10

lr=0.0002
lr_policy='lambda'
lr_decay_iters = 50
beta1 = 0.5 # momentum term of adam

threads = 4
# seed = 123
lambda_L1 = 100.0

# torch.manual_seed(seed)

# this is the cpu version:
# device = torch.device("cpu")

# if using the gpu, open the following code:
# torch.cuda.manual_seed(seed)
device = torch.device("cuda: 1")
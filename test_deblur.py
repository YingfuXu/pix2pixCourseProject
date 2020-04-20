from libraries import *
from params import *
from dataloader import *
from networks import define_G, define_D
from networkBase import GANLoss, get_scheduler, update_learning_rate

model_name = 'netG_unet_256_epoch_20_0_lr_0002'
model_path = "../pix2pixFiles/GPU_result/cGAN_deblur_simplified_0002NoDecay/checkpoint/" + model_name + ".pth"
data_test_dir = "../datasets/blur/new/test/"
test_set = Image_Dataset('test_set', target_img_size, data_test_dir, input_grayscale=False)

test_data_loader = DataLoader(dataset=test_set, num_workers=threads, batch_size=batch_size, shuffle=False)

net_g = torch.load(model_path, map_location='cpu')

criterionMSE = nn.MSELoss().to(device)

if not os.path.exists(data_test_dir + model_name):
    os.mkdir(data_test_dir + model_name)

sum_psnr = 0
sum_ssim = 0

for index, batch in enumerate(test_data_loader, 1):
    input_img_tensor, target_img_tensor = batch[0].to(device), batch[1].to(device)
    filename = batch[2][0]
    print(index)

    # input_save_path = checkpoints_dir + "/epochTestVisual/epoch_{}_input_{}".format(epoch, filename[0])
    # target_save_path = checkpoints_dir + "/epochTestVisual/epoch_{}_target_{}".format(epoch, filename[0])
    # prediction_save_path = data_test_dir + model_name + "/" + filename
    # print(prediction_save_path)
    
    prediction_img_tensor = net_g(input_img_tensor) # return tensor
    # print('prediction', prediction_img_tensor.size()
    # input_img_tensor = input_img_tensor.detach().squeeze(0).cpu()
    # target_img_tensor = target_img_tensor.detach().squeeze(0).cpu()
    # prediction_img_tensor = prediction_img_tensor.detach().squeeze(0).cpu()

    # input_img = tensor2image(input_img_tensor.detach().squeeze(0).cpu(), imtype=np.uint8, return_numpy=False, save_image_dir=input_save_path, tensor_normalized=True)
    # output_img = tensor2image(target_img_tensor.detach().squeeze(0).cpu(), imtype=np.uint8, return_numpy=False, save_image_dir=target_save_path, tensor_normalized=True)
    # prediction_img = tensor2image(prediction_img_tensor.detach().squeeze(0).cpu(), save_image_dir=prediction_save_path)
    
    # calculate PSNR in each testing image
    mse = criterionMSE(prediction_img_tensor, target_img_tensor)
    psnr = 10 * log10(1 / mse.item())
    sum_psnr += psnr
    # calculate SSIM in each testing image https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.compare_ssim
    prediction_img_np = tensor2image(prediction_img_tensor.detach().squeeze(0).cpu(), return_numpy=True)
    target_img_np = tensor2image(target_img_tensor.detach().squeeze(0).cpu(), return_numpy=True)
    # ssim = structural_similarity(prediction_img_np, target_img_np, multichannel=True)
    ssim = compare_ssim(prediction_img_np, target_img_np, multichannel=True)
    sum_ssim += ssim

PSNR_message = '===> Avg. PSNR: %.4f dB' % (sum_psnr / len(test_data_loader))
print(PSNR_message)  # print the message
# PSNR_message = '%d  %.4f' % (epoch, sum_psnr / len(visualization_data_loader))
# with open(PSNR_log_name, "a") as PSNR_log_file:
#     PSNR_log_file.write('%s\n' % PSNR_message)  # save the message

SSIM_message = '===> Avg. SSIM: %.4f' % (sum_ssim / len(test_data_loader))
print(SSIM_message)  # print the message
# SSIM_message = '%d  %.4f' % (epoch, sum_ssim / len(visualization_data_loader))
# with open(SSIM_log_name, "a") as SSIM_log_file:
#     SSIM_log_file.write('%s\n' % SSIM_message)  # save the message
        
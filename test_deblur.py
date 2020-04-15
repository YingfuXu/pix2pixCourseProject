from libraries import *
from params import *
from dataloader import *
from networks import define_G, define_D
from networkBase import GANLoss, get_scheduler, update_learning_rate

model_name = 'netG_unet_256_epoch_10_10_lr_0005'
model_path = "./GPU_result/cGAN_deblur_simplified_0005/checkpoint/" + model_name + ".pth"
data_test_dir = "./test/"
test_set = Image_Dataset('test_set', target_img_size, data_test_dir, input_grayscale=False)

test_data_loader = DataLoader(dataset=test_set, num_workers=threads, batch_size=batch_size, shuffle=False)

net_g = torch.load(model_path, map_location='cpu')

if not os.path.exists(data_test_dir + model_name):
    os.mkdir(data_test_dir + model_name)

for index, batch in enumerate(test_data_loader, 1):
    input_img_tensor, target_img_tensor = batch[0].to(device), batch[1].to(device)
    filename = batch[2][0]

    # input_save_path = checkpoints_dir + "/epochTestVisual/epoch_{}_input_{}".format(epoch, filename[0])
    # target_save_path = checkpoints_dir + "/epochTestVisual/epoch_{}_target_{}".format(epoch, filename[0])
    prediction_save_path = data_test_dir + model_name + "/" + filename
    print(prediction_save_path)
    
    prediction_img_tensor = net_g(input_img_tensor) # return tensor
    # print('prediction', prediction_img_tensor.size()
    # input_img_tensor = input_img_tensor.detach().squeeze(0).cpu()
    # target_img_tensor = target_img_tensor.detach().squeeze(0).cpu()
    # prediction_img_tensor = prediction_img_tensor.detach().squeeze(0).cpu()
    # input_img = tensor2image(input_img_tensor.detach().squeeze(0).cpu(), imtype=np.uint8, return_numpy=False, save_image_dir=input_save_path, tensor_normalized=True)
    # output_img = tensor2image(target_img_tensor.detach().squeeze(0).cpu(), imtype=np.uint8, return_numpy=False, save_image_dir=target_save_path, tensor_normalized=True)
    prediction_img = tensor2image(prediction_img_tensor.detach().squeeze(0).cpu(), save_image_dir=prediction_save_path)
        

# for image_name in image_filenames:
#     img = load_img(image_dir + image_name)
#     img = transform(img)
#     input = img.unsqueeze(0).to(device)
#     out = net_g(input)
#     out_img = out.detach().squeeze(0).cpu()

#     if not os.path.exists(os.path.join("result", dataset)):
#         os.makedirs(os.path.join("result", dataset))
#     save_img(out_img, "result/{}/{}".format(dataset, image_name))
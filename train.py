from libraries import *
from params import *
from dataloader import *
from networks import define_G, define_D
from networkBase import GANLoss, get_scheduler, update_learning_rate


if not os.path.exists(project_dir):
    os.mkdir(project_dir)
if not os.path.exists(checkpoints_dir):
    os.mkdir(checkpoints_dir)

# define training set
training_set = Image_Dataset('training_set', target_img_size, data_train_dir, input_grayscale=False)
# define visualization set (run the model in training to visualize how it works on a small number of the same images)
visual_set = Image_Dataset('visualization_set', target_img_size, data_visual_dir, input_grayscale=False)

training_data_loader = DataLoader(dataset=training_set, num_workers=threads, batch_size=batch_size, shuffle=True)
# testing_data_loader = DataLoader(dataset=DatasetFromFolder(dataroot_test, direction), num_workers=threads, batch_size=test_batch_size)
visualization_data_loader = DataLoader(dataset=visual_set, num_workers=threads, batch_size=test_batch_size, shuffle=False)

print('===> Building models')

'''loading the generator and discriminator'''
net_g = define_G(input_nc, output_nc, ngf, netG=netG_type, norm='batch', use_dropout=False, gpu_id=device)
net_d = define_D(input_nc + output_nc, ndf, 'n_layers', gpu_id=device)

'''set loss fn'''
criterionGAN = GANLoss().to(device)
criterionL1 = nn.L1Loss().to(device)
criterionMSE = nn.MSELoss().to(device)

'''setup optimizer''' 
optimizer_g = optim.Adam(net_g.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_d = optim.Adam(net_d.parameters(), lr=lr, betas=(beta1, 0.999))

'''set the learning rate adjust policy'''
net_g_scheduler = get_scheduler(optimizer_g)
net_d_scheduler = get_scheduler(optimizer_d)

'''training process'''
for epoch in range(epoch_count, n_epoch + n_epoch_decay + 1):

    epoch_start_time = time.time() # start timer in each epoch
    
    for iteration, batch in enumerate(training_data_loader, 1):
        # forward
        real_a, real_b = batch[0].to(device), batch[1].to(device)
        fake_b = net_g(real_a)

        ######################
        # (1) Update D network
        ######################

        optimizer_d.zero_grad()
        
        # D train with fake
        fake_ab = torch.cat((real_a, fake_b), 1) # real_a: (1, 3, 256, 256)
        pred_fake = net_d.forward(fake_ab.detach())
        loss_d_fake = criterionGAN(pred_fake, False)

        # D train with real
        real_ab = torch.cat((real_a, real_b), 1)
        pred_real = net_d.forward(real_ab)
        loss_d_real = criterionGAN(pred_real, True)
        
        # Combined D loss
        loss_d = (loss_d_fake + loss_d_real) * 0.5

        loss_d.backward()
       
        optimizer_d.step()

        ######################
        # (2) Update G network
        ######################

        optimizer_g.zero_grad()

        # First, G(A) should fake the discriminator
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = net_d.forward(fake_ab)
        loss_g_gan = criterionGAN(pred_fake, True)

        # Second, G(A) = B
        loss_g_l1 = criterionL1(fake_b, real_b) * lambda_L1
        
        loss_g = loss_g_gan + loss_g_l1
        
        loss_g.backward()

        optimizer_g.step()
        
        if iteration % print_loss_number_iteration == 0:
            print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_GAN: {:.4f} Loss_L1: {:.4f}".format(
                epoch, iteration, len(training_data_loader), loss_d.item(), loss_g_gan.item(), loss_g_l1.item()))
            loss_message = '%d  %d  %.4f    %.4f    %.4f' % (epoch, iteration, loss_d.item(), loss_g_gan.item(), loss_g_l1.item())
            with open(loss_log_name, "a") as loss_log_file:
                loss_log_file.write('%s\n' % loss_message)  # save the message

    # finished one epoch

    lr_g = update_learning_rate(net_g_scheduler, optimizer_g)
    lr_d = update_learning_rate(net_d_scheduler, optimizer_d)

    lr_message = '%d  %.4f    %.4f' % (epoch, lr_g, lr_d)

    with open(lr_log_name, "a") as lr_log_file:
        lr_log_file.write('%s\n' % lr_message)  # save the learning rate

    # run the net on test images and save
    sum_psnr = 0
    sum_ssim = 0
    for index, batch in enumerate(visualization_data_loader, 1):
        input_img_tensor, target_img_tensor = batch[0].to(device), batch[1].to(device)
        filename = batch[2]
        if not os.path.exists(checkpoints_dir + "/epochTestVisual"):
            os.mkdir(checkpoints_dir + "/epochTestVisual")

        # input_save_path = checkpoints_dir + "/epochTestVisual/epoch_{}_input_{}".format(epoch, filename[0])
        # target_save_path = checkpoints_dir + "/epochTestVisual/epoch_{}_target_{}".format(epoch, filename[0])
        prediction_save_path = checkpoints_dir + "/epochTestVisual/epoch_{}_predict_{}".format(epoch, filename[0])

        prediction_img_tensor = net_g(input_img_tensor) # return tensor
        # print('prediction', prediction_img_tensor.size()
        # input_img_tensor = input_img_tensor.detach().squeeze(0).cpu()
        # target_img_tensor = target_img_tensor.detach().squeeze(0).cpu()
        # prediction_img_tensor = prediction_img_tensor.detach().squeeze(0).cpu()

        # input_img = tensor2image(input_img_tensor.detach().squeeze(0).cpu(), imtype=np.uint8, return_numpy=False, save_image_dir=input_save_path, tensor_normalized=True)
        # output_img = tensor2image(target_img_tensor.detach().squeeze(0).cpu(), imtype=np.uint8, return_numpy=False, save_image_dir=target_save_path, tensor_normalized=True)
        prediction_img = tensor2image(prediction_img_tensor.detach().squeeze(0).cpu(), save_image_dir=prediction_save_path)
        
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

    PSNR_message = '===> Avg. PSNR: %.4f dB' % (sum_psnr / len(visualization_data_loader))
    print(PSNR_message)  # print the message
    PSNR_message = '%d  %.4f' % (epoch, sum_psnr / len(visualization_data_loader))
    with open(PSNR_log_name, "a") as PSNR_log_file:
        PSNR_log_file.write('%s\n' % PSNR_message)  # save the message

    SSIM_message = '===> Avg. SSIM: %.4f' % (sum_ssim / len(visualization_data_loader))
    print(SSIM_message)  # print the message
    SSIM_message = '%d  %.4f' % (epoch, sum_ssim / len(visualization_data_loader))
    with open(SSIM_log_name, "a") as SSIM_log_file:
        SSIM_log_file.write('%s\n' % SSIM_message)  # save the message

    # timer
    print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, n_epoch + n_epoch_decay, time.time() - epoch_start_time))
    timer_message = '%d  %.4f' % (epoch, time.time() - epoch_start_time)
    with open(timer_log_name, "a") as timer_log_file:
        timer_log_file.write('%s\n' % timer_message)  # save the message

    # checkpoint
    if epoch % save_model_number_epoch == 0:
        # print("10 more epochs finished!")
        # if not os.path.exists(checkpoints_dir):
        #     os.mkdir(checkpoints_dir)
        net_g_model_out_path = checkpoints_dir + "/netG_model_epoch_{}.pth".format(epoch)
        net_d_model_out_path = checkpoints_dir + "/netD_model_epoch_{}.pth".format(epoch)
        torch.save(net_g, net_g_model_out_path)
        torch.save(net_d, net_d_model_out_path)
        print("Checkpoint saved to {}".format(checkpoints_dir))

# save the model after finishing training
# if not os.path.exists(checkpoints_dir):
#     os.mkdir(checkpoints_dir)
net_g_model_out_path = checkpoints_dir + "/netG_{}_epoch_{}_{}_lr_{}.pth".format(netG_type, n_epoch, n_epoch_decay, lr)
net_d_model_out_path = checkpoints_dir + "/netD_{}_epoch_{}_{}_lr_{}.pth".format('n_layers', n_epoch, n_epoch_decay, lr)
torch.save(net_g, net_g_model_out_path)
torch.save(net_d, net_d_model_out_path)
print("Training Finished! Checkpoint saved to {}".format(checkpoints_dir))

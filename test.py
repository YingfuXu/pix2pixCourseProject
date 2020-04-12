from libraries import *

model_path = "checkpoint/{}/netG_model_epoch_{}.pth".format(dataset, nepochs)

net_g = torch.load(model_path).to(device)

image_filenames = [x for x in os.listdir(image_dir) if is_image_file(x)]

transform_list = [transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

transform = transforms.Compose(transform_list)

for image_name in image_filenames:
    img = load_img(image_dir + image_name)
    img = transform(img)
    input = img.unsqueeze(0).to(device)
    out = net_g(input)
    out_img = out.detach().squeeze(0).cpu()

    if not os.path.exists(os.path.join("result", dataset)):
        os.makedirs(os.path.join("result", dataset))
    save_img(out_img, "result/{}/{}".format(dataset, image_name))
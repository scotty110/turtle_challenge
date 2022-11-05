import torch
from pytorch_pretrained_biggan import (
    BigGAN,
    truncated_noise_sample,
    convert_to_images,
    one_hot_from_int,
)

import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image

import torchvision
import torchvision.transforms as transforms

# Make DataSet
from torch.utils.data import Dataset

class turtle_ds(Dataset):
    def __init__(self, device, num_images:int=200):
        '''
        Num_images round to nearest 10
        '''
        num_passes = int( num_images/10 )
        self.device = device 
        self.pair_list = make_dataset( self.device, num_passes )
        self.device = torch.device("cpu") #device 

    def __len__(self):
        return len(self.pair_list)

    def __getitem__(self, index):
        p = self.pair_list[index]
        x = p[0].to(self.device) 
        y = torch.squeeze( p[1].to(self.device), dim=0)
        #return x,y
        #pixel_values,labels
        return {'pixel_values':x, 'labels':y}


def save_pair(img_pair:list):
    import os
    dir_path = './training_data'
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    
    img = img_pair[0].cpu().numpy()
    img = (255*img).astype(np.uint8)
    img_save = Image.fromarray(np.transpose(img, (1, 2, 0)))
    img_save.save('{}/img.png'.format(dir_path))
    return 


def random_paste(background_image, turtle_image, min_scale=0.25, max_scale=0.65):
    """Randomly scales and pastes the turtle image onto the background image"""
    
    w, h = turtle_image.size
    # first, we will randomly downscale the turtle image
    new_w = int(random.uniform(min_scale, max_scale) * w)
    new_h = int(random.uniform(min_scale, max_scale) * h)
    resized_turtle_image = turtle_image.resize((new_w, new_h))

    # second, will randomly choose the locations where to paste the new image
    start_w = random.randint(0, w - new_w)
    start_h = random.randint(0, h - new_h)

    # third, will create the blank canvas of the same size as the original image
    canvas_image = Image.new('RGBA', (w, h))

    # and paste the resized turtle onto it, preserving the mask
    canvas_image.paste(resized_turtle_image, (start_w, start_h), resized_turtle_image)
    
    # Turtle image is of mode RGBA, while background image is of mode RGB;
    # `.paste` requires both of them to be of the same type.
    background_image = background_image.copy().convert('RGBA')
    # finally, will paste the resized turtle onto the background image
    background_image.paste(resized_turtle_image, (start_w, start_h), resized_turtle_image)
    return background_image, canvas_image


def make_dataset(device, num_passes:int=20):
    # Load turtle image
    turtle_image = Image.open('./turtle.png')
    turtle_image_256x256 = turtle_image.resize((256, 256))

    #device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # boilerplate pytorch code enforcing reproducibility
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # load the 256x256 model
    model = BigGAN.from_pretrained('biggan-deep-256').to(device).eval()

    # every time we will run with batch size of 3 in order to not run out of memory
    #num_passes = 10
    #batch_size = 3

    # TODO
    #num_passes = 20
    batch_size = 10

    # default noise value from the provided repository
    truncation = 0.4

    background_images = []

    for _ in range(num_passes):
        # BigGAN uses imagenet and hence each time we will choose one of 1000 categories
        class_vector = torch.from_numpy(
            one_hot_from_int(np.random.randint(0, 1000, size=batch_size), batch_size=batch_size)
        ).to(device)
        noise_vector = torch.from_numpy(
            truncated_noise_sample(truncation=truncation, batch_size=batch_size)
        ).to(device)

        # Generate the images and convert them to PIL image
        with torch.no_grad():
            output = model(noise_vector, class_vector, truncation).cpu()
            background_images.extend(convert_to_images(output))

    # We won't need the GAN model anymore,
    # so we can safely delete it and free up some memory
    del model
    torch.cuda.empty_cache()

    print("We generated {} images".format(len(background_images)))

    tensor_transform = transforms.ToTensor()
    training_set = []  # image, segmentation mask

    for background_image in background_images:
      # paste the turtle onto background image
      aug_image, aug_mask = random_paste(background_image.copy(), turtle_image_256x256.copy())
      # convert PIL images to pytorch tensors
      training_pair = [
          tensor_transform(aug_image)[:3],  # keep the rgb only
          # For the mask, we only need the last (4th) channel,
          # and we will encode the mask as boolean
          (tensor_transform(aug_mask)[-1:] > 0).long(), #TODO
      ]
      training_set.append(training_pair)
    return training_set 



def get(device):
    #device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    training_ds = turtle_ds(device, 200)
    eval_ds = turtle_ds(device, 50)

    return training_ds, eval_ds





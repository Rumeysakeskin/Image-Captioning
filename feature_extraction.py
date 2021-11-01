
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from Inception import inception_v3
from torch.autograd import Variable
torch.set_default_tensor_type('torch.cuda.FloatTensor')

from encoder import get_encoder
#from autoencoder import *
from decoder import get_decoder

from utils import *
from create_vocabulary import *
from tokenization import *

import numpy as np
from tqdm import tqdm
import time
torch.manual_seed(12321)
torch.cuda.manual_seed(12321)
torch.cuda.manual_seed_all(12321)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Load images
val_captions, val_image_names = load_mscoco_annotations_val()
train_captions, train_image_names = load_mscoco_annotations_train(root='Resized')

batch_size = 128
feature_size = 2048
# Image feature extraction
preprocess = transforms.Compose([
    transforms.Resize(299), #299
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

encoder = inception_v3(pretrained=True)
encoder = encoder.to(device)
image_batch = torch.zeros(batch_size, 3, 299, 299)

for ids, im_path in tqdm(enumerate(train_image_names)):
    encoder.eval()
    im = load_image(im_path)
    im = gray_to_rgb(im)
    im = preprocess(im)

    im_path = im_path.split("Resized/train2017/")[1]
    image_id = im_path[:-4]
    batch_index = ids % batch_size
    image_tensor = im.unsqueeze(0)
    
    with torch.no_grad():
        feature_tensor = encoder(image_tensor.to(device)).squeeze(0)

    torch.save(feature_tensor, f"inception_v3_features/features_{image_id}.pt")

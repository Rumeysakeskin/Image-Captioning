import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from Inception import inception_v3
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

val_captions, val_image_names = load_mscoco_annotations_val()
train_captions, train_image_names = load_mscoco_annotations_train(root='Resized')


voc = Voc(name="Vocabulary")

train_normalized_captions = normalizeAllCaptions(train_captions)

print()
print("Creating Vocabulary...")
for caption in tqdm(train_normalized_captions):
    voc.addCaption(caption=caption)

voc.trim(min_count=77)
tokenized_train_captions = tokenize(voc, train_normalized_captions)
voc_size = len(voc.index2word)
core_img_path = []
core_img_capt_tokens = []
for img_path, img_cap in zip(train_image_names, tokenized_train_captions):

    if 15 >= len(img_cap) >= 9:
        core_img_path.append(img_path)
        core_img_capt_tokens.append(img_cap)
len_captions = np.zeros((70,), dtype=int)
for caption in tqdm(tokenized_train_captions):
    len_captions[len(caption)] += 1
print(len_captions)
train_captions_tokens = torch.LongTensor(pad_sequences(core_img_capt_tokens))
print(train_captions_tokens.shape)
train_image_names = core_img_path
voc.save_vocabulary()

batch_size = 128 # 128
hidden_size = 128
output_size = voc_size  # num words
embed_size = 128
feature_size = 2048
num_layers = 1
preprocess = transforms.Compose([
    transforms.Resize(299), #299
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# encoder = get_encoder()
encoder = inception_v3(pretrained=True)
encoder = encoder.to(device)
decoder = torch.load('merge_1_layer_40.pth') #get_decoder(output_size=output_size)
decoder = decoder.to(device)

def train_step(tokens_tensor, image_tensor, decoder, decoder_optimizer, encoder, criterion):
    decoder_optimizer.zero_grad()
    # encoder_optimizer.zero_grad()
    batch_size = tokens_tensor.size(0)
    sequence_length = tokens_tensor.size(1)

    loss = 0
    #with torch.no_grad():
    #    feature_tensor = encoder(image_tensor.to(device))

    feature_tensor = image_tensor.to(device)
    feature_tensor = feature_tensor.unsqueeze(0)

    decoder_hidden = torch.zeros(num_layers, batch_size, hidden_size).to(device)

    #for i in range(num_layers):
    #  decoder_hidden[i] = feature_tensor

    # decoder_hidden = feature_tensor.unsqueeze(0)
    for seq in range(sequence_length - 1):
        input_tensor = tokens_tensor[:, seq]
        input_tensor = input_tensor.unsqueeze(0)
        input_tensor = input_tensor.type(torch.LongTensor).to(device)
        output = tokens_tensor[:, seq + 1].type(torch.LongTensor).to(device)

        decoder_output, decoder_hidden = decoder(input_tensor, decoder_hidden, feature_tensor)
        loss += criterion(decoder_output, output)

    loss.backward()
    decoder_optimizer.step()
    return loss.item() / sequence_length

def generate_caption(image_tensor, encoder, decoder, max_len=30):
    #image_tensor = image_tensor.unsqueeze(0)
    #with torch.no_grad():
    #    feature = encoder(image_tensor.to(device))

    feature = image_tensor.to(device)
    feature = feature.unsqueeze(0)

    decoder_hidden = torch.zeros(num_layers, 1, hidden_size).to(device)
    feature =  feature.unsqueeze(0)
    #for i in range(num_layers):
    #    decoder_hidden[i] = feature

    # input = torch.tensor(voc.word2index["soc"]).type(torch.LongTensor).to(device)
    decoder_input = torch.ones(1, 1).type(torch.LongTensor).to(device)
    caption = ""
    for i in range(max_len):
        out, decoder_hidden = decoder(decoder_input, decoder_hidden, feature)
        out = out.argmax(dim=1)

        caption += voc.index2word[int(out)] + " "

        decoder_input = out.unsqueeze(0)

    print(caption)


def train(encoder, decoder, batch_size=128, n_iters=10):
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder = encoder.to(device)
    decoder = decoder.to(device)
    decoder_optimizer = optim.Adam(decoder.parameters())
    compare_loss = 99999999
    criterion = nn.NLLLoss()
    batch_index = 0
    data_sample = train_captions_tokens.shape[0]

    image_batch = torch.zeros(batch_size, feature_size)  # feature_size= 2048
    #image_batch = torch.zeros(batch_size, 3, 299, 299)
    caption_batch = torch.zeros(batch_size, 16)

    for iters in tqdm(range(1, n_iters + 1)):
        for ids, im_path in tqdm(enumerate(train_image_names)):
            encoder.eval()
            decoder.train()
            #im = load_image(im_path)
            #im = gray_to_rgb(im)
            #im = preprocess(im)
            im_path = im_path.split("Resized/train2017/")[1]
            image_id = im_path[:-4]

            image = torch.load(f"inception_v3_features/features_{image_id}.pt")

            batch_index = ids % batch_size
            image_batch[batch_index] = image
            caption_batch[batch_index] = train_captions_tokens[ids]
        
            if ids == len(train_image_names) - 1:
                loss = train_step(tokens_tensor=caption_batch[:batch_index + 1], image_tensor=image_batch[:batch_index + 1], decoder=decoder, decoder_optimizer=decoder_optimizer, encoder=encoder, criterion=criterion)
                print_loss_total += loss
                batch_index = 0
                break
        
            if batch_index == batch_size -1 :
                loss = train_step(tokens_tensor=caption_batch, image_tensor=image_batch, decoder=decoder, decoder_optimizer=decoder_optimizer, encoder=encoder, criterion=criterion)
                print_loss_total += loss

            if ids % (batch_size + 20) == 0:
                print_loss_avg = print_loss_total / (batch_size + 20)
                print_loss_total = 0
                print(print_loss_avg)
                print()
                encoder.eval()
                decoder.eval()
                generate_caption(image_batch[0], encoder=encoder, decoder=decoder)
        torch.save(decoder, f"merge_1_layer_{iters+40}.pth")
                # torch.save(encoder, "best_encoder.pth")

train(encoder=encoder, decoder=decoder, batch_size=batch_size)


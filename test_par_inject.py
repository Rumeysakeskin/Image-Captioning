
import torch.nn.functional as F
import torch.nn as nn
import torch
from torchvision.transforms import transforms
from utils import *
from create_vocabulary import *
from glob import glob
from tqdm import tqdm
from Inception import inception_v3
torch.set_default_tensor_type('torch.cuda.FloatTensor')


class Test(nn.Module):
    def __init__(self, iters):
        super(Test, self).__init__()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.voc = Voc(name="Vocabulary")
        self.voc.load_vocabulary()
        self.num_layers = 1
        self.hidden_size = 128

        im_path = "Resized/val2017/000000579070.jpg"
        decoder_path = f"par_inject_1_layer_{iters}.pth"
        self.preprocess = transforms.Compose([
            transforms.Resize(299),  # 299
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        im = self.preprocess(gray_to_rgb(load_image(im_path)))
        encoder = inception_v3(pretrained=True)
        decoder = torch.load(decoder_path, map_location=device)
        # print(encoder)
        # print(decoder)
        self.encoder = encoder.eval()
        self.decoder = decoder.eval()

        self.val_images = glob('Resized/val2017/*.jpg')
        self.data = []

        self.test_(iters)


    def generate_caption(self,image_tensor, encoder, decoder, max_len=16):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        image_tensor = image_tensor.unsqueeze(0)
        with torch.no_grad():
            feature = encoder(image_tensor.to(device))
        "Does not work..."
        decoder_hidden = torch.zeros(self.num_layers, 1, self.hidden_size).to(device)
        feature = feature.unsqueeze(0)
        # for i in range(num_layers):
        #    decoder_hidden[i] = feature

        # input = torch.tensor(voc.word2index["soc"]).type(torch.LongTensor).to(device)
        decoder_input = torch.ones(1, 1).type(torch.LongTensor).to(device)
        caption = ""
        for i in range(max_len):
            with torch.no_grad():
                out, decoder_hidden = decoder(decoder_input, decoder_hidden, feature)
            out = out.argmax(dim=1)
            if out == 2:
                break
            caption += self.voc.index2word[str(int(out))] + " "

            decoder_input = out.unsqueeze(0)

        return caption

    def test_(self, iters):


        for image_path in tqdm(self.val_images):
            im = self.preprocess(gray_to_rgb(load_image(image_path)))

            im_path = image_path.split('/')
            im_id = int(im_path[2][:12])

            #im_id = image_path[:-4]

            caption = self.generate_caption(image_tensor=im, encoder=self.encoder, decoder=self.decoder)
            # print(caption)
            # show_image(load_image(image_path))
            self.data.append({
                "image_id": im_id,
                "caption": caption
            })

        json_file = "results" + "/" + f"par_inject_1_layer_{iters}-epoch" + ".json"
        with open(json_file, "w") as file:
            json.dump(self.data, file)


def get_test(iters = 0):
    return Test(iters = iters)
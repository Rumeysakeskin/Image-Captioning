import torch
from torchvision.transforms import transforms
from utils import *
from create_vocabulary import *
from glob import glob
from tqdm import tqdm
from Inception import inception_v3
torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
voc = Voc(name="Vocabulary")
voc.load_vocabulary()
im_path = "Resized/val2017/000000579070.jpg"
# encoder_path = "VAEs/best_original.pt"
decoder_path = "merge_decoder_10.pth"
preprocess = transforms.Compose([
    transforms.Resize(299), #299
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

im = preprocess(gray_to_rgb(load_image(im_path)))
encoder = inception_v3(pretrained=True)
decoder = torch.load(decoder_path, map_location=device)
# print(encoder)
# print(decoder)
encoder = encoder.eval()
decoder = decoder.eval()

num_layers = 1
hidden_size = 128  # 2048
def generate_caption(image_tensor, encoder, decoder, max_len=16):
    image_tensor = image_tensor.unsqueeze(0)
    with torch.no_grad():
         feature = encoder(image_tensor.to(device)) # encoder.extract_feature(image_tensor.to(device))
    "Does not work..."
    decoder_hidden = torch.zeros(num_layers, 1, hidden_size).to(device)
    feature = feature.unsqueeze(0)
    # for i in range(num_layers):
    #     decoder_hidden[i] = feature

    # input = torch.tensor(voc.word2index["soc"]).type(torch.LongTensor).to(device)
    decoder_input = torch.ones(1, 1).type(torch.LongTensor).to(device)
    caption = ""
    for i in range(max_len):
        with torch.no_grad():
            out, decoder_hidden = decoder(decoder_input, decoder_hidden, feature)
        out = out.argmax(dim=1)
        if out == 2:
            break
        caption += voc.index2word[str(int(out))] + " "

        decoder_input = out.unsqueeze(0)

    return caption
val_images = glob('Resized/val2017/*.jpg')
data = []
for image_path in tqdm(val_images):
    im = preprocess(gray_to_rgb(load_image(image_path)))
    im_path = image_path.split('/')
    im_id = int(im_path[2][:12])
    caption = generate_caption(image_tensor=im, encoder=encoder, decoder=decoder)
    # print(caption)
    # show_image(load_image(image_path))
    data.append({
            "image_id": im_id,
            "caption" : caption
        })

json_file = "results" + "/" + "merge_epoch-10-results" + ".json"
with open(json_file, "w") as file:
    json.dump(data, file)




import os
import sys
from tqdm import tqdm
import requests 
from zipfile import ZipFile
import json
from PIL import Image
import matplotlib.pyplot as plt

def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    print("Downloading... " + url + " to " + save_path)
    with open(save_path, 'wb') as fd:
        for chunk in tqdm(r.iter_content(chunk_size=chunk_size)):
            fd.write(chunk)

def extract_and_remove_zip_file(full_path, target_dir):
    with ZipFile(full_path, 'r') as zipObj:
        # Extract all the contents of zip file in current directory
        zipObj.extractall(path=target_dir)
    os.remove(full_path)

def check_dataset_folder(dataset_folder):
    try:
        os.mkdir(dataset_folder)
        print("Directory ", dataset_folder, " Created ")
    except FileExistsError:
        print("Directory ", dataset_folder, " already exists")

def get_dataset_folder():

    return "Datasets"

def download_dataset(Annotation=False, Train=False, Val=False):

    dataset_folder = get_dataset_folder()
    check_dataset_folder(dataset_folder)

    if Annotation:
        ## Download Annotations
        ann_dataset_url_path, ann_dataset_save_path  = get_mscoco_captioning_2017_annotations_path()
        ann_save_path = dataset_folder + "/" + ann_dataset_save_path
        download_url(url=ann_dataset_url_path, save_path=ann_save_path)
        extract_and_remove_zip_file(full_path=ann_save_path, target_dir=dataset_folder)

    if Train:
        ## Download Train Images
        train_dataset_url_path, train_dataset_save_path  = get_mscoco_captioning_train_2017_images_path()
        train_save_path = dataset_folder+ "/" + train_dataset_save_path
        download_url(url=train_dataset_url_path, save_path=train_save_path)
        extract_and_remove_zip_file(full_path=train_save_path, target_dir=dataset_folder)

    if Val:
        ## Download Val Images
        val_dataset_url_path, val_dataset_save_path  = get_mscoco_captioning_val_2017_images_path()
        val_save_path = dataset_folder + "/" + val_dataset_save_path
        download_url(url=val_dataset_url_path, save_path=val_save_path)
        extract_and_remove_zip_file(full_path=val_save_path, target_dir=dataset_folder)
    
def get_mscoco_captioning_train_2017_images_path():
    # returns download url of image captioning 2017 train images
    return "http://images.cocodataset.org/zips/train2017.zip","train2017.zip"
   
def get_mscoco_captioning_val_2017_images_path():
    # returns download url of image captioning 2017 validation images
    return "http://images.cocodataset.org/zips/val2017.zip","val2017.zip"

def get_mscoco_captioning_2017_annotations_path():
    # returns download url of image captioning 2017 annotations
    return "http://images.cocodataset.org/annotations/annotations_trainval2017.zip","annotations2017.zip"

def load_mscoco_annotations_val(root="Resized"):
   
    ann_path = get_val_ann_path()

    with open(ann_path) as f:
        annotations = json.load(f)

    # Store captions and image names in vectors
    all_captions = []
    all_img_names = []
    print("Loading dataset...")
    for annot in tqdm(annotations['annotations']):
        caption = 'soc ' + annot['caption'] + ' eoc'
        image_id = annot['image_id']
        full_coco_image_path = get_val_image_path(root=root) + "/" + '%012d.jpg' % (image_id)

        all_img_names.append(full_coco_image_path)
        all_captions.append(caption)
    return all_captions, all_img_names


def load_mscoco_annotations_train(root="Resized"):
    ann_path = get_train_ann_path()

    with open(ann_path) as f:
        annotations = json.load(f)

    # Store captions and image names in vectors
    all_captions = []
    all_img_names = []
    print("Loading dataset...")
    for annot in tqdm(annotations['annotations']):
        caption = 'soc ' + annot['caption'] + ' eoc'
        image_id = annot['image_id']
        full_coco_image_path = get_train_image_path(root=root) + "/" + '%012d.jpg' % (image_id)

        all_img_names.append(full_coco_image_path)
        all_captions.append(caption)
    return all_captions, all_img_names


def get_val_ann_path():
    return "Annotations" + "/" + "captions_val2017.json" 

def get_val_image_path(root="Resized"):
    return root + "/" + "val2017"

def get_train_ann_path():
    return "Annotations" + "/" + "captions_train2017.json"

def get_train_image_path(root="Resized"):
    return root + "/" + "train2017"

def load_image(path):
    # reads the image with given path
    
    
    return Image.open(path)

def show_image(img):
    # shows the given image
    img.show()

def gray_to_rgb(im):
    return im.convert('RGB')
    
    
   

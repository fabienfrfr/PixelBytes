#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: fabienfrfr
"""

from .dataset import image_pixelization

from datasets import load_dataset, Dataset
import io, os, getpass
from huggingface_hub import HfApi, login
#from transformers import pipeline

from PIL import Image
import cv2, re, unicodedata
import requests, pandas as pd
from bs4 import BeautifulSoup
import numpy as np, pylab as plt


def get_pkmns_miniatures():
    url = "https://www.pokepedia.fr/Liste_des_Pok%C3%A9mon_dans_l%27ordre_du_Pok%C3%A9dex_National"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    response = requests.get(url, headers=headers)
    print(f"Status Code: {response.status_code}")
    soup = BeautifulSoup(response.content, 'html.parser')
    pokemon_miniatures = {}
    # Find line in table
    rows = soup.find_all('tr')
    for row in rows:
        cells = row.find_all('td')
        if len(cells) >= 3:  # Assurez-vous qu'il y a suffisamment de cellules
            number_cell, image_cell, name_cell = cells[0], cells[1], cells[3] # English version
            if number_cell.text.strip().isdigit():
                number = number_cell.text.strip().zfill(4)
                name = name_cell.text.strip()
                img = image_cell.find('img')
                if img and 'src' in img.attrs:
                    image_url = "https://www.pokepedia.fr" + img['src'] if img['src'].startswith('/') else img['src']
                    pokemon_miniatures[number] = {'name': name, 'url': image_url}
    return pokemon_miniatures

## Caption part
def get_pkmn_info(pkmn_name):
    url = f"https://bulbapedia.bulbagarden.net/wiki/{pkmn_name}_(Pokémon)"
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    title = soup.find('h1', {'id': 'firstHeading'}).text.strip() if soup.find('h1', {'id': 'firstHeading'}) else "No head."
    # Search pkmn biology
    biology_section = soup.find('span', {'id': 'Biology'})
    if biology_section:
        first_paragraph = biology_section.find_parent('h2').find_next_sibling('p')
        extract = first_paragraph.text.strip() if first_paragraph else "Aucune information trouvée."
    else:
        content = soup.find('div', {'id': 'mw-content-text'})
        extract = content.find('p').text if content and content.find('p') else "No information.."
    return {"title": title, "extract": extract}

def simplify_character(text):
    return re.sub(r'[^\x00-\x7F]', '', unicodedata.normalize('NFKD', text.lower()).encode('ASCII', 'ignore').decode('ASCII'))

## Image part
def download_pkmn_miniature(url_pkmn):
    response = requests.get(url_pkmn)
    if response.status_code == 200:
        image_array = np.frombuffer(response.content, np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)
        if img.shape[2] == 4:
            rgb_channels = img[:, :, :3]
            alpha_channel = img[:, :, 3] / 255.0
            # Merge image with white background
            white_background = np.ones_like(rgb_channels, dtype=np.uint8) * 255
            img = cv2.convertScaleAbs(rgb_channels * alpha_channel[..., None] + white_background * (1 - alpha_channel[..., None]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return image_pixelization(img, palette, max_size=25)
    else:
        print(f"Download error")
        return None

def create_pkmn_dataset(pkmn_dict, image_dir="data"):
    os.makedirs(image_dir, exist_ok=True)
    # Init pipeline format (if you caption : not used here // old)
    #captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    
    def arraybytes_to_png(arr, number):
        pil_img = Image.fromarray(arr.astype('uint8'))
        img_file_path = f"data/{number}.png" #img_byte_arr = io.BytesIO()
        pil_img.save(img_file_path, format='PNG') #pil_img.save(img_byte_arr, format='PNG')
        return img_file_path # img_byte_arr.getvalue()
    # Generate image captionning
    #names = []
    captions = []
    #processed_images = []
    for number, info in list(pkmn_dict.items())[:]:
        print(number,info)
        try :
            # get img
            img = download_pkmn_miniature(info['url'])
            pil_img = Image.fromarray(img.astype('uint8'))
            # get description [caption = captioner(pil_img)[0]['generated_text']] if not captionned ?
            soup_info = get_pkmn_info(info['name'])
            caption = simplify_character(soup_info["extract"])
            print(info, caption)
            # merge after verif
            #names.append(info['name'])
            captions.append(caption)
            img_file_path = os.path.join(image_dir, f"{number}.png") #img_byte_arr = io.BytesIO()
            pil_img.save(img_file_path, format='PNG') #pil_img.save(img_byte_arr, format='PNG')
            #processed_images.append(img_byte_arr.getvalue())
        except :
            print("Pkmn not found...")
    # Init image dataset
    dataset = load_dataset("imagefolder", data_dir=image_dir)
    # Add caption
    dataset = dataset["train"].add_column("caption", captions)
    #dataset = dataset.add_column("name", names)
    return dataset
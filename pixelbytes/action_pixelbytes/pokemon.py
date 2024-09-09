#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: fabienfrfr
"""

import os, re, shutil, requests
import numpy as np, pandas as pd
from bs4 import BeautifulSoup
from scipy import signal
import soundfile as sf #import librosa
from datasets import load_dataset, Dataset, Features, Value, Image, Audio
from huggingface_hub import login

class PokemonSpriteDownloader:
    BASE_URL = "https://www.pokencyclopedia.info"
    SPRITE_URLS = {'front': f"{BASE_URL}/en/index.php?id=sprites/gen5/ani_black-white",
                   'back': f"{BASE_URL}/en/index.php?id=sprites/gen5/ani-b_black-white" }

    def download_sprites(self, sprite_type='front'):
        # Create a folder for the sprites
        folder_name = f"pokemon_{sprite_type}_gifs"
        os.makedirs(folder_name, exist_ok=True)
        # Fetch the webpage and parse it
        soup = BeautifulSoup(requests.get(self.SPRITE_URLS[sprite_type]).content, 'html.parser')
        pokemon_data = []
        # Find all GIF images
        for gif in soup.find_all('img', src=lambda s: s.endswith('.gif')):
            gif_url = f"{self.BASE_URL}/{gif['src'].lstrip('/')}"
            file_name = gif_url.split('/')[-1]
            pokemon_name = gif.get('alt', 'Unknown').split()[0]
            # Download the GIF
            try:
                response = requests.get(gif_url)
                response.raise_for_status()
                with open(os.path.join(folder_name, file_name), 'wb') as file:
                    file.write(response.content)
                pokemon_data.append({'Pokemon Name': pokemon_name, 'File Name': file_name})
                print(f"Downloaded: {file_name}")
            except requests.RequestException as e:
                print(f"Error downloading {file_name}: {e}")
        # Save data to CSV using Pandas
        pd.DataFrame(pokemon_data).to_csv(f"pokemon_{sprite_type}_data.csv", index=False)
        print(f"Finished! All {sprite_type} GIFs downloaded and data saved.")

    def download_all(self):
        for sprite_type in self.SPRITE_URLS:
            self.download_sprites(sprite_type)

class PokemonCryDownloader:
    BASE_URL = "https://www.pokepedia.fr/Fichier:Cri_{:04d}_HOME.ogg"
    DOMAIN = "https://www.pokepedia.fr"

    def download_cries(self, start=1, end=1010):
        # Create a folder to store the cries
        folder_name = "pokemon_cries"
        os.makedirs(folder_name, exist_ok=True)
        pokemon_data = []
        for pokemon_id in range(start, end + 1):
            url = self.BASE_URL.format(pokemon_id)
            try:
                # Fetch the page and find the audio link
                soup = BeautifulSoup(requests.get(url).content, 'html.parser')
                audio_link = soup.find('a', {'class': 'internal'})['href']
                # Correct the URL if it's relative
                if audio_link.startswith('/'):
                    audio_link = self.DOMAIN + audio_link
                file_name = f"cry_{pokemon_id:04d}.ogg"
                # Download the audio file
                response = requests.get(audio_link)
                response.raise_for_status()  # Check if the request was successful
                with open(os.path.join(folder_name, file_name), 'wb') as file:
                    file.write(response.content)
                pokemon_data.append({'Pokemon ID': pokemon_id, 'File Name': file_name})
                print(f"Downloaded: {file_name}")
            except Exception as e:
                print(f"Error for Pokémon {pokemon_id}: {e}")
        # Save the data to a CSV file
        pd.DataFrame(pokemon_data).to_csv("pokemon_cries_data.csv", index=False)
        print("Finished! All cries have been downloaded and data saved.")

def process_audio(input_file, output_file):
    # Load audio and ensure it's mono
    audio, sr = sf.read(input_file)
    audio = audio.mean(axis=1) if audio.ndim > 1 else audio
    # Normalize original audio
    original = audio / np.max(np.abs(audio))
    # GameBoy parameters
    gb_rate = 8192
    # Resample and filter
    resampled = signal.resample(original, int(len(original) * gb_rate / sr))
    b, a = signal.butter(4, [100 / (0.5 * gb_rate), 3000 / (0.5 * gb_rate)], btype='band')
    filtered = signal.lfilter(b, a, resampled)
    # Resample back to original length
    processed = signal.resample(filtered, len(original))
    # Normalize processed audio
    processed = processed / np.max(np.abs(processed))
    # Combine original and processed signals
    stereo = np.column_stack((original, processed))
    # Save as OGG
    sf.write(output_file, stereo, sr, format='ogg', subtype='vorbis')

def process_all_audio(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for file in sorted(os.listdir(input_dir)):
        if file.endswith(".ogg"):
            number = ''.join(filter(str.isdigit, file))
            process_audio(os.path.join(input_dir, file), os.path.join(output_dir, f"{number}.ogg"))
    print(f"Processed audio files saved in '{output_dir}'")

def process_move_gifs(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    pattern = re.compile(r'ani_bw_(\d+)(?:-[a-z])?.gif')
    for filename in os.listdir(input_dir):
        match = pattern.match(filename)
        if match:
            new_filename = f"{int(match.group(1)):04d}.gif"
            shutil.move(os.path.join(input_dir, filename), os.path.join(output_dir, new_filename))
            print(f"Moved: {filename} -> {new_filename}")

def create_pokemon_csv(dataset_name, output_file='pokemon_data.csv'):
    dataset = load_dataset(dataset_name)
    data = [{'Filename': item['image'].filename, 'Caption': item['caption']}for item in dataset['train']]
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"CSV file '{output_file}' has been created.")

def create_huggingface_dataset(dataset_dir):
    # Get file lists
    images = os.listdir(os.path.join(dataset_dir, "images"))
    numbers = [os.path.splitext(img)[0] for img in images]
    numbers.sort(key=lambda x: int(x))
    # Create data dictionary
    data = {
        "number": numbers,
        "image": [os.path.join(dataset_dir, "images", f"{num}.gif") for num in numbers],
        "audio": [os.path.join(dataset_dir, "audio", f"{num}.ogg") for num in numbers],
        "text": [open(os.path.join(dataset_dir, "text", f"{num}.txt"), 'r').read().strip() for num in numbers]
    }
    # Create and cast dataset
    dataset = Dataset.from_dict(data)
    features = Features({"number": Value("string"), "image": Image(), "audio": Audio(sampling_rate=16000,mono=False), "text": Value("string")})
    return dataset.cast(features)

def push_dataset_to_hub(dataset, repo_name):
    # Ask for the Hugging Face token
    token = input("Please enter your Hugging Face token: ")
    # Login to Hugging Face with the provided token
    login(token=token)
    # Push to Hub
    dataset.push_to_hub(repo_name)

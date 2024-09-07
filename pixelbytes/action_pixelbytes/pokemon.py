#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: fabienfrfr
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import os

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
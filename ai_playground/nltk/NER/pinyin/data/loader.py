import os
import json

script_dir = os.path.dirname(os.path.abspath(__file__))

chars_path = os.path.join(script_dir, 'chars.json')
surnames_path = os.path.join(script_dir, 'surnames.json')

with open(chars_path, 'r', encoding='utf-8') as file:
    chars = json.load(file)

with open(surnames_path, 'r', encoding='utf-8') as file:
    surnames = json.load(file)

words = []
for i in range(10):
    words_path = os.path.join(script_dir, f'words-{i}.json')
    with open(words_path, 'r', encoding='utf-8') as file:
        words.append(json.load(file))


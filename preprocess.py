import re
import time
import string
from tqdm import tqdm

from text import format_input_text
from g2p_en import G2p
import string
file_path = "LJSpeech/"

g2p = G2p()

symbols = ['AA', 'AA0', 'AA1', 'AA2', 'AE', 'AE0', 'AE1', 'AE2', 'AH', 'AH0', 'AH1', 'AH2',
  'AO', 'AO0', 'AO1', 'AO2', 'AW', 'AW0', 'AW1', 'AW2', 'AY', 'AY0', 'AY1', 'AY2',
  'B', 'CH', 'D', 'DH', 'EH', 'EH0', 'EH1', 'EH2', 'ER', 'ER0', 'ER1', 'ER2', 'EY',
  'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', 'IH', 'IH0', 'IH1', 'IH2', 'IY', 'IY0', 'IY1',
  'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OW0', 'OW1', 'OW2', 'OY', 'OY0',
  'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UH0', 'UH1', 'UH2', 'UW',
  'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH']

text_dict = {}
punctuations = "!\"“”‘’#$%&()*+,-./:;<=>?@[\]^_{|}~'"

num_lines = sum(1 for line in open(file_path+"/metadata.csv", 'r'))

with open(file_path+"/metadata.csv") as f:
    for line in tqdm(f, total=num_lines):
        key, _, text = line.strip().split("|")

        text = format_input_text(text)

        with open(file_path+"/wavs/"+key+".lab", 'w', encoding='utf8') as fw:
            fw.write(text)

        text = text.strip().translate(str.maketrans('', '', punctuations)).lower().split(" ")
        phones = []
        for t in text:
            phone = g2p(t)
            phone = ' '.join([p for p in phone if p in symbols])
            phones.append(phone)

        text_dict.update({text[i]: phones[i] for i in range(len(text))})

with open(file_path+"/dict/g2p_dict.txt", 'w', encoding='utf8') as fw:
    for text, phones in sorted(text_dict.items()):
        fw.write(text + " " + phones + "\n")

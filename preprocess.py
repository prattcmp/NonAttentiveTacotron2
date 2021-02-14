import re
import time
import string
import argparse
from multiprocessing import Pool, Manager
from functools import partial
from tqdm import tqdm
from pathlib import Path

from text import format_input_text
from g2p_en import G2p
import string


g2p = G2p()

symbols = ['AA', 'AA0', 'AA1', 'AA2', 'AE', 'AE0', 'AE1', 'AE2', 'AH', 'AH0', 'AH1', 'AH2',
  'AO', 'AO0', 'AO1', 'AO2', 'AW', 'AW0', 'AW1', 'AW2', 'AY', 'AY0', 'AY1', 'AY2',
  'B', 'CH', 'D', 'DH', 'EH', 'EH0', 'EH1', 'EH2', 'ER', 'ER0', 'ER1', 'ER2', 'EY',
  'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', 'IH', 'IH0', 'IH1', 'IH2', 'IY', 'IY0', 'IY1',
  'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OW0', 'OW1', 'OW2', 'OY', 'OY0',
  'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UH0', 'UH1', 'UH2', 'UW',
  'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH']

punctuations = "!\"“”‘’#$%&()*+,-./:;<=>?@[\]^_{|}~'—"
num_workers = 8

str_punc_regex = r"[a-zA-Z0-9][^\w\s]{2,}[a-zA-Z0-9]"

# Fixes punctuation like: This,"dog
def stacked_punctuation(match):
    match = match.group()
    match = [m for m in match] 
    new_match = [match[0]]

    # Start after first punctuation and end before last punctuation
    for i in range(1, len(match) - 2):
        new_match.append(match[i])
        new_match.append(' ')

    new_match.append(match[-2])
    new_match.append(match[-1])

    print(''.join(new_match))
    return ''.join(new_match)

def process_text(text, pathlib_file):
    text = text.replace("'", "")
    text = text.replace("—", "")
    text = re.sub(str_punc_regex, stacked_punctuation, text)

    with pathlib_file.open(mode='w', encoding='utf8') as fw:
        fw.write(text)

    text = text.strip().translate(str.maketrans('', '', punctuations)).upper().split(" ")
    phones, new_text = [], []
    for word in text:
        phone = g2p(word)
        if phone is None or not phone or len(phone) == 0:
            continue
        phone = ' '.join([p for p in phone if p in symbols])
        phones.append(phone)
        new_text.append(word)

    return {new_text[i]: phones[i] for i in range(len(new_text))}

def save_to_dict(text_dict, dict_path):
    with open(dict_path, 'w', encoding='utf8') as fw:
        for text, phones in sorted(text_dict.items()):
            fw.write(text + " " + phones + "\n")


def ljspeech():
    file_path = "LJSpeech"
    text_dict = {}
    num_lines = sum(1 for line in open(file_path+"/metadata.csv", 'r'))

    with open(file_path+"/metadata.csv") as f:
        for line in tqdm(f, total=num_lines):
            key, _, text = line.strip().split("|")

            text = format_input_text(text)

            pathlib_file = Path(file_path + '/wavs/' + key + '.lab')

            text_dict.update(process_text(text, pathlib_file))

        save_to_dict(text_dict, file_path+"/dict/lj_dict.txt")


def vctk_worker(files, text_dict):
    for fi in tqdm(files):
        text = fi.read_text().strip()

        text = format_input_text(text)

        pathlib_file = Path(file_path + '/wavs/' + key.split('_')[0] + '/' + fi.stem + '.lab')

        text_dict.update(process_text(text, pathlib_file))


def vctk():
    file_path = "VCTK"
    manager = Manager()
    text_dict = manager.dict()
    files = list(Path(file_path+"/txt").rglob('*.txt'))

    with Pool(num_workers) as p:
        p.map(vctk_worker, *(files, text_dict))

    save_to_dict(text_dict, file_path+"/dict/vctk_dict.txt")


def libritts_worker(text_dict, fi):
    text = fi.read_text().strip()

    text = format_input_text(text)

#    pathlib_file = fi.rename(str(fi.parent.absolute()) + '/' + fi.stem.split('.')[0] + '.lab')
    pathlib_file = fi.rename(fi.with_suffix('.lab'))

    text_dict.update(process_text(text, pathlib_file))


def libritts():
    file_path = "LibriTTS"
    manager = Manager()
    text_dict = manager.dict()
#    files = list(Path(file_path+"/train-clean-360").rglob('*.normalized.txt'))
    files = list(Path(file_path+"/train-clean-360").rglob('*.lab'))

    func = partial(libritts_worker, text_dict)
    with Pool(num_workers) as p:
        p.map(func, files)

    save_to_dict(text_dict, file_path+"/dict/libritts_dict.txt")

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-lj', '--ljspeech', action='store_true',
                        required=False,
                        help='Load LJSpeech')
    parser.add_argument('-vctk', '--vctk', action='store_true',
                        help='Load VCTK')
    parser.add_argument('-libritts', '--libritts', action='store_true',
                        help='Load LibriTTS')

    args = parser.parse_args()

    if args.ljspeech:
        ljspeech()
    if args.vctk:
        vctk()
    if args.libritts:
        libritts()

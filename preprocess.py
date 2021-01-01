import re
import time
import string
from tqdm import tqdm

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
        # Process text i.e. removes dashes and length abbreviations
        abbrs = [a.span() for a in re.finditer(r"\b[A-Z\.]{2,}s?\b", text)]
        for abbr in abbrs:
            begin, end = abbr
            word = text[begin:end+1]
            if word.count('.') > 1:
                word = word.replace('.', '')
            word = list(word)

            text = text[:begin] + ' '.join(word) + text[end + 1:]

        # Remove random punctuation in between words (i.e. "noon:time" or "free?dom")
        # This significantly improves robustness
        new_text = []
        for i in range(len(text)):
            if i-1 < 0 or i+1 >= len(text) or text[i] == "'":
                new_text.append(text[i])
                continue

            found = False
            for p in punctuations:
                if p == text[i] and text[i-1].isalpha() and text[i+1].isalpha():
                    found = True
                    break
            if not found:
                new_text.append(text[i])
            else:
                new_text.append(' ')

        text = ''.join(new_text)

        # Replace parentheses with commas where possible, otherwise just remove them
        pattern = re.compile(r"(\(|\[)[a-zA-Z0-9\']+(\)|\])")
        parentheses = pattern.search(text)
        while parentheses:
            p1, p2 = parentheses.start(), parentheses.end()
            p2 -= 1 # p2 is the character AFTER the closing parenthesis ")", so we need to subtract 1
            
            if p1-1 >= 0 and text[p1-1] == " ":
                new_text = text[:p1-1] + ', ' + text[p1+1:p2] 
            else:
                new_text = text[:p1] + text[p1+1:p2] 
            if p2+1 < len(text) and text[p2+1] == " ":
                new_text += ',' + text[p2+1:]
            else:
                new_text += text[p2+1:]
            
            text = new_text
            parentheses = pattern.search(text)

        # Remove dashes (-) and hyphens (--) and single quotes (')
        text = ' '.join(text.replace("'", "").replace('-', ' ').split())


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

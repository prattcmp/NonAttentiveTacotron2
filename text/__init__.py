""" from https://github.com/keithito/tacotron """
import re
from text import cleaners
from text.symbols import symbols
from textgrid import TextGrid

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')

dash_pattern = re.compile(r"(?<=[.,!?] )-- ")
parentheses_pattern = re.compile(r"(?<=[.,!?] )[\(\[]|[\)\]](?=[.,!?])|^[\(\[]|[\)\]]$")

punctuations = "!\"“”‘’#$%&()*+,-./:;<=>?@[\]^_{|}~'" 

def replace_symbols(text):
    # replace semi-colons and colons with commas
    text = text.replace(";", ",")
    text = text.replace(":", ",")

    # replace dashes with commas
    text = dash_pattern.sub("", text)
    text = text.replace(" --", ",")
    text = text.replace(" - ", ", ")

    # split hyphenated words
    text = text.replace("-", " ")

    # replace parentheses with commas
    text = parentheses_pattern.sub("", text)
    text = text.replace(")", ",")
    text = text.replace(" (", ", ")
    text = text.replace("]", ",")
    text = text.replace(" [", ", ")
    
    return text

def format_input_text(text):
    # Process text i.e. removes dashes and length abbreviations
    #abbrs = [a.span() for a in re.finditer(r"\b[A-Z\.]{2,}s?\b", text)]
    abbr = re.search(r"\b[A-Z\.]{2,}s?\b", text)
    while abbr is not None:
        begin, end = abbr.span()
        word = text[begin:end+1]
        if word.count('.') > 1 or word[-1] != '.':
            word = word.replace('.', '')
        word = list(word)

        text = text[:begin] + ' '.join(word)[:-1] + text[end + 1:]

        abbr = re.search(r"\b[A-Z\.]{2,}s?\b", text)

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

    # Replace single-word parentheses with commas where possible, otherwise just remove them
    # "The duck (Ducky) jumped." => "The duck, Ducky, jumped."
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

    text = replace_symbols(text)

    return text

def text_to_sequence(text, g2p, use_textgrid=False):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

    The text is converted to an ARPAbet sequence. For example, "HH AW1 S S T AH0 N."

    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through

    Returns:
      List of integers corresponding to the symbols in the text
  '''

  if not use_textgrid:
    text = re.sub(re.compile(r'\s+'), ' ', text)
    text = format_input_text(text)
    print(text)
    text = g2p(text)

  arpabet = _arpabet_to_sequence(text)

  end_token = len(symbols) + 1

  if arpabet[-1] != end_token:
    arpabet.append(end_token)
  
  return arpabet


def sequence_to_text(sequence):
  '''Converts a sequence of IDs back to a string'''
  result = ''
  for symbol_id in sequence:
    if symbol_id in _id_to_symbol:
      s = _id_to_symbol[symbol_id]
      # Enclose ARPAbet back in curly braces:
      if len(s) > 1 and s[0] == '@':
        s = '{%s}' % s[1:]
      result += s
  return result.replace('}{', ' ')


def _clean_text(text, cleaner_names):
  for name in cleaner_names:
    cleaner = getattr(cleaners, name)
    if not cleaner:
      raise Exception('Unknown cleaner: %s' % name)
    text = cleaner(text)
  return text


def _symbols_to_sequence(symbols):
  for s in symbols:
      if not _should_keep_symbol(s):
          print(s)

  symbols = [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]

  return symbols


def _arpabet_to_sequence(text):
  return _symbols_to_sequence([s for s in text])


def _should_keep_symbol(s):
  return s in _symbol_to_id and s is not '_' and s is not '~'

import random
import string
import glob
import numpy as np
from pathlib import Path
import torch
import torch.utils.data
import time

import layers
from utils import load_wav_to_torch, load_filepaths_and_text
from text import text_to_sequence
from g2p_en import G2p
from textgrid import TextGrid

def find_punctuation(audio_name, word_intervals_dict, phoneme_intervals_dict):
    punctuations = "!\"#$%&()*+,-./:;<=>?@[\]^_{|}~"

    with open("LJSpeech/wavs/" + audio_name + ".lab") as f:
        text = f.readline().strip()
        punc_positions = {pos: char for pos, char in enumerate(text) if char in punctuations}

        found_phonemes = {}
        # Find nearest word to punctuation
        for pos, char in punc_positions.items():
            letter = "1"
            cur_pos = pos
            not_found = False

            while not letter.isalpha():
                cur_pos -= 1
                if cur_pos >= 0:
                    letter = text[cur_pos]
                else:
                    not_found = True
                    break

            if not_found:
                continue
            
            end = cur_pos
            while letter != " ":
                cur_pos -= 1

                if cur_pos >= 0:
                    letter = text[cur_pos]
                else:
                    cur_pos = -1
                    break
            begin = cur_pos + 1

            # Found the word that comes before punctuation, now find its phoneme
            new_text = text[begin:end+1]
            word = new_text.translate(str.maketrans('', '', punctuations)).lower()

#            if word == "anoura":
#                print(word, text, word_intervals_dict)

            word_maxTime = word_intervals_dict[word]
            phoneme_idx = phoneme_intervals_dict[word_maxTime]
            found_phonemes[phoneme_idx] = char

        return found_phonemes

class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, audiopaths_and_text, hparams, use_textgrid=False, valset=False):
        self.g2p = G2p()
        self.use_textgrid = use_textgrid
        if use_textgrid:
            # TODO: HARDCODED PATH! SHOULD CHANGE THIS AT SOME POINT
            self.textgrid_paths = glob.glob("LJSpeech/durations/*.TextGrid")
        else:
            self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        random.seed(hparams.seed)
        random.shuffle(self.textgrid_paths if use_textgrid else self.audiopaths_and_text)

        if valset == True:
            self.textgrid_paths = self.textgrid_paths[:101]
        else:
            self.textgrid_paths = self.textgrid_paths[101:]

        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)


    def get_from_textgrid(self, path):
        # Get audiopaths, durations, and phonemes
        audio_name = Path(path).stem
        # TODO: HARDCODED PATH! SHOULD CHANGE THIS AT SOME POINT
        mel = self.get_mel("LJSpeech/wavs/" + audio_name + ".wav")

        # [words, phones] - select [1] for phonemes
        intervals = TextGrid.fromFile(path)
        word_intervals = intervals[0]
        phoneme_intervals = intervals[1]

        word_intervals_dict = {w.mark: str(w.maxTime) for w in word_intervals}
        phoneme_intervals_dict = {str(phoneme_intervals[i].maxTime): i for i in range(len(phoneme_intervals))}

        # Returns: Python Dictionary(key: phoneme_idx, value: punctuation_char)
        punctuations = find_punctuation(audio_name, word_intervals_dict, phoneme_intervals_dict)

        phonemes = []
        durations = []
        for i in range(len(phoneme_intervals)):
            interval = phoneme_intervals[i]
            phoneme = interval.mark 
            # Append the punctuation directly to the phoneme
            if i in punctuations:
                phoneme += punctuations[i]

            duration = interval.maxTime - interval.minTime

            # ITS BEAUTIFUL! WE HAVE EQUAL LENGTH TARGET DURATIONS AND INPUT PHONEMES
            # INPUT PHONEMES HAVE PUNCTUATION ENCODINGS!!!
            phonemes.append(phoneme)
            durations.append(duration)

        text = self.get_text(phonemes)
        duration = self.get_duration(durations)

        assert len(text) == len(duration), "duration length and phoneme token length are unequal. this breaks the entire model"

        return (text, mel, duration)
                

    def get_mel_text_pair(self, audiopath_and_text):
        # separate filename and text
        audiopath, text = audiopath_and_text[0], audiopath_and_text[1]

        text = self.get_text(text)
        mel = self.get_mel(audiopath)
        return (text, mel)

    def get_mel(self, filename):
        if not self.load_mel_from_disk:
            audio, sampling_rate = load_wav_to_torch(filename)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} {} SR doesn't match target {} SR".format(
                    sampling_rate, self.stft.sampling_rate))
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
        else:
            melspec = torch.from_numpy(np.load(filename))
            assert melspec.size(0) == self.stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    melspec.size(0), self.stft.n_mel_channels))

        return melspec

    def get_text(self, text):
        return torch.IntTensor(text_to_sequence(text, self.g2p, self.use_textgrid))

    def get_duration(self, duration):
        return torch.FloatTensor(duration)

    def __getitem__(self, index):
        if self.use_textgrid:
            return self.get_from_textgrid(self.textgrid_paths[index])

        return self.get_mel_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        if self.use_textgrid:
            return len(self.textgrid_paths)

        return len(self.audiopaths_and_text)


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        duration_padded = torch.FloatTensor(len(batch), max_input_len)
        duration_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            duration = batch[ids_sorted_decreasing[i]][2]
            duration_padded[i, :duration.size(0)] = duration

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            output_lengths[i] = mel.size(1)

        return text_padded, input_lengths, mel_padded, duration_padded, \
            output_lengths

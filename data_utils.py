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
from speakerembedding.Pattern_Generator import Pattern_Generate_Inference
from speakerembedding.Datasets import Correction


def find_punctuation(filename, word_intervals_dict, phoneme_intervals_dict):
    punctuations = "!\"#$%&()*+,-./:;<=>?@[\]^_{|}~'—"

    lab_path = filename.with_suffix('.lab')

    with lab_path.open() as f:
        text = f.readline().strip()
        punc_positions = {pos: char for pos, char in enumerate(text) if char in punctuations}

        found_phonemes = {}
        # Find nearest word to punctuation
        for pos, char in punc_positions.items():
            letter = "1"
            cur_pos = pos
            not_found = False

            while not letter.isalpha() and not letter == "'":
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

#            if word == "anger":
#                print(new_text, word, text, word_intervals_dict)
#                print(lab_path.stem)

            try:
                word_maxTime = word_intervals_dict[word]
            except:
                print(new_text, word, text, word_intervals_dict)
                print(lab_path.stem)
                continue
                
            phoneme_idx = phoneme_intervals_dict[word_maxTime]
            found_phonemes[phoneme_idx] = char

        return found_phonemes

class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, audiopaths_and_text, hparams, use_textgrid=False, dataset_type='train'):
        self.g2p = G2p()
        self.use_textgrid = use_textgrid
        if use_textgrid:
            # TODO: HARDCODED PATH! SHOULD CHANGE THIS AT SOME POINT
            self.textgrid_paths = list(Path('LibriTTS/durations').rglob("*.TextGrid"))
        else:
            self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)

        if dataset_type == 'val' or dataset_type == 'train':
            random.seed(hparams.seed)
            random.shuffle(self.textgrid_paths if use_textgrid else self.audiopaths_and_text)

        if dataset_type == 'val':
            self.textgrid_paths = self.textgrid_paths[:101]
        elif dataset_type == 'train':
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
        folder = audio_name.split('_')[0]
        wav_file = Path("LibriTTS/train-clean-360/" + folder + '/' + audio_name + ".wav")
        # TODO: HARDCODED PATH! SHOULD CHANGE THIS AT SOME POINT
        mel = self.get_mel(wav_file)
        # Text string for torchmoji embeddings
        tokenized_text = self.get_tokenized_text(wav_file)
        # Mel for speaker embedding code
        mel2 = Pattern_Generate_Inference(wav_file, top_db= 20)

        # [words, phones] - select [1] for phonemes
        intervals = TextGrid.fromFile(path)
        word_intervals = intervals[0]
        phoneme_intervals = intervals[1]

        word_intervals_dict = {w.mark: str(w.maxTime) for w in word_intervals}
        phoneme_intervals_dict = {str(phoneme_intervals[i].maxTime): i for i in range(len(phoneme_intervals))}

        # Returns: Python Dictionary(key: phoneme_idx, value: punctuation_char)
        punctuations = find_punctuation(wav_file, word_intervals_dict, phoneme_intervals_dict)
        word_boundaries_dict = {w.maxTime: 1 for w in word_intervals}
        phonemes = []
        durations = []
        captured_space = False
        for i in range(len(phoneme_intervals)):
            interval = phoneme_intervals[i]
            phoneme = interval.mark 

            next_interval, next_phoneme, next_duration = None, None, None
            if i < len(phoneme_intervals) - 1:
                next_interval = phoneme_intervals[i+1]
                next_phoneme = next_interval.mark

            if captured_space and phoneme == "sp":
                continue
            if phoneme == "sp":
                print("Prior:", phoneme_intervals[i+1].mark)

            captured_space = False

            # Try to replace the sp token with punctuation; otherwise, give it a token with dur 0
            next_duration = None
            if i in punctuations:
                if next_phoneme == "sp":
                    captured_space = True
                    next_duration = next_interval.maxTime - next_interval.minTime
                else: next_duration = 0.0 
                next_phoneme = punctuations[i]

            duration = interval.maxTime - interval.minTime

            # ITS BEAUTIFUL! WE HAVE EQUAL LENGTH TARGET DURATIONS AND INPUT PHONEMES
            # INPUT PHONEMES HAVE PUNCTUATION ENCODINGS!!!
            phonemes.append(phoneme)
            durations.append(duration)
            if next_duration is not None:
                phonemes.append(next_phoneme)
                durations.append(next_duration)

            # Add word boundary token AFTER we've modified our other tokens
            if interval.maxTime in word_boundaries_dict:
                word_boundary_token = "sil"
                phonemes.append(word_boundary_token)
                if not captured_space and next_phoneme == "sp":
                    captured_space = True
                    durations.append(next_interval.maxTime - next_interval.minTime)
                else:
                    durations.append(0.0)

        text = self.get_arpabet(phonemes)
        duration = self.get_duration(durations)

        # Assume the stop/end token is missing
        if (len(text) - 1) == len(duration):
            duration = torch.cat((duration, torch.zeros(1, dtype=torch.float)), 0)
            
        assert len(text) == len(duration), (str(len(text)) + " " + str(len(duration)) + " " + str(text) + " " + str(duration) + " " + str(phonemes) + " duration length and phoneme token length are unequal\n\nFILE: " + str(wav_file))

        if text is None:
            return None

        return (text, mel, duration, audio_name, tokenized_text, mel2)
                

    def get_mel_text_pair(self, audiopath_and_text):
        # separate filename and text
        audiopath, text = audiopath_and_text[0], audiopath_and_text[1]

        text = self.get_text(text)
        mel = self.get_mel(audiopath)
        return (text, mel)

    def get_mel(self, filename):
        npy_path = filename.with_suffix('.npy')
        if not npy_path.is_file():
            audio, sampling_rate = load_wav_to_torch(filename)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} {} SR doesn't match target {} SR".format(
                    sampling_rate, self.stft.sampling_rate))
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)

            np.save(npy_path, melspec.cpu().detach().numpy())
        else:
            melspec = torch.from_numpy(np.load(npy_path))
            assert melspec.size(0) == self.stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    melspec.size(0), self.stft.n_mel_channels))

        return melspec

    def get_arpabet(self, text):
        if text is None or text == '':
            return None

        return torch.IntTensor(text_to_sequence(text, self.g2p, self.use_textgrid))

    def get_tokenized_text(self, path):
        if path is None or path == '':
            return None

        text = path.with_suffix('.lab').read_text()

        return text

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
        '''
        self.samples = 5
        self.frame_length = 96
        self.overlap_length = 48
        self.required_length = self.samples * (self.frame_length - self.overlap_length) + self.overlap_length
        '''
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
        audio_names = [''] * len(ids_sorted_decreasing)
        text_strings = [''] * len(ids_sorted_decreasing)
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            output_lengths[i] = mel.size(1)

            audio_names[i] = batch[ids_sorted_decreasing[i]][3]
            text_strings[i] = batch[ids_sorted_decreasing[i]][-2]

        mels2 = []
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][-1]
            '''
            mel = Correction(patterns, self.required_length)
            mel = np.stack([
                mel[index:index + self.frame_length]
                for index in range(0, self.required_length - self.overlap_length, self.frame_length - self.overlap_length)
                ])
            '''
            mels2.append(mel)

        mels2 = torch.FloatTensor(np.vstack(mels2)).transpose(2, 1)   # [Speakers * Samples, Mel_dim, Time]


        return text_padded, input_lengths, mel_padded, duration_padded, \
            output_lengths, audio_names, text_strings, mels2

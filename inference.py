import matplotlib
import matplotlib.pylab as plt

from g2p_en import G2p
 
import tkinter
import argparse
import sys
import numpy as np
import torch

from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence

from hifigan.inference_e2e import inference

def plot_data(data, figsize=(16, 4)):
    matplotlib.use('TkAgg')
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='lower', 
                       interpolation='none')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--text', type=str, required=True,
                        help='what the model will say')
    parser.add_argument('-c', '--checkpoint_path', required=True, type=str,
                        help='directory to Tacotron checkpoints used for inference')
    parser.add_argument('-o', '--output_file', default='generated_speech', required=False, type=str,
                        help='filename of output .wav file')
    parser.add_argument('--profiling', action='store_true',
                        required=False, help='enables the profiler')

    args = parser.parse_args()

    hparams = create_hparams()

    checkpoint_path = args.checkpoint_path
    model = load_model(hparams)
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    _ = model.cuda().eval()

    text = args.text
    g2p = G2p()
    sequence = np.array(text_to_sequence(text, g2p))[None, :]
    sequence = torch.autograd.Variable(
        torch.from_numpy(sequence)).cuda().long()

    mel_outputs, mel_outputs_postnet, alignments = model.inference(sequence)

    # Synthesize audio with Hifi-GAN
    checkpoint_file = "hifigan/generator_v1"
    mel = mel_outputs_postnet.float().data
    inference(args.output_file, checkpoint_file, input_mel=mel_outputs_postnet.float().data)

    plot_data((mel_outputs.float().data.cpu().numpy()[0],
            mel_outputs_postnet.float().data.cpu().numpy()[0],
            alignments.float().data.cpu().numpy()[0].T))


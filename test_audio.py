import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from scipy.io import wavfile
from torch.autograd import Variable
from tqdm import tqdm
import librosa

from model import Generator
# from data_preprocess import slice_signal, window_size, sample_rate
# from utils import emphasis


def emphasis(signal_batch, emph_coeff=0.95, pre=True):
    """
    Pre-emphasis or De-emphasis of higher frequencies given a batch of signal. 对一批信号进行高频预加重或去加重

    Args:
        signal_batch: batch of signals, represented as numpy arrays
        emph_coeff: emphasis coefficient 强调系数
        pre: pre-emphasis or de-emphasis signals

    Returns:
        result: pre-emphasized or de-emphasized signal batch 预强调或去强调信号批
    """
    result = np.zeros(signal_batch.shape)
    for sample_idx, sample in enumerate(signal_batch):
        for ch, channel_data in enumerate(sample):
            if pre:
                result[sample_idx][ch] = np.append(channel_data[0], channel_data[1:] - emph_coeff * channel_data[:-1])  # 几个通道？？？？
            else:
                result[sample_idx][ch] = np.append(channel_data[0], channel_data[1:] + emph_coeff * channel_data[:-1])
    return result

def slice_signal(file, window_size, stride, sample_rate):
    """
    Helper function for slicing the audio file
    by window size and sample rate with [1-stride] percent overlap (default 50%).
    """
    wav, sr = librosa.load(file, sr=sample_rate)
    hop = int(window_size * stride)
    slices = []
    for end_idx in range(window_size, len(wav), hop):
        start_idx = end_idx - window_size
        slice_sig = wav[start_idx:end_idx]
        #print(type(slice_sig),' ',slice_sig.shape,'begin:',start_idx,'end_idx:',end_idx)
        slices.append(slice_sig)

    if(len(slices)*window_size<len(wav)):
        slice_sig = np.zeros((window_size,))
        temp = wav[len(slices)*window_size:]
        slice_sig[:len(temp)] = temp
        slices.append(slice_sig)
        #print(type(slice_sig), ' ', slice_sig.shape,'begin:',0,'end_idx:',len(temp))

    return slices

window_size = 2 ** 14  # about 1 second of samples 16384
sample_rate = 16000

def main(path = ''):
    parser = argparse.ArgumentParser(description='Test Single Audio Enhancement')
    parser.add_argument('--test_folder', default='../test-mix-2-babble', type=str, help='audio file name')
    parser.add_argument('--epoch_name', default='/media/sundekai/DATA/1-sundekai/segan-base_1221_正常计算_10/epochs/generator-53.pkl', type=str, help='generator epoch name')
    parser.add_argument('--enhanced_save', action='store_true', default=True, help='is or not save enhanced_speech')

    opt = parser.parse_args()
    TEST_FOLDER = opt.test_folder
    EPOCH_NAME = opt.epoch_name

    if path != '':
        EPOCH_NAME = path

    generator = Generator()
    model_parameter = torch.load(EPOCH_NAME, map_location='cpu')
    generator.load_state_dict(model_parameter)
    if torch.cuda.is_available():
        generator.cuda()

    for audio in os.listdir(TEST_FOLDER):
        # print('doing',audio,'...')
        audio = os.path.join(TEST_FOLDER, audio)
        noisy_slices = slice_signal(audio, window_size, 1, sample_rate)
        enhanced_speech = []
        for noisy_slice in tqdm(noisy_slices, desc='Generate enhanced audio'):
            z = nn.init.normal(torch.Tensor(1, 1024, 8))
            noisy_slice = torch.from_numpy(emphasis(noisy_slice[np.newaxis, np.newaxis, :])).type(torch.FloatTensor)
            if torch.cuda.is_available():
                noisy_slice, z = noisy_slice.cuda(), z.cuda()
            noisy_slice, z = Variable(noisy_slice), Variable(z)
            generated_speech = generator(noisy_slice, z).data.cpu().numpy()
            generated_speech = emphasis(generated_speech, emph_coeff=0.95, pre=False)
            generated_speech = generated_speech.reshape(-1)
            enhanced_speech.append(generated_speech)

        if (opt.enhanced_save):
            save_path = '../test-result'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            enhanced_speech = np.array(enhanced_speech).reshape(1, -1)
            file_name = os.path.join(save_path,
                                     'enhanced_{}.wav'.format(os.path.basename(audio).split('.')[0]))
            wavfile.write(file_name, sample_rate, enhanced_speech.T)

if __name__ == '__main__':
    main()








import argparse
import os

import librosa
import pypesq as pesq
from pystoi import stoi
import numpy as np



def SegSNR(ref_wav, in_wav, windowsize, shift):
    if len(ref_wav) == len(in_wav):
        pass
    else:
        print('音频的长度不相等!')
        minlenth = min(len(ref_wav), len(in_wav))
        ref_wav = ref_wav[: minlenth]
        in_wav = in_wav[: minlenth]
    # 每帧语音中有重叠部分，除了重叠部分都是帧移，overlap=windowsize-shift
    # num_frame = (len(ref_wav)-overlap) // shift
    #           = (len(ref_wav)-windowsize+shift) // shift
    num_frame = (len(ref_wav) - windowsize + shift) // shift  # 计算帧的数量

    SegSNR = np.zeros(int(num_frame))
    # 计算每一帧的信噪比
    for i in range(num_frame):
        noise_frame_energy = np.sum(ref_wav[i * shift: i * shift + windowsize] ** 2)  # 每一帧噪声的功率
        speech_frame_energy = np.sum(in_wav[i * shift: i * shift + windowsize] ** 2)  # 每一帧信号的功率
        print(noise_frame_energy,"-----",speech_frame_energy)
        SegSNR[i] = np.log10(speech_frame_energy / noise_frame_energy)

    return 10 * np.mean(SegSNR)

def numpy_SNR(labels, logits):
    # origianl_waveform和target_waveform都是一维数组 (seq_len, )
    # np.sum实际功率;np.mean平均功率，二者结果一样
    signal = np.sum(labels ** 2)
    noise = np.sum((labels - logits) ** 2)
    snr = 10 * np.log10(signal / noise)
    return snr

def main():
    parser = argparse.ArgumentParser(description='Calculate performance index')
    parser.add_argument('--test_mix_folder', default='../test-mix-2-babble', type=str, help='test-set-mix')
    parser.add_argument('--test_clean_folder', default='../test-clean-2-babble', type=str,
                        help='test-set-clean')
    parser.add_argument('--enhanced_folder', default='../test-result', type=str, help='test-set-enhanced')

    opt = parser.parse_args()
    MIX_FOLDER = opt.test_mix_folder
    CLEAN_FOLDER = opt.test_clean_folder
    ENHANCED_FOLDER = opt.enhanced_folder

    pesqs = []
    stois = []

    for cleanfile in os.listdir(CLEAN_FOLDER):
        mixfile = cleanfile.replace('clean', 'mix')
        enhancedfile = 'enhanced_' + mixfile

        cleanfile = os.path.join(CLEAN_FOLDER, cleanfile)
        mixfile = os.path.join(MIX_FOLDER, mixfile)
        enhancedfile = os.path.join(ENHANCED_FOLDER, enhancedfile)

        ref, sr1 = librosa.load(cleanfile, 16000)
        #deg_mix, sr2 = librosa.load(mixfile, 16000)
        deg_enh, sr3 = librosa.load(enhancedfile, 16000)

        #pesq1 = pesq.pesq(ref, deg_mix)
        pesq2 = pesq.pesq(ref, deg_enh[:len(ref)])
        #print("pesq:", pesq1, " --> ", pesq2)

        pesqs.append(pesq2);

        #stoi1 = stoi(ref, deg_mix, fs_sig=16000)
        stoi2 = stoi(ref, deg_enh[:len(ref)], fs_sig=16000)
        #print("stoi:", stoi1, " --> ", stoi2)
        stois.append(stoi2)

    print('Epesq:', np.mean(pesqs),"Estoi:", np.mean(stois))


if __name__ == '__main__':
    main()
import argparse
import os

import torch
import torch.nn as nn
from scipy.io import wavfile
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_preprocess import sample_rate
from model import Generator, Discriminator
from utils import AudioDataset, emphasis

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Audio Enhancement')
    parser.add_argument('--batch_size', default=10, type=int, help='train batch size')
    parser.add_argument('--num_epochs', default=86, type=int, help='train epochs number')

    opt = parser.parse_args()
    BATCH_SIZE = opt.batch_size
    NUM_EPOCHS = opt.num_epochs

    # load data
    print('loading data...')
    train_dataset = AudioDataset(data_type='train')
    #test_dataset = AudioDataset(data_type='test')#获取路径名文件夹内每个音频文件
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    #test_data_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)#加载文件

    '''
    # generate reference batch
    ref_batch = train_dataset.reference_batch(BATCH_SIZE) #获得随机选取的参考 批次大小的 张量
    '''

    # create D and G instances
    discriminator = Discriminator()
    generator = Generator()#模型初始化
    if torch.cuda.is_available(): #是否有GPU
        discriminator.cuda() #.cuda()转GPU
        generator.cuda()
        # ref_batch = ref_batch.cuda()
    # ref_batch = Variable(ref_batch)
    # Variable就是 变量 的意思。实质上也就是可以变化的量，区别于int变量，它是一种可以变化的变量，这正好就符合了反向传播，参数更新的属性。
#具体来说，在pytorch中的Variable就是一个存放会变化值的地理位置，里面的值会不停发生片花，就像一个装鸡蛋的篮子，鸡蛋数会不断发生变化。那谁是里面的鸡蛋呢，自然就是pytorch中的tensor了

    print("# generator parameters:", sum(param.numel() for param in generator.parameters())) #通过Module.parameters()获取网络的参数
    print("# discriminator parameters:", sum(param.numel() for param in discriminator.parameters()))#numel()函数：返回数组中元素的个数
    # optimizers
    g_optimizer = optim.RMSprop(generator.parameters(), lr=0.0001) #使用RMSporp的反向传播的优化算法（pytorch的四种方法之一）
    d_optimizer = optim.RMSprop(discriminator.parameters(), lr=0.0001)#学习率0.0001

    for epoch in range(NUM_EPOCHS):
        train_bar = tqdm(train_data_loader)
        for train_batch, train_clean, train_noisy in train_bar:

            # latent vector - normal distribution
            z = nn.init.normal(torch.Tensor(train_batch.size(0), 1024, 8)) #初始化噪声z
            if torch.cuda.is_available():
                train_batch, train_clean, train_noisy = train_batch.cuda(), train_clean.cuda(), train_noisy.cuda()
                z = z.cuda()#转GPU
            train_batch, train_clean, train_noisy = Variable(train_batch), Variable(train_clean), Variable(train_noisy)
            z = Variable(z)#tensor转variable

            # TRAIN D to recognize clean audio as clean
            # training batch pass
            discriminator.zero_grad()    #optimizer.zero_grad()意思是把梯度置零，也就是把loss关于weight的导数变成0.
            outputs = discriminator(train_batch)
            clean_loss = torch.mean((outputs - 1.0) ** 2)  # L2 loss - we want them all to be 1 求均值二范式
            #clean_loss.backward()

            # TRAIN D to recognize generated audio as noisy
            generated_outputs = generator(train_noisy, z)
            outputs = discriminator(torch.cat((generated_outputs, train_noisy), dim=1))#cat((a,b),1) a,b按列拼接
            noisy_loss = torch.mean(outputs ** 2)  # L2 loss - we want them all to be 0 求均值二范式
            #noisy_loss.backward()

            d_loss = clean_loss + noisy_loss
            d_loss.backward()
            d_optimizer.step()  # update parameters   反向传播更新

            # TRAIN G so that D recognizes G(z) as real
            generator.zero_grad()
            generated_outputs = generator(train_noisy, z)
            gen_noise_pair = torch.cat((generated_outputs, train_noisy), dim=1)
            outputs = discriminator(gen_noise_pair)

            g_loss_ = 0.5 * torch.mean((outputs - 1.0) ** 2)
            # L1 loss between generated output and clean sample
            l1_dist = torch.abs(torch.add(generated_outputs, torch.neg(train_clean)))
            g_cond_loss = 100 * torch.mean(l1_dist)  # conditional loss
            g_loss = g_loss_ + g_cond_loss#(损失加上正则项么 g_cond_loss )

            # backprop + optimize
            g_loss.backward() #计算梯度
            g_optimizer.step()  #通过梯度下降法更新权重                    #反向传播

            train_bar.set_description( #训练一个bach描述一下
                'Epoch {}: d_clean_loss {:.4f}, d_noisy_loss {:.4f}, g_loss {:.4f}, g_conditional_loss {:.4f}'
                    .format(epoch + 1, clean_loss.item(), noisy_loss.item(), g_loss.item(), g_cond_loss.item()))

        # TEST model  #训练一个epoch后test (每训练一次都要test一遍么还要存下来)
        # test_bar = tqdm(test_data_loader, desc='Test model and save generated audios')
        # for test_file_names, test_noisy in test_bar:
        #     z = nn.init.normal(torch.Tensor(test_noisy.size(0), 1024, 8)) #初始化噪声z
        #     if torch.cuda.is_available():
        #         test_noisy, z = test_noisy.cuda(), z.cuda() #转cuda
        #     test_noisy, z = Variable(test_noisy), Variable(z) #放variable
        #     fake_speech = generator(test_noisy, z).data.cpu().numpy()  # convert to numpy array
        #     fake_speech = emphasis(fake_speech, emph_coeff=0.95, pre=False) #高频预加重
        #
        #     for idx in range(fake_speech.shape[0]):
        #         generated_sample = fake_speech[idx]
        #         file_name = os.path.join('results',
        #                                  '{}_e{}.wav'.format(test_file_names[idx].replace('.npy', ''), epoch + 1))
        #         wavfile.write(file_name, sample_rate, generated_sample.T)

        # save the model parameters for each epoch
        g_path = os.path.join('epochs', 'generator-{}.pkl'.format(epoch + 1))
        #d_path = os.path.join('epochs', 'discriminator-{}.pkl'.format(epoch + 1))
        torch.save(generator.state_dict(), g_path)
        #torch.save(discriminator.state_dict(), d_path)

        #pytorch 中的 state_dict 是一个简单的python的字典对象,将每一层与它的对应参数建立映射关系.(如model的每一层的weights及偏置等等)
        #(注意,只有那些参数可以训练的layer才会被保存到模型的state_dict中,如卷积层,线性层等等)
        #优化器对象Optimizer也有一个state_dict,它包含了优化器的状态以及被使用的超参数(如lr, momentum,weight_decay等)
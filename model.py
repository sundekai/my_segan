import torch
import torch.nn as nn
from torch.nn.modules import Module
from torch.nn.parameter import Parameter


class Generator(nn.Module):
    """G"""

    def __init__(self):
        super().__init__()
        # encoder gets a noisy signal as input [B x 1 x 16384]
        '''
        class torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        in_channels(int) – 输入信号的通道。
        out_channels(int) – 卷积产生的通道。有多少个out_channels，就需要多少个1维卷积
        kernel_size(int or tuple) - 卷积核的尺寸，卷积核的大小为(k,)，第二个维度是由in_channels来决定的，所以实际上卷积大小为kernel_size*in_channels
        stride(int or tuple, optional) - 卷积步长
        padding (int or tuple, optional)- 输入的每一条边补充0的层数
        dilation(int or tuple, `optional``) – 卷积核元素之间的间距
        groups(int, optional) – 从输入通道到输出通道的阻塞连接数
        bias(bool, optional) - 如果bias=True，添加偏置   
        
        
        输入: (N,C_in,L_in)
        输出: (N,C_out,L_out)
        输入输出的计算方式：
        $$L_{out}=floor((L_{in}+2padding-dilation(kernerl_size-1)-1)/stride+1)$$
        
        '''

        self.enc1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=32, stride=2, padding=15)  # [B x 16 x 8192] 1->16
        self.enc1_nl = nn.PReLU()

        #   PReLU(x)=max(0,x)+a∗min(0,x)  Parametric ReLU    torch.nn.PReLU(num_parameters=1, init（a）=0.25)
        '''
        torch.nn.PReLU(num_parameters=1, init=0.25)：$PReLU(x) = max(0,x) + a * min(0,x)

        a是一个可学习参数。当没有声明时，nn.PReLU()在所有的输入中只有一个参数a；如果是nn.PReLU(nChannels)，a将应用到每个输入。
        
        注意：当为了表现更佳的模型而学习参数a时不要使用权重衰减（weight decay）
        
        参数：

        num_parameters：需要学习的a的个数，默认等于1
        init：a的初始值，默认等于0.25
        '''
        self.enc2 = nn.Conv1d(16, 32, 32, 2, 15)  # [B x 32 x 4096]
        self.enc2_nl = nn.PReLU()
        self.enc3 = nn.Conv1d(32, 32, 32, 2, 15)  # [B x 32 x 2048]
        self.enc3_nl = nn.PReLU()
        self.enc4 = nn.Conv1d(32, 64, 32, 2, 15)  # [B x 64 x 1024]
        self.enc4_nl = nn.PReLU()
        self.enc5 = nn.Conv1d(64, 64, 32, 2, 15)  # [B x 64 x 512]
        self.enc5_nl = nn.PReLU()
        self.enc6 = nn.Conv1d(64, 128, 32, 2, 15)  # [B x 128 x 256]
        self.enc6_nl = nn.PReLU()
        self.enc7 = nn.Conv1d(128, 128, 32, 2, 15)  # [B x 128 x 128]
        self.enc7_nl = nn.PReLU()
        self.enc8 = nn.Conv1d(128, 256, 32, 2, 15)  # [B x 256 x 64]
        self.enc8_nl = nn.PReLU()
        self.enc9 = nn.Conv1d(256, 256, 32, 2, 15)  # [B x 256 x 32]
        self.enc9_nl = nn.PReLU()
        self.enc10 = nn.Conv1d(256, 512, 32, 2, 15)  # [B x 512 x 16]
        self.enc10_nl = nn.PReLU()
        self.enc11 = nn.Conv1d(512, 1024, 32, 2, 15)  # [B x 1024 x 8]
        self.enc11_nl = nn.PReLU()

        # decoder generates an enhanced signal
        # each decoder output are concatenated with homologous encoder output,
        # so the feature map sizes are doubled
        self.dec10 = nn.ConvTranspose1d(in_channels=2048, out_channels=512, kernel_size=32, stride=2, padding=15) # 解卷积
        '''
        shape:
        输入: (N,C_in,L_in)
        输出: (N,C_out,L_out)
        $$L_{out}=(L_{in}-1)stride-2padding+kernel_size+output_padding$$ 
        '''
        self.dec10_nl = nn.PReLU()  # out : [B x 512 x 16] -> (concat) [B x 1024 x 16]
        self.dec9 = nn.ConvTranspose1d(1024, 256, 32, 2, 15)  # [B x 256 x 32]
        self.dec9_nl = nn.PReLU()
        self.dec8 = nn.ConvTranspose1d(512, 256, 32, 2, 15)  # [B x 256 x 64]
        self.dec8_nl = nn.PReLU()
        self.dec7 = nn.ConvTranspose1d(512, 128, 32, 2, 15)  # [B x 128 x 128]
        self.dec7_nl = nn.PReLU()
        self.dec6 = nn.ConvTranspose1d(256, 128, 32, 2, 15)  # [B x 128 x 256]
        self.dec6_nl = nn.PReLU()
        self.dec5 = nn.ConvTranspose1d(256, 64, 32, 2, 15)  # [B x 64 x 512]
        self.dec5_nl = nn.PReLU()
        self.dec4 = nn.ConvTranspose1d(128, 64, 32, 2, 15)  # [B x 64 x 1024]
        self.dec4_nl = nn.PReLU()
        self.dec3 = nn.ConvTranspose1d(128, 32, 32, 2, 15)  # [B x 32 x 2048]
        self.dec3_nl = nn.PReLU()
        self.dec2 = nn.ConvTranspose1d(64, 32, 32, 2, 15)  # [B x 32 x 4096]
        self.dec2_nl = nn.PReLU()
        self.dec1 = nn.ConvTranspose1d(64, 16, 32, 2, 15)  # [B x 16 x 8192]
        self.dec1_nl = nn.PReLU()
        self.dec_final = nn.ConvTranspose1d(32, 1, 32, 2, 15)  # [B x 1 x 16384]
        self.dec_tanh = nn.Tanh()

        # initialize weights
        self.init_weights()

    def init_weights(self):
        """
        Initialize weights for convolution layers using Xavier initialization.
        """
        for m in self.modules():# .modules()返回模型里的组成元素  即所有层
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d): #如果是卷积层和反卷积层
                nn.init.xavier_normal(m.weight.data) # xavier_normal 初始化
                                                     # torch.nn.init.xavier_normal_(tensor, gain=1)
                # tensor([[-0.1777,  0.6740,  0.1139],
                #         [ 0.3018, -0.2443,  0.6824]])

    def forward(self, x, z):
        """
        Forward pass of generator.

        Args:
            x: input batch (signal)
            z: latent vector
        """
        # encoding step
        e1 = self.enc1(x)
        e2 = self.enc2(self.enc1_nl(e1))
        e3 = self.enc3(self.enc2_nl(e2))
        e4 = self.enc4(self.enc3_nl(e3))
        e5 = self.enc5(self.enc4_nl(e4))
        e6 = self.enc6(self.enc5_nl(e5))
        e7 = self.enc7(self.enc6_nl(e6))
        e8 = self.enc8(self.enc7_nl(e7))
        e9 = self.enc9(self.enc8_nl(e8))
        e10 = self.enc10(self.enc9_nl(e9))
        e11 = self.enc11(self.enc10_nl(e10))
        # c = compressed feature, the 'thought vector'
        c = self.enc11_nl(e11)

        # concatenate the thought vector with latent variable
        encoded = torch.cat((c, z), dim=1)

        # decoding step
        d10 = self.dec10(encoded)
        # dx_c : concatenated with skip-connected layer's output & passed nonlinear layer
        d10_c = self.dec10_nl(torch.cat((d10, e10), dim=1))
        d9 = self.dec9(d10_c)
        d9_c = self.dec9_nl(torch.cat((d9, e9), dim=1))
        d8 = self.dec8(d9_c)
        d8_c = self.dec8_nl(torch.cat((d8, e8), dim=1))
        d7 = self.dec7(d8_c)
        d7_c = self.dec7_nl(torch.cat((d7, e7), dim=1))
        d6 = self.dec6(d7_c)
        d6_c = self.dec6_nl(torch.cat((d6, e6), dim=1))
        d5 = self.dec5(d6_c)
        d5_c = self.dec5_nl(torch.cat((d5, e5), dim=1))
        d4 = self.dec4(d5_c)
        d4_c = self.dec4_nl(torch.cat((d4, e4), dim=1))
        d3 = self.dec3(d4_c)
        d3_c = self.dec3_nl(torch.cat((d3, e3), dim=1))
        d2 = self.dec2(d3_c)
        d2_c = self.dec2_nl(torch.cat((d2, e2), dim=1))
        d1 = self.dec1(d2_c)
        d1_c = self.dec1_nl(torch.cat((d1, e1), dim=1))
        out = self.dec_tanh(self.dec_final(d1_c))
        return out


class Discriminator(nn.Module):
    """D"""

    def __init__(self):
        super().__init__()
        # D gets a noisy signal and clear signal as input [B x 2 x 16384]
        negative_slope = 0.03
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=32, kernel_size=31, stride=2, padding=15)  # [B x 32 x 8192]
        self.lrelu1 = nn.LeakyReLU(negative_slope)
        '''
        
        torch.nn.LeakyReLU(negative_slope=0.01, inplace=False)

        对输入的每一个元素运用$f(x) = max(0, x) + {negative_slope} * min(0, x)$

        参数：

        negative_slope：控制负斜率的角度，默认等于0.01
        inplace-选择是否进行覆盖运算
        '''

        self.conv2 = nn.Conv1d(32, 64, 31, 2, 15)  # [B x 64 x 4096]
        self.lrelu2 = nn.LeakyReLU(negative_slope)
        self.conv3 = nn.Conv1d(64, 64, 31, 2, 15)  # [B x 64 x 2048]
        self.dropout1 = nn.Dropout()
        self.lrelu3 = nn.LeakyReLU(negative_slope)
        self.conv4 = nn.Conv1d(64, 128, 31, 2, 15)  # [B x 128 x 1024]
        self.lrelu4 = nn.LeakyReLU(negative_slope)
        self.conv5 = nn.Conv1d(128, 128, 31, 2, 15)  # [B x 128 x 512]
        self.lrelu5 = nn.LeakyReLU(negative_slope)
        self.conv6 = nn.Conv1d(128, 256, 31, 2, 15)  # [B x 256 x 256]
        self.dropout2 = nn.Dropout()
        self.lrelu6 = nn.LeakyReLU(negative_slope)
        self.conv7 = nn.Conv1d(256, 256, 31, 2, 15)  # [B x 256 x 128]
        self.lrelu7 = nn.LeakyReLU(negative_slope)
        self.conv8 = nn.Conv1d(256, 512, 31, 2, 15)  # [B x 512 x 64]
        self.lrelu8 = nn.LeakyReLU(negative_slope)
        self.conv9 = nn.Conv1d(512, 512, 31, 2, 15)  # [B x 512 x 32]
        self.dropout3 = nn.Dropout()
        self.lrelu9 = nn.LeakyReLU(negative_slope)
        self.conv10 = nn.Conv1d(512, 1024, 31, 2, 15)  # [B x 1024 x 16]
        self.lrelu10 = nn.LeakyReLU(negative_slope)
        self.conv11 = nn.Conv1d(1024, 2048, 31, 2, 15)  # [B x 2048 x 8]
        self.lrelu11 = nn.LeakyReLU(negative_slope)
        # 1x1 size kernel for dimension and parameter reduction
        self.conv_final = nn.Conv1d(2048, 1, kernel_size=1, stride=1)  # [B x 1 x 8]
        self.lrelu_final = nn.LeakyReLU(negative_slope)
        self.fully_connected = nn.Linear(in_features=8, out_features=1)  # [B x 1]
        self.sigmoid = nn.Sigmoid()

        # initialize weights
        self.init_weights()

    def init_weights(self):
        """
        Initialize weights for convolution layers using Xavier initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal(m.weight.data)

    def forward(self, x_c_bach):
        """
        Forward pass of discriminator.

        Args:
            x_c_bach: input batch (signal)
        """

        # train pass
        x_c_bach = self.conv1(x_c_bach)
        x_c_bach = self.lrelu1(x_c_bach)

        x_c_bach = self.conv2(x_c_bach)
        x_c_bach = self.lrelu2(x_c_bach)

        x_c_bach = self.conv3(x_c_bach)
        x_c_bach = self.dropout1(x_c_bach)
        x_c_bach = self.lrelu3(x_c_bach)

        x_c_bach = self.conv4(x_c_bach)
        x_c_bach = self.lrelu4(x_c_bach)

        x_c_bach = self.conv5(x_c_bach)
        x_c_bach = self.lrelu5(x_c_bach)

        x_c_bach = self.conv6(x_c_bach)
        x_c_bach = self.dropout2(x_c_bach)
        x_c_bach = self.lrelu6(x_c_bach)

        x_c_bach = self.conv7(x_c_bach)
        x_c_bach = self.lrelu7(x_c_bach)

        x_c_bach = self.conv8(x_c_bach)
        x_c_bach = self.lrelu8(x_c_bach)

        x_c_bach = self.conv9(x_c_bach)
        x_c_bach = self.dropout3(x_c_bach)
        x_c_bach = self.lrelu9(x_c_bach)

        x_c_bach = self.conv10(x_c_bach)
        x_c_bach = self.lrelu10(x_c_bach)

        x_c_bach = self.conv11(x_c_bach)
        x_c_bach = self.lrelu11(x_c_bach)

        x_c_bach = self.conv_final(x_c_bach)
        x_c_bach = self.lrelu_final(x_c_bach)
        # reduce down to a scalar value
        x_c_bach = torch.squeeze(x_c_bach)
        x_c_bach = self.fully_connected(x_c_bach)
        return self.sigmoid(x_c_bach)

import torch.nn as nn
import params
import torch.nn.functional as F

class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, num_layers=2,bidirectional=True,dropout=0.3)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class CRNN(nn.Module):

    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        conv1 = nn.Conv2d(in_channels=1,out_channels=64,kernel_size=3,padding=1)
        pool1 = nn.MaxPool2d(kernel_size=(2,2))
        conv2 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
        pool2 = nn.MaxPool2d(kernel_size=(2,1))
        drop_out2 = nn.Dropout2d(p=0.2)
        conv3 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,padding=1)
        pool3 = nn.MaxPool2d(kernel_size=(2,2))
        batch_norm3 = nn.BatchNorm2d(128)
        conv4 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,padding=1)
        conv5 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,padding=1)
        pool5 = nn.MaxPool2d(kernel_size=(2,2))
        drop_out5 = nn.Dropout2d(p=0.2)
        conv6 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,padding=1)
        pool6 = nn.MaxPool2d(kernel_size=(2,1))
        batch_norm6 = nn.BatchNorm2d(512)
        conv7 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding=1)
        pool7 = nn.MaxPool2d(kernel_size=(2,1))




        



        cnn = nn.Sequential()
        cnn.add_module('Conv1',conv1)
        cnn.add_module('poo1',pool1)

        cnn.add_module('Conv2',conv2)
        cnn.add_module('poo2',pool2)
        cnn.add_module('Drop2',drop_out2)

        cnn.add_module('conv3',conv3)
        cnn.add_module('pool3',pool3)
        cnn.add_module('b_norm3',batch_norm3)

        cnn.add_module('conv4',conv4)

        cnn.add_module('conv5',conv5)
        cnn.add_module('pool5',pool5)
        cnn.add_module('drop5',drop_out5)

        cnn.add_module('conv6',conv6)
        cnn.add_module('pool6',pool6)
        cnn.add_module('batch_norm',batch_norm6)
        cnn.add_module('conv7',conv7)
        cnn.add_module('pool7',pool7)

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))


    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = self.rnn(conv)
        
        # add log_softmax to converge output
        output = F.log_softmax(output, dim=2)

        return output


    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0   # replace all nan/inf in gradients to zero


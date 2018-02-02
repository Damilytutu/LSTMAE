import torch
from torch import nn
from torch import autograd
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn import init
from torch.nn.modules import rnn
from torch.nn.parameter import Parameter


class LSTMCell(rnn.RNNCellBase):
    def __init__(self, input_size, hidden_size, dropout=0.):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.W = Parameter(torch.Tensor(input_size + hidden_size, hidden_size * 4))
        self.b = Parameter(torch.Tensor(hidden_size * 4))

        self._dropout_mask = None

        self.reset_parameters()

    def reset_parameters(self):
        weight_x = init.orthogonal(torch.Tensor(
            self.input_size, 4 * self.hidden_size))
        weight_h = torch.eye(self.hidden_size).repeat(1, 4)
        self.W.data.set_(torch.cat((weight_x, weight_h), 0))
        self.b.data.fill_(0.)
        self.b.data[:self.hidden_size].fill_(1.)

    def set_dropout_mask(self, batch_size):
        if self.training:
            self._dropout_mask = Variable(torch.bernoulli(
                torch.Tensor(self.hidden_size)
                .fill_(1-self.dropout)), requires_grad=False).cuda()
        else:
            self._dropout_mask = 1 - self.dropout

    def forward(self, input, hidden_state):
        h_0, c_0 = hidden_state     # input: 150 100 100  h_0: (256, 100) c_0:(256, 100)
        batch_size = input.size(0)
        inp = torch.cat((h_0, input), 1)

        bias_batch = (self.b.unsqueeze(0).expand(batch_size, *self.b.size()))
        pre_activations = torch.addmm(bias_batch, inp, self.W)

        f, i, o, g = torch.split(pre_activations, split_size=self.hidden_size, dim=1)
        c_1 = F.sigmoid(f) * c_0 + F.sigmoid(i) * F.tanh(g)
        h_1 = F.sigmoid(o) * F.tanh(c_1)

        h_1 = h_1 * self._dropout_mask + h_0 * (1 - self._dropout_mask)
        c_1 = c_1 * self._dropout_mask + c_0 * (1 - self._dropout_mask)

        return h_1, c_1


class GenLSTM(nn.Module):
    def __init__(self, batch_size, input_dim, hidden_dim, class_num):
        super(GenLSTM, self).__init__()
        
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = class_num
        
        self.lstmen1 = LSTMCell(input_dim, hidden_dim, dropout=0.1)
        self.lstmen2 = LSTMCell(hidden_dim, hidden_dim, dropout=0.1)
        
        self.lstmde1 = LSTMCell(hidden_dim, input_dim, dropout=0.1)
        self.lstmde2 = LSTMCell(input_dim, input_dim, dropout=0.1)
        
    def init_hidden_1(self, batch_size):
        return (autograd.Variable(torch.zeros(batch_size, self.input_dim).cuda(), requires_grad=False), \
                autograd.Variable(torch.zeros(batch_size, self.input_dim).cuda(), requires_grad=False))
    
    def init_hidden_2(self, batch_size):
        return (autograd.Variable(torch.zeros(batch_size, self.hidden_dim).cuda(), requires_grad=False), \
                autograd.Variable(torch.zeros(batch_size, self.hidden_dim).cuda(), requires_grad=False))

    def forward(self, struct):
        batch_size = struct.size(0)
        hiddens_en1 = self.init_hidden_2(batch_size)
        hiddens_en2 = self.init_hidden_2(batch_size)
        hiddens_de1 = self.init_hidden_1(batch_size)
        hiddens_de2 = self.init_hidden_1(batch_size)
        
        feature = []
        
        self.lstmen1.set_dropout_mask(batch_size)
        self.lstmen2.set_dropout_mask(batch_size)
        self.lstmde1.set_dropout_mask(batch_size)
        self.lstmde2.set_dropout_mask(batch_size)
   
        for i, inp_t in enumerate(struct.chunk(struct.size(1), dim=1)):
            inp_t = inp_t.squeeze()
            hiddens_en1 = self.lstmen1(inp_t, hiddens_en1)
            hiddens_en2 = self.lstmen2(hiddens_en1[0], hiddens_en2)
            hiddens_de1 = self.lstmde1(hiddens_en2[0], hiddens_de1)
            hiddens_de2 = self.lstmde2(hiddens_de1[0], hiddens_de2)
            feature.append(hiddens_de2[0])

        feature = torch.stack(feature, 1).squeeze(2)    #feature: (256, 100, 150)

        return feature


class AttLSTM(nn.Module):
    def __init__(self, batch_size, input_dim, hidden_dim, class_num):
        super(AttLSTM, self).__init__()
        
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.num_classes = class_num

        self.lstm1 = LSTMCell(input_dim, hidden_dim, dropout=0.1)
        self.lstm2 = LSTMCell(hidden_dim, hidden_dim, dropout=0.1)
        self.lstm3 = LSTMCell(hidden_dim, hidden_dim, dropout=0.1)

        self.fc = nn.Linear(hidden_dim, class_num)

    def init_hidden(self, batch_size):
        return (autograd.Variable(torch.zeros(batch_size, self.hidden_dim).cuda(), requires_grad=False), \
                autograd.Variable(torch.zeros(batch_size, self.hidden_dim).cuda(), requires_grad=False))

    def forward(self, struct):
        batch_size = struct.size(0)
        hiddens1 = self.init_hidden(batch_size)
        hiddens2 = self.init_hidden(batch_size)
        hiddens3 = self.init_hidden(batch_size)
        feature = []

        self.lstm1.set_dropout_mask(batch_size)
        self.lstm2.set_dropout_mask(batch_size)
        self.lstm3.set_dropout_mask(batch_size)

        for i, inp_t in enumerate(struct.chunk(struct.size(1), dim=1)):
            inp_t = inp_t.squeeze()
            hiddens1 = self.lstm1(inp_t, hiddens1)
            hiddens2 = self.lstm2(hiddens1[0], hiddens2)
            hiddens3 = self.lstm3(hiddens2[0], hiddens3)
            feature.append(hiddens3[0])

        feature = torch.stack(feature, 1).squeeze(2)

        feature = feature[:, -1, :]
        prediction = self.fc(feature)
        prediction = F.elu(prediction)
        return prediction

    
    

    
if __name__ == '__main__':
    ts1 = torch.Tensor(256, 100, 150)
    vr1 = Variable(ts1).cuda()
    net = AttLSTM(256, 150, 100, 60).cuda()
    net(vr1)

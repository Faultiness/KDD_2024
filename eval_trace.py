import pandas as pd
#import modin.pandas as pd
#import ray.dataframe as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import time
import os
import numpy.ma as ma
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from matplotlib import pyplot as plt
from torch.autograd import Variable
import torch.nn.functional as F
import math

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
factor_scale = StandardScaler()
f_count = 157
m_batch = 187030
dft_size = 8500
cls_gap = 12246
t_batch = 185445
tr_size = 117
tr_item = 1585

class MLPBase(nn.Module):
    def __init__(self, input_size, out_size = 1, dropout_rate = 0.5):
        super(MLPBase, self).__init__()
        self.fc = nn.Linear(input_size, 1)
    def forward(self, out):
        out = self.fc(out)
        return out[:, 0]

class LSTMBase(nn.Module):
    def __init__(self, network_input_size, hidden_layer_size, out_size, hidden_layer_num = 1):
        super(LSTMBase, self).__init__()
        self.lstm = nn.LSTM(input_size = network_input_size, hidden_size = hidden_layer_size, num_layers = hidden_layer_num)
        self.out = nn.Linear(hidden_layer_size, out_size)
    def forward(self, out):
        out,_ = self.lstm(out)
        s,b,h = out.size()
        out = out.view(s*b,h)
        out = self.out(out)
        out = out.view(s,b,-1)
        return out

class LSTMBaseEnhanced(nn.Module):
    def __init__(self, network_input_size, hidden_layer_size, out_size, hidden_layer_num = 3):
        super(LSTMBaseEnhanced, self).__init__()
        self.lstm = nn.LSTM(input_size = network_input_size, hidden_size = hidden_layer_size, num_layers = hidden_layer_num)
        self.out = nn.Linear(hidden_layer_size, out_size)
    def forward(self, out):
        out,_ = self.lstm(out)
        s,b,h = out.size()
        out = out.view(s*b,h)
        out = self.out(out)
        out = out.view(s,b,-1)
        return out

class GRUBase(nn.Module):
    def __init__(self, network_input_size, hidden_layer_size, out_size, hidden_layer_num = 2):
        super(GRUBase, self).__init__()
        self.gru = nn.GRU(input_size = network_input_size, hidden_size = hidden_layer_size, num_layers = hidden_layer_num)
        self.out = nn.Linear(hidden_layer_size, out_size)
    def forward(self, out):
        out,_ = self.gru(out)
        s,b,h = out.size()
        out = out.view(s*b,h)
        out = self.out(out)
        out = out.view(s,b,-1)
        return out

class pEncoder(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(pEncoder, self).__init__()       
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = 1 / (10000 ** ((2 * np.arange(d_model)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term[0::2])
        pe[:, 1::2] = torch.cos(position * div_term[1::2])
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    def forward(self, x): return x + self.pe[:x.size(0), :].repeat(1,x.shape[1],1)

class TransAmBase(nn.Module):
    def __init__(self, feature_size=100, num_layers=1, dropout=0.1):
        super(TransAmBase, self).__init__()
        self.src_mask = None
        self.pos_encoder = pEncoder(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=10, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)        
        self.decoder = nn.Linear(feature_size,1)
        self.init_weights()
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-0.1, 0.1)

    def forward(self,src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src,self.src_mask)
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class LSTNetBase(nn.Module):
    def __init__(self, args, data):
        super(LSTNetBase, self).__init__()
        self.P = 12
        self.m = 1
        self.hidR = 100
        self.hidC = 100
        self.hidS = 50
        self.Ck = 6
        self.skip = 6
        self.pt = (self.P - self.Ck)/self.skip
        self.hw = 3
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size = (self.Ck, self.m))
        self.GRU1 = nn.GRU(self.hidC, self.hidR)
        self.dropout = nn.Dropout(p = args.dropout)
        if (self.skip > 0):
            self.GRUskip = nn.GRU(self.hidC, self.hidS)
            self.linear1 = nn.Linear(self.hidR + self.skip * self.hidS, self.m)
        else:
            self.linear1 = nn.Linear(self.hidR, self.m)
        if (self.hw > 0):
            self.highway = nn.Linear(self.hw, 1)
        self.output = None
        if (args.output_fun == 'sigmoid'):
            self.output = F.sigmoid
        if (args.output_fun == 'tanh'):
            self.output = F.tanh
 
    def forward(self, x):
        batch_size = x.size(0)
        
        c = x.view(-1, 1, self.P, self.m)
        c = F.relu(self.conv1(c))
        c = self.dropout(c)
        c = torch.squeeze(c, 3)

        r = c.permute(2, 0, 1).contiguous()
        _, r = self.GRU1(r)
        r = self.dropout(torch.squeeze(r,0))

        if (self.skip > 0):
            s = c[:,:, int(-self.pt * self.skip):].contiguous()
            s = s.view(batch_size, self.hidC, self.pt, self.skip)
            s = s.permute(2,0,3,1).contiguous()
            s = s.view(self.pt, batch_size * self.skip, self.hidC)
            _, s = self.GRUskip(s)
            s = s.view(batch_size, self.skip * self.hidS)
            s = self.dropout(s)
            r = torch.cat((r,s),1)
        
        res = self.linear1(r)
        
        if (self.hw > 0):
            z = x[:, -self.hw:, :]
            z = z.permute(0,2,1).contiguous().view(-1, self.hw)
            z = self.highway(z)
            z = z.view(-1,self.m)
            res = res + z
            
        if (self.output): res = self.output(res)
        return res

class EvalTrace():
    def __init__(self):
        self.factor_target = np.load('target.npy')
        factor_scaled_numpy = np.load('factors.npy')
        self.super_factor_numpy = np.zeros(shape = (m_batch, cls_gap))
        index = 0
        stTime = time.time()
        for i in range(f_count):
            for j in range(i + 1, f_count):
                self.super_factor_numpy[:, index] = factor_scaled_numpy[:, i] + factor_scaled_numpy[:, j]
                index += 1
        for i in range(f_count):
            for j in range(i + 1, f_count):
                self.super_factor_numpy[:, index] = factor_scaled_numpy[:, i] * factor_scaled_numpy[:, j]
                index += 1

        #self.super_factor_numpy = np.where(np.isnan(self.super_factor_numpy), np.ma.array(self.super_factor_numpy, mask = np.isnan(self.super_factor_numpy)).mean(axis = 0), self.super_factor_numpy)
        print('nan count:', len(self.super_factor_numpy[np.isnan(self.super_factor_numpy)]))
        edTime = time.time()
        self.super_factor_pd = pd.DataFrame(self.super_factor_numpy)
        del self.super_factor_numpy
        #self.super_factor_pd = pd.DataFrame(factor_scale.fit_transform(np.asarray(self.super_factor_pd)))
        #del self.super_factor_numpy
        print('gene super ok', edTime - stTime)
        self.MLP = MLPBase(dft_size, 1).to(device)
        self.mlp_optimizer = torch.optim.Adam(self.MLP.parameters())
        self.mlp_criterion = nn.MSELoss()
        self.LSTM = LSTMBase(dft_size, 100, 1).to(device)
        self.lstm_optimizer = torch.optim.Adam(self.LSTM.parameters())
        self.lstm_criterion = nn.MSELoss()
        self.GRU = GRUBase(dft_size, 100, 1).to(device)
        self.gru_optimizer = torch.optim.Adam(self.GRU.parameters())
        self.gru_criterion = nn.MSELoss()
        self.LSTMEnhanced = LSTMBaseEnhanced(dft_size, 500, 1).to(device)
        self.lstm_enhanced_optimizer = torch.optim.Adam(self.LSTMEnhanced.parameters())
        self.lstm_enhanced_criterion = nn.MSELoss()
        self.Trans = TransAmBase().to(device)
        self.trans_optimizer = torch.optim.Adam(self.Trans.parameters())
        self.trans_criterion = nn.MSELoss()
        self.LSTNet = LSTNetBase().to(device)
        self.lstnet_optimizer = torch.optim.Adam(self.LSTNet.parameters())
        self.lstnet_criterion = nn.MSELoss()

    def generate_factor_interaction_v2(self, factor_trace):
        factor_list = []
        for i in range(len(factor_trace)):
            if factor_trace[i] == '0': continue
            factor_list.append(i if factor_trace[i] == '1' else i+cls_gap if factor_trace[i] == '2' else i+cls_gap*2)
        new_factor = self.super_factor_pd.iloc[:, factor_list].to_numpy()
        return new_factor

    def eval_mlp(self, factor_trace, eval_trace = False):
        stTime = time.time()
        new_factor = self.generate_factor_interaction_v2(factor_trace)
        edTime = time.time()
        print('shape:', new_factor.shape, ', t =', edTime - stTime)

        if new_factor.shape[1] < dft_size: new_factor = np.column_stack((new_factor, np.zeros((m_batch, dft_size - new_factor.shape[1]))))
        else: new_factor = new_factor[:, dft_size:]
        X_train, X_test, y_train, y_test = train_test_split(new_factor, self.factor_target, test_size = 0.3)
        train_xt, train_yt = torch.from_numpy(X_train.astype(np.float32)), torch.from_numpy(y_train.astype(np.float32))
        if not eval_trace: test_xt = torch.from_numpy(X_test.astype(np.float32))

        train_data = TensorDataset(train_xt, train_yt)
        train_loader = enumerate(DataLoader(dataset = train_data, batch_size = m_batch, shuffle = True))

        if eval_trace:
            self.MLP.train()
            step, (batch_x, batch_y) = next(train_loader)
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            self.mlp_optimizer.zero_grad()
            out = self.MLP(batch_x)
            loss = self.mlp_criterion(out, batch_y)
            return loss

    def update_mlp(self, mean_loss): 
        mean_loss.backward()
        self.mlp_optimizer.step()
    
    def eval_lstm(self, factor_trace, eval_trace = False):
        stTime = time.time()
        new_factor = self.generate_factor_interaction_v2(factor_trace)
        edTime = time.time()
        print('shape:', new_factor.shape, ', t =', edTime - stTime)

        if new_factor.shape[1] < dft_size: new_factor = np.column_stack((new_factor, np.zeros((m_batch, dft_size - new_factor.shape[1]))))
        else: new_factor = new_factor[:, dft_size:]
        X_train, y_train = new_factor[:t_batch, :], self.factor_target[:t_batch]
        X_train_t, y_train_t = Variable(torch.Tensor(X_train)), Variable(torch.Tensor(y_train))
        X_train_tensor = torch.reshape(X_train_t, (tr_size, tr_item, X_train_t.shape[1]))
        y_train_tensor = torch.reshape(y_train_t, (tr_size, tr_item, 1))

        X_test_t = Variable(torch.Tensor(new_factor))
        X_test_tensor = torch.reshape(X_test_t, (tr_size+1, tr_item, X_test_t.shape[1]))
        train_data = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = enumerate(DataLoader(dataset = train_data, batch_size = tr_item * 24, shuffle = True))

        if eval_trace:
            self.LSTM.train()
            step, (batch_x, batch_y) = next(train_loader)
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            self.lstm_optimizer.zero_grad()
            out = self.LSTM(batch_x)
            loss = self.lstm_criterion(out, batch_y)
            return loss

    def update_lstm(self, mean_loss): 
        mean_loss.backward()
        self.lstm_optimizer.step()
    
    def eval_lstm_enhanced(self, factor_trace, eval_trace = False):
        stTime = time.time()
        new_factor = self.generate_factor_interaction_v2(factor_trace)
        edTime = time.time()
        print('shape:', new_factor.shape, ', t =', edTime - stTime)

        if new_factor.shape[1] < dft_size: new_factor = np.column_stack((new_factor, np.zeros((m_batch, dft_size - new_factor.shape[1]))))
        else: new_factor = new_factor[:, dft_size:]
        X_train, y_train = new_factor[:t_batch, :], self.factor_target[:t_batch]
        X_train_t, y_train_t = Variable(torch.Tensor(X_train)), Variable(torch.Tensor(y_train))
        X_train_tensor = torch.reshape(X_train_t, (tr_size, tr_item, X_train_t.shape[1]))
        y_train_tensor = torch.reshape(y_train_t, (tr_size, tr_item, 1))

        X_test_t = Variable(torch.Tensor(new_factor))
        X_test_tensor = torch.reshape(X_test_t, (tr_size+1, tr_item, X_test_t.shape[1]))
        train_data = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = enumerate(DataLoader(dataset = train_data, batch_size = tr_item * 24, shuffle = True))

        if eval_trace:
            self.LSTMEnhanced.train()
            step, (batch_x, batch_y) = next(train_loader)
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            self.lstm_enhanced_optimizer.zero_grad()
            out = self.LSTMEnhanced(batch_x)
            loss = self.lstm_enhanced_criterion(out, batch_y)
            return loss

    def update_lstm_enhanced(self, mean_loss): 
        mean_loss.backward()
        self.lstm_enhanced_optimizer.step()

    def eval_gru(self, factor_trace, eval_trace = False):
        stTime = time.time()
        new_factor = self.generate_factor_interaction_v2(factor_trace)
        edTime = time.time()
        print('shape:', new_factor.shape, ', t =', edTime - stTime)

        if new_factor.shape[1] < dft_size: new_factor = np.column_stack((new_factor, np.zeros((m_batch, dft_size - new_factor.shape[1]))))
        else: new_factor = new_factor[:, dft_size:]
        X_train, y_train = new_factor[:t_batch, :], self.factor_target[:t_batch]
        X_train_t, y_train_t = Variable(torch.Tensor(X_train)), Variable(torch.Tensor(y_train))
        X_train_tensor = torch.reshape(X_train_t, (tr_size, tr_item, X_train_t.shape[1]))
        y_train_tensor = torch.reshape(y_train_t, (tr_size, tr_item, 1))

        X_test_t = Variable(torch.Tensor(new_factor))
        X_test_tensor = torch.reshape(X_test_t, (tr_size+1, tr_item, X_test_t.shape[1]))
        train_data = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = enumerate(DataLoader(dataset = train_data, batch_size = tr_item * 24, shuffle = True))

        if eval_trace:
            self.GRU.train()
            step, (batch_x, batch_y) = next(train_loader)
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            self.gru_optimizer.zero_grad()
            out = self.GRU(batch_x)
            loss = self.gru_criterion(out, batch_y)
            return loss

    def update_gru(self, mean_loss): 
        mean_loss.backward()
        self.gru_optimizer.step()

    def eval_update_xgboost(self, factor_trace):
        stTime = time.time()
        new_factor = self.generate_factor_interaction_v2(factor_trace)
        edTime = time.time()
        print('shape:', new_factor.shape, ', t =', edTime - stTime)
        
        X_train, X_test, y_train, y_test = train_test_split(new_factor, self.factor_target, test_size = 0.3)
        model = xgb.XGBRegressor(n_estimators=10, learning_rate=0.05, max_depth=5, silent=True, objective='reg:gamma')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        loss = mean_squared_error(y_test, y_pred)
        return loss

    def eval_trans(self, factor_trace, eval_trace = False):
        stTime = time.time()
        new_factor = self.generate_factor_interaction_v2(factor_trace)
        edTime = time.time()
        print('shape:', new_factor.shape, ', t =', edTime - stTime)

        X_train, y_train = new_factor[:t_batch, :], self.factor_target[:t_batch]
        X_train_t, y_train_t = Variable(torch.Tensor(X_train)), Variable(torch.Tensor(y_train))
        X_train_tensor = torch.reshape(X_train_t, (tr_size, tr_item, X_train_t.shape[1]))
        y_train_tensor = torch.reshape(y_train_t, (tr_size, tr_item, 1))

        X_test_t = Variable(torch.Tensor(new_factor))
        X_test_tensor = torch.reshape(X_test_t, (tr_size+1, tr_item, X_test_t.shape[1]))
        train_data = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = enumerate(DataLoader(dataset = train_data, batch_size = tr_item * 24, shuffle = True))

        if eval_trace:
            self.Trans.train()
            step, (batch_x, batch_y) = next(train_loader)
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            self.trans_optimizer.zero_grad()
            out = self.Trans(batch_x)
            loss = self.trans_criterion(out, batch_y)
            return loss

    def update_trans(self, mean_loss): 
        mean_loss.backward()
        self.trans_optimizer.step()

    def eval_lstnet(self, factor_trace, eval_trace = False):
        stTime = time.time()
        new_factor = self.generate_factor_interaction_v2(factor_trace)
        edTime = time.time()
        print('shape:', new_factor.shape, ', t =', edTime - stTime)

        if new_factor.shape[1] < dft_size: new_factor = np.column_stack((new_factor, np.zeros((m_batch, dft_size - new_factor.shape[1]))))
        else: new_factor = new_factor[:, dft_size:]
        X_train, y_train = new_factor[:t_batch, :], self.factor_target[:t_batch]
        X_train_t, y_train_t = Variable(torch.Tensor(X_train)), Variable(torch.Tensor(y_train))
        X_train_tensor = torch.reshape(X_train_t, (tr_size, tr_item, X_train_t.shape[1]))
        y_train_tensor = torch.reshape(y_train_t, (tr_size, tr_item, 1))

        X_test_t = Variable(torch.Tensor(new_factor))
        X_test_tensor = torch.reshape(X_test_t, (tr_size+1, tr_item, X_test_t.shape[1]))
        train_data = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = enumerate(DataLoader(dataset = train_data, batch_size = tr_item * 24, shuffle = True))

        if eval_trace:
            self.LSTNet.train()
            step, (batch_x, batch_y) = next(train_loader)
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            self.lstnet_optimizer.zero_grad()
            out = self.LSTNet(batch_x)
            loss = self.lstnet_criterion(out, batch_y)
            return loss

    def update_lstnet(self, mean_loss): 
        mean_loss.backward()
        self.lstnet_optimizer.step()
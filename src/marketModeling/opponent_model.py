import torch
from torch import nn
import torch.nn.functional as F

from .util_funcs import loss_full
from .Transformer import Transformer, NeuralNet

import os.path as osp
import os
import numpy as np


torch.manual_seed(config.seed)
seed=config.seed


class opponent(object):
    def __init__(self, 
                alpha,
                beta,
                epochs,
                num_outputs,
                h1,
                h2,
                vocal_size,
                model_path,
                camp,
                sample_size,
                num_features,
                emb_dropout,
                lin_dropout,
                c0,
                reward,
                agent_num,
                agent_index,
                use_cuda,
                op):
        self.alpha          = alpha
        self.beta           = beta

        self.num_outputs    = num_outputs
        self.h1             = h1
        self.h2             = h2
        self.epochs         = epochs
        self.model_path     = model_path
        self.camp           = camp
        self.sample_size    = sample_size
        
        self.vocal_size     = vocal_size
        self.num_features   = num_features
        self.emb_dropout    = emb_dropout
        self.lin_dropout    = lin_dropout
        
        self.c0             = c0
        self.reward         = reward
        self.agent_num      = agent_num
        self.agent_index    = agent_index
        self.use_cuda       = use_cuda
        self.op             = op
        self.use_cuda       = use_cuda
        
        if self.op == 'ffn':
            self.model = NeuralNet(self.vocal_size, 
                                self.num_features,
                                self.num_outputs, 
                                self.h1,
                                self.h2, 
                                self.emb_dropout,
                                self.lin_dropout)
        else:
            #TODO Hard Coding?
            self.model = Transformer(vocal_size     = vocal_size,
                                    num_features    = num_features,
                                    num_outputs     = num_outputs,
                                    model_dim       = 16,
                                    drop_prob       = 0.1,
                                    point_wise_dim  = 64,
                                    num_sublayer    = 1,
                                    num_head        = 2,
                                    is_cuda         = use_cuda)
        if use_cuda:
            self.model.cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=alpha)
        self.best_model_index = 0

    def train(self, training_generator, val_generator, file_prefix, camp):
        train_loss          = []
        iter_train_loss     = []
        val_loss            = []
        val_anlp            = []
        counter             = 0
        best_score          = 1e10    
        best_score_val      = 1e10
        save_path           = self.model_path
        if not osp.exists(save_path):
                os.makedirs(save_path)
        
        for epoch in range(self.epochs):
            print('start epoch %d' % epoch)
            self.model.train(True)
            running_loss = 0.0
            n_batch = 0
            for x_batch, c_batch,  m1_batch, m2_batch in training_generator:
                n_batch += 1
                
                if self.use_cuda:
                    x_batch, c_batch, m1_batch, m2_batch = Variable(x_batch.cuda()), \
                                                            Variable(c_batch.cuda()), \
                                                            Variable(m1_batch.cuda()), \
                                                            Variable(m2_batch.cuda())
                    x_batch = x_batch.type(torch.cuda.LongTensor)

                else:
                    x_batch, c_batch, m1_batch, m2_batch = Variable(x_batch), Variable(c_batch), \
                                                        Variable(m1_batch), Variable(m2_batch)
                    x_batch = x_batch.type(torch.LongTensor)

                self.optimizer.zero_grad()
                ypred_batch = self.model(x_batch)
                loss = loss_full(c_batch, m1_batch, m2_batch, ypred_batch)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                iter_train_loss.append(loss.item())
                
            # shuffle은 하나?
            # train_loss.append(np.log(running_loss / n_batch)) --- loss에 log는 왜 취하는거지?
            total_loss = running_loss / n_batch
            train_loss.append(total_loss)                    # --- 사실 이것도 정확하지는 않자나.

            # ------------ val
            self.model.eval()
            running_loss = 0.0
            n_batch = 0
            anlp = 0
            hz_list = []
            hzp_list = []
            
            if best_score > total_loss:
                torch.save(self.model.state_dict(), save_path + self.camp + '_train_sample' + str(self.sample_size)
                        + 'epoch_' + str(epoch) + 'lr_' + str(self.alpha) + file_prefix + '.pt')    
                best_score = total_loss

            for x_batch, c_batch,  m1_batch, m2_batch in val_generator:
                batch_size = x_batch.shape[0]
                n_batch += 1

                if self.use_cuda:
                    x_batch, c_batch, m1_batch, m2_batch = Variable(x_batch.cuda()), \
                                                                    Variable(c_batch.cuda()), \
                                                                    Variable(m1_batch.cuda()), \
                                                                    Variable(m2_batch.cuda())
                    x_batch = x_batch.type(torch.cuda.LongTensor)

                else:
                    x_batch, c_batch, m1_batch, m2_batch = Variable(x_batch), \
                                                            Variable(c_batch), \
                                                            Variable(m1_batch), \
                                                            Variable(m2_batch)
                    x_batch = x_batch.type(torch.LongTensor)

                ypred_valbatch = self.model(x_batch)
                loss = loss_full(c_batch, m1_batch, m2_batch, ypred_valbatch)
                running_loss += loss.item()

                hz = torch.sum(ypred_valbatch * m1_batch, dim=1)
                hzp = torch.prod(1 - ypred_valbatch * m2_batch, dim=1)
                #score = torch.sum(torch.log(hz * hzp)) TODO 체크할 것.
                score = torch.mean(hz * hzp)
                hz_list.append(torch.mean(hz).item())
                hzp_list.append(torch.mean(hzp).item())
                anlp += score.item() 

            temp_val_loss = running_loss / n_batch
            val_anlp.append(anlp / n_batch)
            val_loss.append(temp_val_loss)
            print('validation anlp : {}'.format(val_anlp[-1]))
            
            # TODO 일단 빠르게 진행해보고 추후 모델 저장여부 결정.
            if best_score_val > temp_val_loss:
                torch.save(self.model.state_dict(), save_path + self.camp + '_val_sample' + str(self.sample_size)
                        + 'epoch_' + str(epoch) + 'lr_' + str(self.alpha) + file_prefix + '.pt')    
                best_score_val = temp_val_loss
        
        return train_loss, val_loss, val_anlp

    def test(self, test_generator, best_model_index, c0, file_prefix, mode='test'):
        self.model.load_state_dict(torch.load(self.model_path + self.camp + '_train_sample' + str(self.sample_size)
                                              + 'epoch_' + str(best_model_index) + 'lr_' + str(self.alpha)
                                              + file_prefix + '.pt'))
        self.model.eval()
        anlp = 0
        nbatch = 0

        prediction_path = '../predictions/agent_' + str(self.agent_index) + '/'

        if not osp.exists(prediction_path):
            os.makedirs(prediction_path)

        f = open(prediction_path + self.camp + '_' + mode + file_prefix + '.txt', 'w')
        print(prediction_path + self.camp + '_' + mode + file_prefix + '.txt')

        for x_batch in test_generator:
            nbatch += 1
            # x_batch = get_onehot_data(x_batch, vocal_size)
            if self.use_cuda:
                x_batch = Variable(x_batch.cuda())
                x_batch = x_batch.type(torch.cuda.LongTensor)
            else:
                x_batch = Variable(x_batch)
                x_batch = x_batch.type(torch.LongTensor)

            ypred_testbatch = self.model(x_batch)

            # write the prediction to a file
            batch_list = ypred_testbatch.tolist()

            # calculate the p(z | x, \theta)
            for item in batch_list:
                n_event = [1-x for x in item]
                for i in range(len(item)):
                    if i == 0:
                        f.write('%.6f ' % item[i])
                    else:
                        pz = item[i] * np.prod(n_event[:i])
                        f.write('%.6f ' % pz)
                f.write('\n')

            # hz = torch.sum(ypred_testbatch * m1_batch, dim=1)
            # hzp = torch.prod(1 - ypred_testbatch * m2_batch, dim=1)
            # score = torch.sum(torch.log(hz * hzp))
            # anlp += - score.item() / x_batch.shape[0]
        # test_anlp = anlp / nbatch
        f.close()
        # return test_anlp

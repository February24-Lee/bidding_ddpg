import torch
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import pandas as pd

import argparse
import random
import pickle
import math
from os import path
from copy import deepcopy

from src.agent import DDPGAgent, LinearAgent
from src.env.actionEnv import ActionEnv
from src.dataLoader.ipinyou_dataloader import IPinyouDataLoader

def decision_budget(total_cost  : float = None,
                    total_imp   : float = None,
                    ratio       : float = None,
                    episode_len : float = None) -> float:
    return  total_cost / total_imp * ratio * episode_len

def to_args_format(args : dict,
                    keyword : str):
    args = {}
    for key in args:
        if keyword in key:
            args[key[len(key) : ]] = args[key]
    return args

def train(args):
    
    print(' ==== start training ===')
    
    with open(path.join(args['data_path'], args['camp'], 'info.txt'), 'rb') as f:
        camp_info = pickle.load(f)
        
    ### TODO 후에 수정 필요.
    tb_logger = SummaryWriter('log/')
    
    train_budget            = decision_budget(camp_info['cost_train'],
                                            camp_info['imp_train'],
                                            args['env_budget_ratio'],
                                            args['env_episode_max'])
    
    train_num_auction       = len(camp_info['imp_train'])
    train_epochs            = math.ceil(train_num_auction / args['env_episode_max'])
    train_iteration         = min(train_epochs * train_num_auction, train_num_auction)
    
    train_dataloadr         = IPinyouDataLoader(data_path=args['data_path'],
                                                camp=args['camp'],
                                                file='train.ctr.txt')
    
    # --- Enviroment
    train_env               = ActionEnv(dataloader=train_dataloadr,
                                        episode_maxlen=args['env_episode_max'])
    
    # --- Agent
    ddpg_agent              = DDPGAgent(*to_args_format(args, keyword='ddpg_'),
                                        budget=train_budget,
                                        tb_logger=tb_logger)
    linear_agent            = LinearAgent(camp_info=camp_info,
                                        max_bid_price=args['ddpg_max_bid_price'],
                                        budget=train_budget)
    
    # --- Train
    bid = train_env.reset()
    bid_log = {}
    episode = []
    for step in range(int(train_iteration)):
        
        if step <= args['warmup'] :
            state0, action = ddpg_agent.random_action(bid['pctr'])
            action = action.detach().numpy()
        else:
            state0, action = ddpg_agent.action(bid['pctr'])
        linear_action = linear_action(bid['pctr'])
        episode.append([action, linear_agent, bid['pctr']])
        
        next_bid, reward, terminal, info = train_env.step(np.concatenate([action, linear_action]))
        
        if (reward[0] is not 0) and (action is not 0.):     # ddpg win
            ddpg_agent.update_result(is_win = True,
                                    click = info['click'],
                                    market_price = info['market_price'])
            linear_agent.update_result(is_win=False)
        elif (reward[0] == 0) and (action is not 0.):     # ddpg win
            ddpg_agent.update_result(is_win = False)
            linear_agent.update_result(is_win=True, 
                                    click = info['click'],
                                    market_price = info['market_price'])
            
        ddpg_agent.fill_memory(state0   = state0,
                            action      = action,
                            reward      = info['pctr'],
                            terminal    = terminal)
        
        if step > args['warmup'] :
            ddpg_agent.update_policy()
            
        if terminal :
            ddpg_total_click        = ddpg_agent.num_click
            ddpg_remained_budget    = ddpg_agent.remained_budget
            ddpg_total_win          = ddpg_agent.num_win
            ddpg_total_attened      = ddpg_agent.num_attend_bid
            ddpg_pctr_list          = ddpg_agent.list_pctr
            ddpg_summary = {"ddpg_total_click"      : ddpg_total_click,
                            "ddpg_remained_budget"  : ddpg_remained_budget,
                            "ddpg_total_win"        : ddpg_total_win,
                            "ddpg_total_attened"    : ddpg_total_attened,
                            "ddpg_pctr_list"        : ddpg_pctr_list}
                            
            lin_total_click        = linear_agent.num_click
            lin_remained_budget    = linear_agent.remained_budget
            lin_total_win          = linear_agent.num_win
            lin_total_attened      = linear_agent.num_attend_bid
            lin_pctr_list          = linear_agent.list_pctr
            
            print('---------------------------')
            print('Episode : {}'.format(train_env.episode_idx))
            print('DDPG | Win : {} ({:.5f}%), Total Click : {} ({:.5f}%)'.format(
                                ddpg_total_win, ddpg_total_win/ddpg_total_attened,
                                ddpg_total_click, ddpg_total_click/ddpg_total_attened))
            print('DDPG | average pctr : {}, remained budget'.format(
                                np.mean(ddpg_pctr_list), ddpg_remained_budget))
            
            print('Lin | Win : {} ({:.5f}%), Total Click : {} ({:.5f}%)'.format(
                                lin_total_win, lin_total_win/lin_total_attened,
                                lin_total_click, lin_total_click/lin_total_attened))
            print('Lin | average pctr : {}, remained budget'.format(
                                np.mean(lin_pctr_list), lin_remained_budget))
            print('---------------------------')
            ###########
            #TODO 나중에 txt 저장 방식 수정 할 것. soft-coding으로
            np.savetxt("log/Ep{}_log.txt".format(train_env.episode_idx), np.array(episode), delimiter=",")
            with open("log/Ep{}_ddpg_summary.pickle".format(train_env.episode_idx), 'wb') as f:
                pickle.dump(ddpg_summary, f)
            ###########
            bid_log["Episode : {}".format(train_env.episode_idx)] = np.array(episode)
            episode = []
            
            bid = train_env.reset()
            ddpg_agent.reset()
            linear_agent.reset()
        else : 
            bid = deepcopy(next_bid)
        
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    # --- auction path
    parser.add_argument('--camp',                   type=str,       default='2259')
    parser.add_argument('--data-path',              type=str,       default='data/make-ipinyou-data/')
    parser.add_argument('--seed',                   type=int,       default=777)
    
    # --- environment
    parser.add_argument('--env-episode-max',        type=int,       default=1000)
    parser.add_argument('--env-budget-ratio',       type=float,     default=0.25)
    
    # --- DDPG 
    parser.add_argument('--ddpg-dim-state',         type=int,       default=2)
    parser.add_argument('--ddpg-dim-action',        type=int,       default=1)
    parser.add_argument('--ddpg-actor-optim-lr',    type=float,     default=0.001)
    parser.add_argument('--ddpg-critic-optim-lr',   type=float,     default=0.001)
    parser.add_argument('--ddpg-ou-theta',          type=float,     default=0.15)
    parser.add_argument('--ddpg-ou-mu',             type=float,     default=0.)
    parser.add_argument('--ddpg-ou-sigma',          type=float,     default=0.2)
    parser.add_argument('--ddpg-memory-size',       type=int,       default=1000)
    parser.add_argument('--ddpg-window-length',     type=int,       default=1)
    parser.add_argument('--ddpg-batch-size',        type=int,       default=64)
    parser.add_argument('--ddpg-soft-copy-tau',     type=float,     default=0.001)
    parser.add_argument('--ddpg-discount',          type=float,     default=0.99)
    parser.add_argument('--ddpg-epsilon',           type=int,       default=50000)
    parser.add_argument('--ddpg-max-bid-price',     type=int,       default=300)
    
    # --- train
    parser.add_argument('--warmup',                 type=int,       default=100)
    
    # --- logger
    parser.add_argument('--log-path',               type=str,      default='log/')
    parser.add_argument('--tb-log-path',            type=str,      default='log/')
    
    args = vars(parser.parse_args())
    
    torch.manual_seed(args['seed'])
    np.random.seed(args['seed'])
    random.seed(args['seed'])
    
    train(args)
    
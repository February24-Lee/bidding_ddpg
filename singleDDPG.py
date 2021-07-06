import torch
from torch.utils.tensorboard import SummaryWriter
from test_tube import Experiment
import wandb

import numpy as np
import pandas as pd

import argparse
import random, pickle, math, os
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
    new_args = {}
    for key in args:
        if keyword in key:
            new_args[key[len(keyword) : ]] = args[key]
    return new_args

def train(args):
    
    print(' ==== start training ===')
    
    with open(path.join(args['data_path'], args['camp'], 'info.txt'), 'rb') as f:
        camp_info = pickle.load(f)

    with open(args['lin_b0_path'], 'rb') as f:
        b0_file = pickle.load(f)
        lin_b0 = b0_file['b0']
        
    ### TODO 후에 수정 필요.
    #tb_logger = SummaryWriter('log/')
    tt_logger = Experiment(name=args['log_name'],
                        save_dir = 'log/')
    tt_logger.tag(args)
    tt_logger.save()
    
    train_budget            = decision_budget(camp_info['cost_train'],
                                            camp_info['imp_train'],
                                            args['env_budget_ratio'],
                                            args['env_episode_max'])
    
    train_num_auction       = camp_info['imp_train']
    train_epochs            = math.ceil(train_num_auction / args['env_episode_max'])
    train_iteration         = min(train_epochs * train_num_auction, train_num_auction)
    
    train_dataloadr         = IPinyouDataLoader(data_path=args['data_path'],
                                                market_path=None,
                                                camp=args['camp'],
                                                file='train.ctr.txt')
    
    # --- Enviroment
    train_env               = ActionEnv(dataloader=train_dataloadr,
                                        episode_maxlen=args['env_episode_max'])
    
    # --- Agent
    ddpg_agent              = DDPGAgent(**to_args_format(args, keyword='ddpg_'),
                                        budget=train_budget,
                                        logger=tt_logger)
    if args['load_model'] is not None:
        print('load model')
        ckpt = torch.load(args['load_model'])
        ddpg_agent.actor_target.load_state_dict(ckpt['actor_state_dict'])
        ddpg_agent.critic_target.load_state_dict(ckpt['critic_state_dict'])
        ddpg_agent.actor_optim.load_state_dict(ckpt['optim_actor_state_dict'])
        ddpg_agent.critic_optim.load_state_dict(ckpt['optim_critic_state_dict'])
        ddpg_agent.actor = deepcopy(ddpg_agent.actor_target)
        ddpg_agent.critic = deepcopy(ddpg_agent.critic_target)               
    
    linear_agent            = LinearAgent(camp_info=camp_info,
                                        max_bid_price=args['ddpg_max_bid_price'],
                                        budget=train_budget)
    linear_agent.b0 = lin_b0
    
    # --- Train
    bid = train_env.reset()
    bid_log = {}
    episode = []
    ddpg_agent.is_training = True
    for step in range(int(train_iteration)):

        if ddpg_agent.remained_budget < 100:
            _ = 1
        
        if step <= args['warmup'] :
            state0, action = ddpg_agent.random_action(bid['pctr'])
        else:
            state0, action = ddpg_agent.action(bid['pctr'])
        linear_action = linear_agent.action(bid['pctr'])
        episode.append([action, linear_action, bid['pctr']])
        
        next_bid, reward, terminal, info = train_env.step(np.array([action, linear_action, bid['market_price']]))
        
        if (reward[0] == 1) and (action != 0.):     # ddpg win
            ddpg_agent.update_result(is_win = True,
                                    click = info['click'],
                                    market_price = info['market_price'],
                                    pctr=info['pctr'])
            linear_agent.update_result(is_win=False)
        elif (reward[1] == 0) and (action != 0.):     # lin win
            ddpg_agent.update_result(is_win = False)
            linear_agent.update_result(is_win=True, 
                                    click = info['click'],
                                    market_price = info['market_price'],
                                    pctr=info['pctr'])
            
        if args['env_reward_style'] == 'base':
            _reward = info['pctr']*reward[0]
        elif args['env_reward_style'] == 'minus':
            _reward = info['pctr']*reward[0] - info['pctr']*(reward[1]+reward[2])
        else:
            raise ValueError

        ddpg_agent.fill_memory(state0   = state0,
                            action      = action,
                            reward      = _reward,
                            terminal    = terminal)
        
        if step > args['warmup'] :
            ddpg_agent.update_policy(step)
            
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
                                ddpg_total_win, ddpg_total_win/(ddpg_total_attened+1e-5),
                                ddpg_total_click, ddpg_total_click/(ddpg_total_attened+1e-5)))
            print('DDPG | average pctr : {}, remained budget'.format(
                                np.mean(ddpg_pctr_list), ddpg_remained_budget))
            
            print('Lin | Win : {} ({:.5f}%), Total Click : {} ({:.5f}%)'.format(
                                lin_total_win, lin_total_win/(lin_total_attened+1e-5),
                                lin_total_click, lin_total_click/(lin_total_attened+1e-5)))
            print('Lin | average pctr : {}, remained budget'.format(
                                np.mean(lin_pctr_list), lin_remained_budget))
            print('---------------------------')
            ###########
            #TODO 나중에 txt 저장 방식 수정 할 것. soft-coding으로
            temp_df = pd.DataFrame(np.array(episode))
            if not path.isdir(path.join(tt_logger.save_dir,tt_logger.name, 'version_{}'.format(tt_logger.version), 'log_history')):
                os.mkdir(path.join(tt_logger.save_dir, tt_logger.name,'version_{}'.format(tt_logger.version), 'log_history'))
            temp_df.to_csv(path.join(tt_logger.save_dir, tt_logger.name,'version_{}'.format(tt_logger.version),'log_history', "Ep{}_log.txt".format(train_env.episode_idx)), index=False)
            with open(path.join(tt_logger.save_dir, tt_logger.name,'version_{}'.format(tt_logger.version),'log_history', "Ep{}_ddpg_summary.pickle".format(train_env.episode_idx)), 'wb') as f:
                pickle.dump(ddpg_summary, f)
            ###########
            bid_log["Episode : {}".format(train_env.episode_idx)] = np.array(episode)
            episode = []
            
            bid = train_env.reset()
            ddpg_agent.reset()
            linear_agent.reset()
        else : 
            bid = deepcopy(next_bid)
    
    # Save model
    torch.save({'critic_state_dict'         : ddpg_agent.critic_target.state_dict(),
                'actor_state_dict'          : ddpg_agent.actor_target.state_dict(),
                'optim_critic_state_dict'   : ddpg_agent.critic_optim.state_dict(),
                'optim_actor_state_dict'    : ddpg_agent.actor_optim.state_dict(),
                }, 
                path.join(tt_logger.save_dir,  tt_logger.name, 'version_{}'.format(tt_logger.version),  'final_model'))
    
if __name__ == "__main__":
    wandb.init()
    
    parser = argparse.ArgumentParser()
    
    # --- auction path
    parser.add_argument('--camp',                   type=str,       default='2259')
    parser.add_argument('--data-path',              type=str,       default='data/make-ipinyou-data/')
    parser.add_argument('--seed',                   type=int,       default=777)
    parser.add_argument('--load-model',             type=str,       )
    parser.add_argument('--epoch',                  type=str,       default=2)
    
    # --- environment
    parser.add_argument('--env-episode-max',        type=int,       default=1000)
    parser.add_argument('--env-budget-ratio',       type=float,     default=0.25)
    parser.add_argument('--env_reward_style',       type=str,       default='base')
    
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
    parser.add_argument('--ddpg-epsilon',           type=int,       default=1)
    parser.add_argument('--ddpg-max-bid-price',     type=int,       default=300)
    parser.add_argument('--ddpg-num_actor_layer',   type=int,       default=4)
    parser.add_argument('--ddpg-dim_actor_layer',   type=int,       default=16)
    parser.add_argument('--ddpg-num_critic_layer',  type=int,       default=4)
    parser.add_argument('--ddpg-dim_critic_layer',  type=int,       default=16)
    
    # --- linear agent
    parser.add_argument('--lin-b0-path',            type=str,       default='data/linear_agent/ipinyou-data/2259/bid-model/lin-bid_1000_0.25_clk_277696.pickle')

    # --- train
    parser.add_argument('--warmup',                 type=int,       default=100)
 
    # --- logger
    parser.add_argument('--log-path',               type=str,      default='log/')
    parser.add_argument('--tb-log-path',            type=str,      default='log/')
    parser.add_argument('--log-name',               type=str,      default='base_reward_base')
    
    args = vars(parser.parse_args())
    
    torch.manual_seed(args['seed'])
    np.random.seed(args['seed'])
    random.seed(args['seed'])
    
    train(args)
    
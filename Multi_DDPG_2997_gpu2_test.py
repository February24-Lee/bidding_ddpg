import torch
from torch.utils.tensorboard import SummaryWriter
from test_tube import Experiment

import numpy as np
import pandas as pd

import argparse
import random, pickle, math, os
from os import path
from copy import deepcopy
import json

from src.agent import DDPGAgent, LinearAgent, McpcAgent
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

def test(args):
    
    print(' ==== start training ===')
    
    # TODO modified
    #with open(path.join(args['data_path'], args['camp'], 'info.txt'), 'rb') as f:
    #    camp_info = pickle.load(f)

    with open(f"data/linear_agent/ipinyou-data/{args['camp']}/info.json") as f:
        camp_info = json.load(f)

    with open(args['lin_b0_path'], 'rb') as f:
        b0_file = pickle.load(f)
        lin_b0 = b0_file['b0']
        
    # --- logger
    #tb_logger = SummaryWriter('log/')
    tt_logger = Experiment(name=args['log_name']+'_'+args['camp']+'_test',
                        save_dir = 'log/')
    tt_logger.tag(args)
    tt_logger.save()
    
    
    # --- bidding env option setting
    test_budget            = decision_budget(camp_info['cost_test'],
                                            camp_info['imp_test'],
                                            args['env_budget_ratio'],
                                            camp_info['imp_test'])
    test_num_auction       = camp_info['imp_test']
    #train_epochs            = math.ceil(train_num_auction / args['env_episode_max'])
    test_iteration         = camp_info['imp_test']
    test_dataloadr         = IPinyouDataLoader(data_path=args['data_path'],
                                                market_path=None,
                                                camp=args['camp'],
                                                file='test.theta.txt')
    
    # --- Enviroment
    test_env                = ActionEnv(dataloader=test_dataloadr,
                                        episode_maxlen=camp_info['imp_test'])
    
    # --- Agent setting (DDPG)
    ddpg_agent0              = DDPGAgent(idx= 0,
                                        **to_args_format(args, keyword='ddpg_'),
                                        budget=test_budget,
                                        logger=tt_logger,
                                        device_id=1)
    ddpg_agent1              = DDPGAgent(idx= 1,
                                        **to_args_format(args, keyword='ddpg_'),
                                        budget=test_budget,
                                        logger=tt_logger,
                                        device_id=1)
    
    # --- load agent weight
    print('load model')
    ckpt = torch.load(args['agent0_save_path'])
    ddpg_agent0.actor_target.load_state_dict(ckpt['actor_state_dict'])
    ddpg_agent0.critic_target.load_state_dict(ckpt['critic_state_dict'])
    ddpg_agent0.actor = deepcopy(ddpg_agent0.actor_target)
    ddpg_agent0.critic = deepcopy(ddpg_agent0.critic_target)            
    ddpg_agent0.actor_target.eval()
    ddpg_agent0.critic_target.eval()
    ddpg_agent0.actor.eval()
    ddpg_agent0.critic.eval()
    
    ckpt = torch.load(args['agent1_save_path'])
    ddpg_agent1.actor_target.load_state_dict(ckpt['actor_state_dict'])
    ddpg_agent1.critic_target.load_state_dict(ckpt['critic_state_dict'])
    ddpg_agent1.actor = deepcopy(ddpg_agent1.actor_target)
    ddpg_agent1.critic = deepcopy(ddpg_agent1.critic_target)               
    ddpg_agent1.actor_target.eval()
    ddpg_agent1.critic_target.eval()
    ddpg_agent1.actor.eval()
    ddpg_agent1.critic.eval()
    
    # --- Agent setting (Lin Model)
    linear_agent            = LinearAgent(camp_info=camp_info,
                                        max_bid_price=args['ddpg_max_bid_price'],
                                        budget=test_budget)
    linear_agent.b0 = lin_b0
    
    # --- Agent setting Mcpc Model
    mcpc_agent             = McpcAgent(camp_info=camp_info,
                                max_bid_price=args['ddpg_max_bid_price'],
                                budget=test_budget)
    mcpc_agent.loadCampInfo()
    
    # --- Train
    bid = test_env.reset()
    episode = []
    #ddpg_agent0.is_training = False
    #ddpg_agent1.is_training = False
    for step in range(int(test_iteration)):
        agent0_state0, agent0_action = ddpg_agent0.action(bid['pctr'], 
                                            1-test_env.num_action/test_env.episode_maxlen)
        agent1_state0, agent1_action = ddpg_agent1.action(bid['pctr'], 
                                            1-test_env.num_action/test_env.episode_maxlen)
        
        linear_action   = linear_agent.action(bid['pctr'])
        mcpc_action     = mcpc_agent.action(bid['pctr'])
        episode.append([agent0_action, agent1_action, linear_action, mcpc_action, bid['pctr'], bid['click'], bid['market_price']])
        
        next_bid, reward, terminal, info  = test_env.step(np.array([agent0_action,
                                                                    agent1_action,
                                                                    linear_action,
                                                                    mcpc_action,
                                                                    bid['market_price']]))
        
        # --- calculate reward
        if reward[0] == 1:     # ddpg0 win
            ddpg_agent0.update_result(is_win = True,
                                    click = info['click'],
                                    market_price = info['market_price'],
                                    pctr=info['pctr'])
            if agent1_action != 0.:
                ddpg_agent1.update_result(is_win = False)
            linear_agent.update_result(is_win=False)
            mcpc_agent.update_result(is_win=False)
        elif reward[1] == 1:     # ddpg1 win
            ddpg_agent1.update_result(is_win = True,
                                    click = info['click'],
                                    market_price = info['market_price'],
                                    pctr=info['pctr'])
            if agent0_action != 0.:
                ddpg_agent1.update_result(is_win = False)
            linear_agent.update_result(is_win=False)
            mcpc_agent.update_result(is_win=False)
        elif reward[2] == 1: 
            linear_agent.update_result(is_win=True, 
                                    click = info['click'],
                                    market_price = info['market_price'],
                                    pctr=info['pctr'])
            if agent0_action != 0.:
                ddpg_agent0.update_result(is_win = False)
            if agent1_action != 0.:
                ddpg_agent1.update_result(is_win = False)
            mcpc_agent.update_result(is_win=False)
        elif reward[3] == 1: 
            mcpc_agent.update_result(is_win=True, 
                                    click = info['click'],
                                    market_price = info['market_price'],
                                    pctr=info['pctr'])
            if agent0_action != 0.:
                ddpg_agent0.update_result(is_win = False)
            if agent1_action != 0.:
                ddpg_agent1.update_result(is_win = False)
            linear_agent.update_result(is_win=False)
            
        # --- 중간체크            
        if step % 10000 == 0:
            # --- DDPG 0
            ddpg_agent0_total_click        = ddpg_agent0.num_click
            ddpg_agent0_remained_budget    = ddpg_agent0.remained_budget
            ddpg_agent0_total_win          = ddpg_agent0.num_win
            ddpg_agent0_total_attened      = ddpg_agent0.num_attend_bid
            ddpg_agent0_pctr_list          = ddpg_agent0.list_pctr
            
            # --- DDPG 1
            ddpg_agent1_total_click        = ddpg_agent1.num_click
            ddpg_agent1_remained_budget    = ddpg_agent1.remained_budget
            ddpg_agent1_total_win          = ddpg_agent1.num_win
            ddpg_agent1_total_attened      = ddpg_agent1.num_attend_bid
            ddpg_agent1_pctr_list          = ddpg_agent1.list_pctr
                            
            lin_total_click        = linear_agent.num_click
            lin_remained_budget    = linear_agent.remained_budget
            lin_total_win          = linear_agent.num_win
            lin_total_attened      = linear_agent.num_attend_bid
            lin_pctr_list          = linear_agent.list_pctr
            
            mcpc_total_click        = mcpc_agent.num_click
            mcpc_remained_budget    = mcpc_agent.remained_budget
            mcpc_total_win          = mcpc_agent.num_win
            mcpc_total_attened      = mcpc_agent.num_attend_bid
            mcpc_pctr_list          = mcpc_agent.list_pctr
            
            print('---------------------------')
            print('step : {}'.format(step))
            print('DDPG0 | Win : {} ({:.5f}%), Total Click : {} ({:.5f}%)'.format(
                                ddpg_agent0_total_win, ddpg_agent0_total_win/(ddpg_agent0_total_attened+1e-5),
                                ddpg_agent0_total_click, ddpg_agent0_total_click/(ddpg_agent0_total_attened+1e-5)))
            print('DDPG0 | average pctr : {}, remained budget {}'.format(
                                np.mean(ddpg_agent0_pctr_list), ddpg_agent0_remained_budget))
            
            print('DDPG1 | Win : {} ({:.5f}%), Total Click : {} ({:.5f}%)'.format(
                                ddpg_agent1_total_win, ddpg_agent1_total_win/(ddpg_agent1_total_attened+1e-5),
                                ddpg_agent1_total_click, ddpg_agent1_total_click/(ddpg_agent1_total_attened+1e-5)))
            print('DDPG1 | average pctr : {}, remained budget {}'.format(
                                np.mean(ddpg_agent1_pctr_list), ddpg_agent1_remained_budget))
            
            print('Lin | Win : {} ({:.5f}%), Total Click : {} ({:.5f}%)'.format(
                                lin_total_win, lin_total_win/(lin_total_attened+1e-5),
                                lin_total_click, lin_total_click/(lin_total_attened+1e-5)))
            print('Lin | average pctr : {}, remained budget'.format(
                                np.mean(lin_pctr_list), lin_remained_budget))
            
            print('Mcpc | Win : {} ({:.5f}%), Total Click : {} ({:.5f}%)'.format(
                                mcpc_total_win, mcpc_total_win/(mcpc_total_attened+1e-5),
                                mcpc_total_click, mcpc_total_click/(mcpc_total_attened+1e-5)))
            print('Mcpc | average pctr : {}, remained budget'.format(
                                np.mean(mcpc_pctr_list), mcpc_remained_budget))
            print('---------------------------')
            
            temp_df = pd.DataFrame(np.array(episode))
            if not path.isdir(path.join(tt_logger.save_dir,tt_logger.name, 'version_{}'.format(tt_logger.version), 'log_history_test')):
                os.mkdir(path.join(tt_logger.save_dir, tt_logger.name,'version_{}'.format(tt_logger.version), 'log_history_test'))
            temp_df.to_csv(path.join(tt_logger.save_dir, tt_logger.name,'version_{}'.format(tt_logger.version),'log_history_test', "Ep{}_log.txt".format(test_env.episode_idx)), index=False)
            episode = []

        bid = deepcopy(next_bid)
                
        
    # --- DDPG 0
    ddpg_agent0_total_click        = ddpg_agent0.num_click
    ddpg_agent0_remained_budget    = ddpg_agent0.remained_budget
    ddpg_agent0_total_win          = ddpg_agent0.num_win
    ddpg_agent0_total_attened      = ddpg_agent0.num_attend_bid
    ddpg_agent0_pctr_list          = ddpg_agent0.list_pctr
    
    # --- DDPG 1
    ddpg_agent1_total_click        = ddpg_agent1.num_click
    ddpg_agent1_remained_budget    = ddpg_agent1.remained_budget
    ddpg_agent1_total_win          = ddpg_agent1.num_win
    ddpg_agent1_total_attened      = ddpg_agent1.num_attend_bid
    ddpg_agent1_pctr_list          = ddpg_agent1.list_pctr
                    
    lin_total_click        = linear_agent.num_click
    lin_remained_budget    = linear_agent.remained_budget
    lin_total_win          = linear_agent.num_win
    lin_total_attened      = linear_agent.num_attend_bid
    lin_pctr_list          = linear_agent.list_pctr
    
    mcpc_total_click        = mcpc_agent.num_click
    mcpc_remained_budget    = mcpc_agent.remained_budget
    mcpc_total_win          = mcpc_agent.num_win
    mcpc_total_attened      = mcpc_agent.num_attend_bid
    mcpc_pctr_list          = mcpc_agent.list_pctr
    
    print('---------------------------')
    print('Episode : final')
    print('DDPG0 | Win : {} ({:.5f}%), Total Click : {} ({:.5f}%)'.format(
                        ddpg_agent0_total_win, ddpg_agent0_total_win/(ddpg_agent0_total_attened+1e-5),
                        ddpg_agent0_total_click, ddpg_agent0_total_click/(ddpg_agent0_total_attened+1e-5)))
    print('DDPG0 | average pctr : {}, remained budget {}'.format(
                        np.mean(ddpg_agent0_pctr_list), ddpg_agent0_remained_budget))
    
    print('DDPG1 | Win : {} ({:.5f}%), Total Click : {} ({:.5f}%)'.format(
                        ddpg_agent1_total_win, ddpg_agent1_total_win/(ddpg_agent1_total_attened+1e-5),
                        ddpg_agent1_total_click, ddpg_agent1_total_click/(ddpg_agent1_total_attened+1e-5)))
    print('DDPG1 | average pctr : {}, remained budget {}'.format(
                        np.mean(ddpg_agent1_pctr_list), ddpg_agent1_remained_budget))
    
    print('Lin | Win : {} ({:.5f}%), Total Click : {} ({:.5f}%)'.format(
                            lin_total_win, lin_total_win/(lin_total_attened+1e-5),
                        lin_total_click, lin_total_click/(lin_total_attened+1e-5)))
    print('Lin | average pctr : {}, remained budget'.format(
                        np.mean(lin_pctr_list), lin_remained_budget))
    
    print('Mcpc | Win : {} ({:.5f}%), Total Click : {} ({:.5f}%)'.format(
                        mcpc_total_win, mcpc_total_win/(mcpc_total_attened+1e-5),
                        mcpc_total_click, mcpc_total_click/(mcpc_total_attened+1e-5)))
    print('Mcpc | average pctr : {}, remained budget'.format(
                        np.mean(mcpc_pctr_list), mcpc_remained_budget))
    print('---------------------------')
            

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # --- auction path
    parser.add_argument('--camp',                   type=str,       default='2997')
    parser.add_argument('--data-path',              type=str,       default='data/make-ipinyou-data/')
    parser.add_argument('--seed',                   type=int,       default=777)
    parser.add_argument('--load-model',             type=str,       )
    parser.add_argument('--epoch',                  type=str,       default=2)
    
    # --- environment
    parser.add_argument('--env-episode-max',        type=int,       default=1000)
    parser.add_argument('--env-budget-ratio',       type=float,     default=1/32)
    parser.add_argument('--env_reward_style',       type=str,       default='minus')
    #parser.add_argument('--env_retrun_size',        type=int,       default=1)          # reward계산시 몇번을 더 볼 것인가.
    
    # --- DDPG 
    parser.add_argument('--ddpg-dim-state',         type=int,       default=6)
    parser.add_argument('--ddpg-dim-action',        type=int,       default=1)
    parser.add_argument('--ddpg-actor-optim-lr',    type=float,     default=0.01)
    parser.add_argument('--ddpg-critic-optim-lr',   type=float,     default=0.01)
    parser.add_argument('--ddpg-ou-theta',          type=float,     default=0.15)
    parser.add_argument('--ddpg-ou-mu',             type=float,     default=0.)
    parser.add_argument('--ddpg-ou-sigma',          type=float,     default=0.02)
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
    parser.add_argument('--lin-b0-path',            type=str,       default=f'data/linear_agent/ipinyou-data/{2997}/bid-model/lin-bid_1000_{1/32}_clk_{14205}.pickle')

    # --- test 
    parser.add_argument('--agent0_save_path',       type=str,       default=f'log/minus_{2997}/version_{2}/ddpg0_final_model.pt')
    parser.add_argument('--agent1_save_path',       type=str,       default=f'log/minus_{2997}/version_{2}/ddpg1_final_model.pt')
    
    # --- logger
    parser.add_argument('--log-path',               type=str,      default='log/')
    parser.add_argument('--tb-log-path',            type=str,      default='log/')
    parser.add_argument('--log-name',               type=str,      default='minus')
    
    args = vars(parser.parse_args())
    
    torch.manual_seed(args['seed'])
    np.random.seed(args['seed'])
    random.seed(args['seed'])
    
    test(args)
    
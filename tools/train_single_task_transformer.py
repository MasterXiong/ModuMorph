# python tools/train_single_task_transformer.py --start 0 --end 10
import os
import argparse
import json
import matplotlib.pyplot as plt


def create_data(source_folder, target_folder):
    os.mkdir(target_folder)
    agent_names = [x.split('.')[0] for x in os.listdir(f'{source_folder}/xml')]
    for agent in agent_names:
        os.mkdir(f'{target_folder}/{agent}')
        os.mkdir(f'{target_folder}/{agent}/xml')
        os.mkdir(f'{target_folder}/{agent}/metadata')
        os.system(f'cp {source_folder}/xml/{agent}.xml {target_folder}/{agent}/xml/')
        os.system(f'cp {source_folder}/metadata/{agent}.json {target_folder}/{agent}/metadata/')
        # os.mkdir('log_single_task/%s' %(agent))


def train(agent, output_folder, seed, task, extra_args=''):
    os.system(f'python tools/train_ppo.py --cfg ./configs/{task}.yaml OUT_DIR ./output/{output_folder}/{agent}/{seed} \
        ENV.WALKER_DIR ./unimals_single_task/{agent} PPO.MAX_STATE_ACTION_PAIRS 10000000.0 RNG_SEED {seed} \
        MODEL.TRANSFORMER.EMBEDDING_DROPOUT False \
        {extra_args}')

def compare():
    agent_names = [x.split('.')[0] for x in os.listdir('unimals_100/train/xml')]
    count = 0
    out_dir = 'output/fig/compare_ST_MT_train'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    ST_reward, MT_reward = 0., 0.
    for agent in agent_names:
        if 'Unimal-v0_results.json' in os.listdir('output/log_single_task/%s' %(agent)):
            count += 1
            print (agent)
            
            with open('output/log_single_task/%s/Unimal-v0_results.json' %(agent), 'r') as f:
                result_single = json.load(f)
            with open('log_origin/Unimal-v0_results.json', 'r') as f:
                result_multi = json.load(f)
            ST_reward += result_single[agent]['reward']['reward'][-1]
            MT_reward += result_multi[agent]['reward']['reward'][-1]
            '''
            plt.figure()
            plt.plot([i*2560*32 for i in range(len(result_single[agent]['reward']['reward']))], result_single[agent]['reward']['reward'][:11], label='ST train')
            plt.plot([i*2560*32 for i in range(len(result_multi[agent]['reward']['reward']))], result_multi[agent]['reward']['reward'], label='MT train')
            plt.legend()
            plt.xlabel('Time step')
            plt.ylabel('Returns')
            plt.title(agent)
            plt.savefig(f'{out_dir}/{agent}.png')
            plt.close()
            '''
    print (count, 'ST:', ST_reward / count, 'MT:', MT_reward / count)


def compare_subset(folder):
    agent_names = [x.split('.')[0] for x in os.listdir(f'unimals_100/train_{folder}/xml')]
    out_dir = f'output/fig/compare_ST_MT_{folder}'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    ST_reward, MT_reward = 0., 0.
    for agent in agent_names:
        print (agent)
        with open('output/log_single_task/%s/Unimal-v0_results.json' %(agent), 'r') as f:
            result_single = json.load(f)
        with open(f'output/log_train_{folder}/Unimal-v0_results.json', 'r') as f:
            result_multi = json.load(f)
        ST_reward += result_single[agent]['reward']['reward'][-1]
        MT_reward += result_multi[agent]['reward']['reward'][-1]
        
        plt.figure()
        plt.plot([i*2560*32 for i in range(len(result_single[agent]['reward']['reward']))], result_single[agent]['reward']['reward'], label='ST train')
        plt.plot([i*2560*32 for i in range(len(result_multi[agent]['reward']['reward']))], result_multi[agent]['reward']['reward'], label='MT train')
        plt.legend()
        plt.xlabel('Time step')
        plt.ylabel('Returns')
        plt.title(agent)
        plt.savefig(f'{out_dir}/{agent}.png')
        plt.close()
        
    print ('ST:', ST_reward / 10., 'MT:', MT_reward / 10.)



if __name__ == '__main__':
    
    # python tools/train_single_task_transformer.py --model_type linear --task ft --output_folder MLP_ST_ft_256*3_KL_5 --seed 1409 --start 0 --end 50
    # python tools/train_single_task_transformer.py --model_type transformer --task ft --output_folder TF_ST_ft_KL_5_wo_dropout --seed 1409 --start 0 --end 10
    # python tools/train_single_task_transformer.py --model_type transformer --task csr --output_folder TF_ST_csr_KL_3_wo_PE+dropout --no_PE --seed 1409 --start 0 --end 10
    # for the KL project
    # constant lr
    # python tools/train_single_task_transformer.py --model_type transformer --task incline --output_folder TF_ST_incline_constant_lr_KL_3_wo_dropout --kl 3. --lr 0.0003 --seed 1409 --start 0 --end 5
    # python tools/train_single_task_transformer.py --model_type linear --task ft --output_folder MLP_ST_ft_512*6_constant_lr_KL_5 --kl 5. --lr 0.0003 --seed 1409 --start 0 --end 10
    # constant lr + learned std
    # python tools/train_single_task_transformer.py --model_type linear --task incline --output_folder MLP_ST_incline_1024*2_constant_lr_learnt_std_KL_5 --kl 5. --lr 0.0003 --std learn --seed 1409 --start 0 --end 5
    # python tools/train_single_task_transformer.py --model_type transformer --task ft --output_folder TF_ST_ft_constant_lr_learnt_std_KL_5_wo_dropout --kl 5. --lr 0.0003 --std learn --seed 1409 --start 0 --end 5
    # change model size
    # python tools/train_single_task_transformer.py --model_type transformer --task incline --output_folder TF_ST_incline_2_layer_KL_5_wo_dropout --tf_num_layer 2 --kl 5. --seed 1409 --start 0 --end 5
    # python tools/train_single_task_transformer.py --model_type linear --task incline --output_folder MLP_ST_incline_1024*2_KL_5 --seed 1409 --start 0 --end 10
    # check limb ratio
    # python tools/train_single_task_transformer.py --model_type linear --task ft --output_folder MLP_ST_ft_1024*2_constant_lr_KL_5 --kl 5. --lr 0.0003 --save_limb_ratio --seed 1409 --start 0 --end 10
    # python tools/train_single_task_transformer.py --model_type transformer --task ft --output_folder TF_ST_ft_constant_lr_KL_5_wo_dropout --kl 5. --lr 0.0003 --save_limb_ratio --seed 1409 --start 0 --end 5
    parser = argparse.ArgumentParser(description="Train a RL agent")
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=100, type=int)
    parser.add_argument("--seed", default=1409, type=int)
    parser.add_argument("--output_folder", default='log_test', type=str)
    parser.add_argument("--task", default='ft', type=str)
    parser.add_argument("--model_type", default='transformer', type=str)
    parser.add_argument("--no_PE", action="store_true")
    parser.add_argument("--kl", default=5., type=float)
    parser.add_argument("--tf_num_layer", default=None, type=int)
    parser.add_argument("--lr", default=0.0003, type=float)
    parser.add_argument("--std", default='fixed', type=str)
    parser.add_argument("--save_limb_ratio", action="store_true")
    args = parser.parse_args()
    
    # create_data('unimals_100/test', 'unimals_single_task_test')

    agent_names = [x.split('.')[0] for x in os.listdir('unimals_100/train/xml')]
    agents = agent_names[args.start:args.end]
    print (agents)

    # extra_args = 'MODEL.TRANSFORMER.POS_EMBEDDING None'
    # extra_args = []
    # if 'embed' in args.output_folder:
    #     extra_args.append('MODEL.TRANSFORMER.PER_NODE_EMBED True')
    # if 'decoder' in args.output_folder:
    #     extra_args.append('MODEL.TRANSFORMER.PER_NODE_DECODER True')
    # # if 'wo_PE+dropout' in args.output_folder:
    # #     extra_args.append('MODEL.TRANSFORMER.EMBEDDING_DROPOUT False')
    # extra_args.append('MODEL.TRANSFORMER.EMBEDDING_DROPOUT False')
    # extra_args.append('MODEL.TRANSFORMER.POS_EMBEDDING None')
    # extra_args.append('PPO.KL_TARGET_COEF 5.')
    # extra_args = ' '.join(extra_args)
    # print (extra_args)

    extra_args = []
    params = args.output_folder.split('_')
    # params for MLP
    for param in params:
        if '*' in param:
            hidden_dim = eval(param.split('*')[0])
            layer_num = eval(param.split('*')[1])
            extra_args.append(f'MODEL.MLP.HIDDEN_DIM {hidden_dim}')
            extra_args.append(f'MODEL.MLP.LAYER_NUM {layer_num}')
            break
    # change learning rate scheme
    for param in params:
        if param == 'constant':
            extra_args.append('PPO.LR_POLICY constant')
            break
    extra_args.append(f'PPO.BASE_LR {args.lr}')
    if args.std == 'learn':
        extra_args.append('MODEL.ACTION_STD_FIXED False')
    if args.save_limb_ratio:
        extra_args.append('SAVE_LIMB_RATIO True')
    # if args.task in ['ft', 'incline']:
    #     extra_args.append('PPO.KL_TARGET_COEF 5.')
    # else:
    #     extra_args.append('PPO.KL_TARGET_COEF 3.')
    extra_args.append(f'PPO.KL_TARGET_COEF {args.kl}')
    if args.model_type == 'linear':
        extra_args.append('MODEL.TYPE mlp')
    if args.no_PE:
        extra_args.append('MODEL.TRANSFORMER.POS_EMBEDDING None')
    if args.tf_num_layer is not None:
        extra_args.append(f'MODEL.TRANSFORMER.NLAYERS {args.tf_num_layer}')
    extra_args = ' '.join(extra_args)
    print (extra_args)

    for agent in agents:
        if os.path.exists(f'output/{args.output_folder}/{agent}/{args.seed}/checkpoint_-1.pt'):
            print (f'already finish {agent} {args.seed}')
            continue
        train(agent, args.output_folder, args.seed, args.task, extra_args)
        # os.system(f'cp unimals_100/train/xml/{agent}.xml unimals_20/xml/')
        # os.system(f'cp unimals_100/train/metadata/{agent}.json unimals_20/metadata/')
    
    #compare()
    '''
    test_idx = [1,2,53,74,90]
    test_idx = [1,2,90]
    for idx in test_idx:
        agent = agent_names[idx]
        with open('log_origin/Unimal-v0_results.json', 'r') as f:
	        result_MT = json.load(f)
        #with open(f'log_single_task/{agent}/Unimal-v0_results.json', 'r') as f:
	    #    result_ST = json.load(f)
        reward_MT = result_MT[agent]['reward']['reward'][-1]
        #reward_ST = result_ST[agent]['reward']['reward'][-1]
        print (agent, 'MT', reward_MT)
    '''
    '''
    # create a subset of training morphologies for a fair comparison with ST training
    agent_names = [x.split('.')[0] for x in os.listdir('unimals_100/train/xml')]
    subset_agents = []
    for agent in agent_names:
        if 'Unimal-v0_results.json' in os.listdir('output/log_single_task/%s' %(agent)):
            subset_agents.append(agent)
    subset_agents = subset_agents[2::3][:10]
    os.mkdir('unimals_100/train_subset_2')
    os.mkdir('unimals_100/train_subset_2/xml')
    os.mkdir('unimals_100/train_subset_2/metadata')
    for agent in subset_agents:
        os.system(f'cp unimals_100/train/xml/{agent}.xml unimals_100/train_subset_2/xml/')
        os.system(f'cp unimals_100/train/metadata/{agent}.json unimals_100/train_subset_2/metadata/')
    '''
    #compare_subset('subset_2')

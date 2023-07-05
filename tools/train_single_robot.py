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


def train(agent, output_folder, seed, task, extra_args=''):
    os.system(f'python tools/train_ppo.py --cfg ./configs/{task}.yaml OUT_DIR ./output/{output_folder}/{agent}/{seed} \
        ENV.WALKER_DIR ./unimals_single_task/{agent} PPO.MAX_STATE_ACTION_PAIRS 10000000.0 RNG_SEED {seed} \
        MODEL.TRANSFORMER.EMBEDDING_DROPOUT False \
        {extra_args}')



if __name__ == '__main__':
    
    # python tools/train_single_task_transformer.py --model_type linear --task ft --output_folder MLP_ST_ft_256*3_KL_5 --seed 1409 --start 0 --end 50
    # python tools/train_single_task_transformer.py --model_type transformer --task ft --output_folder TF_ST_ft_KL_5_wo_dropout --seed 1409 --start 0 --end 10
    # python tools/train_single_task_transformer.py --model_type transformer --task csr --output_folder TF_ST_csr_KL_3_wo_PE+dropout --no_PE --seed 1409 --start 0 --end 10
    parser = argparse.ArgumentParser(description="Train a RL agent on a single robot")
    # By default, we have 100 training robots. We can specify the range of the agents to train on with `start` and `end`
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
    args = parser.parse_args()
    
    # create_data('unimals_100/train', 'unimals_single_task_train')

    agent_names = [x.split('.')[0] for x in os.listdir('unimals_100/train/xml')]
    agents = agent_names[args.start:args.end]
    print (agents)

    extra_args = []
    params = args.output_folder.split('_')
    # params for MLP
    # a hack here: the output folder name should include hidden_dim*layer_num
    # E.g., include "256*3" in the folder name to specify 3-layer MLP with hidden size 256
    for param in params:
        if '*' in param:
            hidden_dim = eval(param.split('*')[0])
            layer_num = eval(param.split('*')[1])
            extra_args.append(f'MODEL.MLP.HIDDEN_DIM {hidden_dim}')
            extra_args.append(f'MODEL.MLP.LAYER_NUM {layer_num}')
            break
    # change learning rate scheme
    # If you want to use a constant learning rate, specify "constant" in the output folder name
    for param in params:
        if param == 'constant':
            extra_args.append('PPO.LR_POLICY constant')
            break
    extra_args.append(f'PPO.BASE_LR {args.lr}')
    # whether to use a fixed or learned action std (fixed by default)
    if args.std == 'learn':
        extra_args.append('MODEL.ACTION_STD_FIXED False')
    # KL threshold to early stop a PPO iteration
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

import argparse

parser = argparse.ArgumentParser(description='MutliModel Time-Series Anomaly Detection')
parser.add_argument("--random_seed", default=42,
                    type=int, help='the random seed')

# training setting
parser.add_argument("--gpu", default=True, type=lambda x: x.lower() == "true")
parser.add_argument("--epochs", default=300, type=int,
                    help='the number of training epochs')
parser.add_argument("--patience", default=15, type=float,
                    help='the number of epoch that loss is uping')
parser.add_argument("--learning_rate", default=1e-3,
                    type=float, help='the data number at one epoch')
parser.add_argument("--weight_decay", default=5e-4, type=float,
                    help='the one of optimzier parameters which prevent overfitting ')
parser.add_argument("--learning_change", default=100, type=int,
                    help='the epoch number that change learning rate')
parser.add_argument("--learning_gamma", default=0.9, type=float,
                    help='the weight that change learning rate')
parser.add_argument("--label_weight", default=1e-2, type=float,
                    help='the unkown weight in reconstruction loss')
parser.add_argument("--label_percent", default=0.5, type=float,
                    help='the proportion of labeled data') 
parser.add_argument("--abnormal_weight", default=96, type=int,
                    help='the abnormal weight in classfication loss')
parser.add_argument("--rec_down", default=1, type=int,
                    help='the number that changes reconstruction loss weight')
parser.add_argument("--para_low", default=1e-2, type=float,
                    help='the min weight of rec loss')

# model setting
parser.add_argument("--feature_node", default=4, type=int,
                    help='the pod kpi data number at one epoch')
parser.add_argument("--feature_edge", default=4, type=int,
                    help='the edge data number at one epoch')
parser.add_argument("--feature_log", default=16, type=int,
                    help='the log data number at one epoch')
parser.add_argument("--raw_node", default=3, type=int,
                    help='the raw pod kpi data number at one epoch')
parser.add_argument("--raw_edge", default=7, type=int,
                    help='the raw edge kpi data number at one epoch')
parser.add_argument("--log_len", default=256, type=int,
                    help='the log template amount')
parser.add_argument("--num_heads_edge", default=4, type=int,
                    help='the number of multiattention heads about trace')
parser.add_argument("--num_heads_node", default=4, type=int,
                    help='the number of multiattention heads about metric')
parser.add_argument("--num_heads_log", default=4, type=int,
                    help='the number of multiattention heads about log')
parser.add_argument("--num_heads_n2e", default=4, type=int,
                    help='the number of multiattention heads about node')
parser.add_argument("--num_heads_e2n", default=2, type=int,
                    help='the number of multiattention heads about edge')
parser.add_argument("--num_layer", default=2, type=int,
                    help='the number of model layers')
parser.add_argument("--dropout", default=0.2, type=float)


# dataset setting
parser.add_argument("--batch_size", default=50, type=int,
                    help='the data number at one epoch')
parser.add_argument("--window", default=10, type=int,
                    help='size of sliding window')
parser.add_argument("--step", default=1, type=int,
                    help='sliding window stride')
parser.add_argument("--num_nodes", default=5, type=int,
                    help='the number of node in graph')

# path setting
parser.add_argument("--data_path", default='./data/MSDS-pre',
                    type=str, help='the path of raw data')
parser.add_argument("--dataset_path", default="./data/MSDS-save",
                    type=str, help='the path of saving data')
parser.add_argument("--result_dir", default="./result",
                    type=str, help='the path of result and log')

parser.add_argument("--main_model", "--config", default='mstgad', type=str,
                    dest='main_model',
                    choices=['mstgad', 'mstgad_mamba', 'mstgad_ad', 'mestgad'],
                    help='model config to run')
parser.add_argument("--evaluate", default=False,
                    type=lambda x: x.lower() == "true", help='Evaluate the exist model')
parser.add_argument("--model_path", default=None,
                    type=str, help='the path of existing model checkpoint')

# Sbatch-friendly aliases
parser.add_argument("--seed", default=None, type=int, dest='seed_override',
                    help='random seed (overrides --random_seed)')
parser.add_argument("--output-dir", default=None, type=str, dest='output_dir_override',
                    help='output directory (overrides --result_dir)')
parser.add_argument("--dataset", default=None, type=str, dest='dataset',
                    choices=['msds', 'aiops'],
                    help='dataset shorthand (sets data_path and dataset_path)')

# MESTGAD / Mamba hyperparameters
parser.add_argument("--lambda-ad", "--lambda_ad", default=0.1, type=float, dest='lambda_ad',
                    help='association discrepancy loss weight (MESTGAD only)')
parser.add_argument("--d-state", default=16, type=int, dest='d_state',
                    help='Mamba SSM hidden state dimension')
parser.add_argument("--d-conv", default=4, type=int, dest='d_conv',
                    help='Mamba local conv kernel width')
parser.add_argument("--expand", default=2, type=int,
                    help='Mamba expansion factor')

args = vars(parser.parse_args())

# Apply seed override
if args['seed_override'] is not None:
    args['random_seed'] = args['seed_override']

# Apply output-dir override
if args['output_dir_override'] is not None:
    args['result_dir'] = args['output_dir_override']

# Apply dataset shorthand
if args['dataset'] == 'msds':
    args['data_path'] = './data/MSDS-pre'
    args['dataset_path'] = './data/MSDS-save'
elif args['dataset'] == 'aiops':
    args['data_path'] = './data/AIOps-pre'
    args['dataset_path'] = './data/AIOps-save'
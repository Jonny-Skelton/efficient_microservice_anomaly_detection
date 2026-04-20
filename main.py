import util.util as util
import util.train as train
import util.data_MSDS as data_loads
from util.parser_MSDS import *

from torch.utils.data import DataLoader
import warnings
import logging
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.append('/code')
warnings.filterwarnings("ignore")

util.seed_everything(args['random_seed'])

if __name__ == '__main__':
    if args['evaluate']:
        dict_json = util.read_params(args)
        for key in dict_json.keys():
            args[key] =  args[key] if key in ['model_path','evaluate', 'result_dir', 'data_path', 'dataset_path'] else dict_json[key]
        args['result_dir'] = args['model_path']
    else:
        args['hash_id'], args['result_dir'] = util.dump_params(args)
        util.json_pretty_dump(args, os.path.join(args['result_dir'], "params.json"))
        args['model_path'] = args['result_dir']

    logging.info("---- Model: ----" + args['main_model'] +"-" + args['hash_id'] + "----" + f"train : {not args['evaluate']}"\
        + "----" + f"evaluate : {args['evaluate']}")

    # dealing & loading data
    processed = data_loads.Process(**args)
    train_dl = DataLoader(processed.dataset[:int(len(processed.dataset)*0.7)],
                          batch_size=args['batch_size'],
                          shuffle=True, pin_memory=False, drop_last=True)
    test_dl = DataLoader(processed.dataset[int(len(processed.dataset)*0.7):],
                        batch_size=args['batch_size'],
                        shuffle=False, pin_memory=False, drop_last=True)

    # declare model and train
    import src.model as model
    import src.MESTGAD as e_model

    if args['main_model'] in ('mestgad', 'mstgad_mamba', 'mstgad_ad'):
        active_model = e_model.MESTGADModel(processed.graph, **args)
    else:
        active_model = model.MyModel(processed.graph, **args)

    trainer = train.MY(active_model, **args)

    #Training
    if not args['evaluate']:
        trainer.fit(train_loader=train_dl, test_loader=test_dl)


    # Evaluating
    logging.info('calculate scores...')
    eval_infos = {}
    with open('./result.log', 'a+') as file:
        file.writelines(f"\n {args['main_model']}-{args['hash_id']} --weight_decay:{args['weight_decay']}   --learning_change:{args['learning_change']} \n")
        for statue in ['loss', 'f1']:
            logging.info(f'calculate label with {statue}...')
            trainer.load_model(args['model_path'], name=statue)
            info = trainer.evaluate(test_dl, isFinall=True)
            file.writelines(statue + '   ' + info + '\n')
            eval_infos[statue] = info
    logging.info("^^^^^^ Current Model: ----" + args['main_model'] + "-" * 4 + args['hash_id'] + " ^^^^^")

    try:
        import json as _json
        import re as _re
        def _parse_eval_info(s):
            m = _re.search(
                r'pr:([\d.]+)\s+rc:([\d.]+)\s+auc:([\d.]+)\s+ap:([\d.]+)'
                r'\s+f1:\s*([\d.]+)\s+pred_right:\s*(\d+)\s+pred_wrong:\s*(\d+)'
                r'\s+actu_right:\s*(\d+)\s+actu_wrong:\s*(\d+)', s)
            if not m:
                return None
            g = m.groups()
            return {'pr': float(g[0]), 'rc': float(g[1]), 'auc': float(g[2]),
                    'ap': float(g[3]), 'f1': float(g[4]),
                    'pred_right': int(g[5]), 'pred_wrong': int(g[6]),
                    'actu_right': int(g[7]), 'actu_wrong': int(g[8])}
        metrics = {
            'hash_id': args['hash_id'],
            'loss_eval': _parse_eval_info(eval_infos.get('loss', '')),
            'f1_eval':   _parse_eval_info(eval_infos.get('f1', '')),
        }
        metrics_path = os.path.join(args['result_dir'], 'metrics.json')
        with open(metrics_path, 'w') as _mf:
            _json.dump(metrics, _mf, indent=2)
    except Exception as _e:
        logging.warning(f"metrics.json write failed: {_e}")


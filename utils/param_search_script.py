from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from learning_process import ExperimentResult

import subprocess
import os
import signal
import sys
import argparse
import pickle


### Parameters search script arguments
def str_to_bool(s):
    # Определение функции для преобразования строки в булево значение
    if s.lower() in ('yes', 'true', 't', 'y', '1', 'True', 'Yes'):
        return True
    elif s.lower() in ('no', 'false', 'f', 'n', '0', 'No', 'False'):
        return False
    else:
        raise ValueError("Неверное представление для булевого значения: {0}".format(s))


argp = argparse.ArgumentParser(
    "param_search_script.py",
    description="Search optimal parameters with Baesian Optimization"
)

argp.add_argument(
    "--rewrite_trials", type=str_to_bool,
    help='Rewrite trials.pickle in same path if true',
    default=False
)

argp.add_argument(
    "--load_trials", type=str_to_bool,
    help="Load trials from path", 
    default=False
)

argp.add_argument(
    "--trials_path", type=str,
    help='Path to save trials object', 
    default='.'
)

argp.add_argument(
    "--trials_name", type=str,
    help='Name of trials file',
    default='trials'
)

argp.add_argument(
    "--n_epochs", type=int,
    help='Epochs per training',
    default=50
)

argp.add_argument(
    "--max_evals", type=int,
    help='Hyperopt evals count',
    default=20
)

argp.add_argument(
    "--terminal", type=str,
    help='For example: konsole, xterm and etc. Note, -e option must be supported',
    default='konsole'
)

args = argp.parse_args()
###


def save_trials_file(trials, path, filename):
    with open(f"{path}/{filename}.pickle", "wb") as f:
        pickle.dump(trials, f)

def load_trials(path, name):
    with open(f"{path}/{name}.pickle", 'rb') as f:
        res = pickle.load(f)
    return res


def signal_handler(sig, frame):
    print("Received Ctrl+C - closing all processes")
    # Всем дочерним процессам закрыться

    if not args.rewrite_trials:
        sys.exit(0)
    # Сохраняем trials, если он есть
    try:
        global trials
        save_trials_file(trials, args.trials_path, args.trials_name)
    except Exception as e:
        print(f"Ошибка сохранения файла trials. Исключение: {e}")

    os.killpg(0, signal.SIGTERM)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

venv_path = 'venv'

space = {
    'learning_rate': hp.uniform('learning_rate', 1e-6, 1e-4),
    'initial_period': hp.uniform('initial_period', 0.1, 5),
    'min_lr': hp.loguniform('min_lr', -3, -1),
    'period_increase_mult': hp.uniform('period_inicrease_mult', 1., 2.),
    'patch_size': hp.choice('patch_size', [2, 4, 8]),
    'embedding_size': hp.choice('embedding_size', [64, 128, 256, 512]),
    'batch_size': hp.choice('batch_size', [16, 32, 64, 128, 256])

}

def extract_accuracy():
    exp_res = ExperimentResult(*[None for i in range(6)])
    exp_res.load()
    if exp_res.train_acc == -1:
        return -1
    return max(exp_res.train_acc)

# Обучение модели как целевая функция
def objective(params):
    command = f"{venv_path}/bin/python train_script.py --lr {params['learning_rate']} " \
            + f"--initial_period {params['initial_period']} --min_lr {params['min_lr']} " \
            + f"--period_increase_mult {params['period_increase_mult']} --patch_size {params['patch_size']} " \
            + f"--embedding_size {params['embedding_size']} --n_epochs {args.n_epochs} --batch_size {params['batch_size']}"

    subprocess.call(f'{args.terminal} -e bash -c "{command}"', shell=True)
    acc = extract_accuracy()
    return {'loss': -acc, 'status': STATUS_OK}

# Запускаем процесс оптимизации

if args.load_trials:
    trials = load_trials(args.trials_path, args.trials_name)
else:
    trials = Trials()

best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=args.max_evals,
    trials=trials
)
save_trials_file(trials, args.trials_path, args.trials_name)

print(best)

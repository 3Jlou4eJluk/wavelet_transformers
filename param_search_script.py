from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from learning_process import ExperimentResult
import subprocess
import os
import signal
import sys

def signal_handler(sig, frame):
    print("Received Ctrl+C - closing all processes")
    # Всем дочерним процессам закрыться
    os.killpg(0, signal.SIGTERM)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

venv_path = 'venv'

# Пример пространства поиска
space = {
    'learning_rate': hp.uniform('learning_rate', 0.001, 0.1),
    'num_leaves': hp.choice('num_leaves', range(20, 100)),
    'max_depth': hp.choice('max_depth', range(2, 30)),
    # Можно добавлять и другие гиперпараметры
}


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
    subprocess.call(
        [
            f'{venv_path}/bin/python', 'train_script.py', 
            '--lr', str(params['learning_rate']), 
            '--initial_period', str(params['initial_period']),
            '--min_lr', str(params['min_lr']),
            '--period_increase_mult', str(params['period_increase_mult']),
            '--patch_size', str(params['patch_size']),
            '--embedding_size', str(params['embedding_size']),
            '--n_epochs', str(1),
            '--batch_size', str(params['batch_size'])
        ]
    )
    acc = extract_accuracy()
    return {'loss': -acc, 'status': STATUS_OK}

# Запускаем процесс оптимизации
trials = Trials()
best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=2,
    trials=trials
)

print(best)

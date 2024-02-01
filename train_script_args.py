import argparse

argparser = argparse.ArgumentParser(
    prog='train_script.py',
    description='Program Training Neural Networks with '\
               +'specified parameters'
)


# Main Args

argparser.add_argument(
    '--n_epochs', type=int,
    help='Train epochs count',
    required=True
)

argparser.add_argument(
    '--lr', type=float,
    help='Initial Learning Rate',
    required=True
)

argparser.add_argument(
    '--batch_size', type=int,
    help='Batch size for training',
    required=True
)



# Scheduler Args
argparser.add_argument(
    '--initial_period', type=float,
    help='Initial steps count for restart',
    required=True
)

argparser.add_argument(
    '--min_lr', type=float,
    help='Minimum learning rate for Cosine Annealing Warm Restarts',
    required=True
)

argparser.add_argument(
    '--period_increase_mult', type=float,
    help='In proportion, taken from steps count',
    required=True
)


# Image partition arguments
argparser.add_argument(
    '--patch_size', type=int,
    help='Patch size, should be divisible by 2',
    required=True
)


# Transformer arguments
argparser.add_argument(
    '--embedding_size', type=int,
    help='Embedding size of TransformerEncoder',
    required=True
)
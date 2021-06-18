import argparse,os
from solver import Solver
from dataloader import get_loader
from torch.utils.data import DataLoader

def main(config):

    # Dataset & DataLoader
    train_loader = get_loader(config, 'train')
    val_loader = get_loader(config, 'val')
    test_loader = get_loader(config, 'test')

    # Solver
    solver = Solver(config, train_loader, val_loader, test_loader)

    solver.plot_labels(['train','val','test'])

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # training hyper-parameters
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=72)
    parser.add_argument('--lr', type=float, default=1e-4)

    # dataset & loader config
    parser.add_argument('--image_pool', type=str, nargs='+', default=['12'])
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--input_type', type=str, default='uvl', choices=['rgb','uvl'])
    parser.add_argument('--output_type', type=str, default=None, choices=['illumination','uv','mixmap'])
    parser.add_argument('--mask_black', type=str, default=None)
    parser.add_argument('--mask_highlight', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=10)

    # path config
    parser.add_argument('--data_root', type=str, default='GALAXY_synthetic')
    parser.add_argument('--model_root', type=str, default='models')
    parser.add_argument('--result_root', type=str, default='results')
    parser.add_argument('--log_root', type=str, default='logs')
    parser.add_argument('--checkpoint', type=str, default='210520_0600')

    # Misc
    parser.add_argument('--save_epoch', type=int, default=-1,
                        help='number of epoch for auto saving, -1 for turn off')
    parser.add_argument('--multi_gpu', type=int, default=1, choices=[0,1],
                        help='0 for single-GPU, 1 for multi-GPU')

    config = parser.parse_args()
    main(config)

def print_config(config):
	print("="*20, "Configuration", "="*20)
	print(config)
	print("="*55)
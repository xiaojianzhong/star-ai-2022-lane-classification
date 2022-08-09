import argparse
import inspect
import logging
import os
import random
import shutil
import warnings
from datetime import datetime

import numpy as np
import torch
import torchinfo
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from configs import CFG
from criterions import build_criterion
from datas import build_dataset, build_dataloader
from metrics import Metric
from models import build_model
from optimizers import build_optimizer
from schedulers import build_scheduler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config',
                        type=str,
                        help='config file')
    parser.add_argument('--checkpoint',
                        type=str,
                        help='checkpoint file')
    parser.add_argument('--comment',
                        type=str,
                        help='comment')
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help='random seed')
    parser.add_argument('--configs',
                        nargs='+',
                        default=[],
                        help='custom configs')
    args = parser.parse_args()
    return args


def main():
    # parse command line arguments
    args = parse_args()

    # log to stdout
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s: %(message)s',
        handlers=[
            logging.StreamHandler(),
        ])

    # create experiment output path if not exists
    experiment_name = datetime.now().strftime('%Y%m%d-%H%M%S_') + os.path.splitext(os.path.basename(args.config))[0]
    if args.comment is not None:
        experiment_name = experiment_name + '_' + args.comment.replace(' ', '-')
    logging.info(f'experiment name: {experiment_name}')
    path = os.path.join('runs', experiment_name)
    os.makedirs(path, exist_ok=True)

    # ignore warnings
    warnings.filterwarnings('ignore')

    # set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

    # initialize TensorBoard summary writer
    writer = SummaryWriter(log_dir=path)

    # merge config with config file
    CFG.merge_from_file(args.config)
    # merge config with custom configs from command line arguments
    CFG.merge_from_list(args.configs)
    # dump config
    with open(os.path.join(path, 'config.yaml'), 'w') as f:
        f.write(CFG.dump())

    # build dataset
    train_dataset = build_dataset('train')
    # build dataloader
    train_dataloader = build_dataloader(train_dataset, 'train')
    # build model
    model = build_model()
    model = DataParallel(model)
    model.cuda()
    # dump model
    shutil.copyfile(inspect.getfile(model.module.__class__), os.path.join(path, 'model.py'))
    with open(os.path.join(path, 'model.txt'), 'w') as f:
        f.write(str(torchinfo.summary(model.module, input_size=(1, *train_dataset[0]['x'].shape), verbose=0)))
    # build criterion
    criterion = build_criterion()
    criterion.cuda()
    # build optimizer
    optimizer = build_optimizer(model.parameters())
    # build scheduler
    scheduler = build_scheduler(optimizer)
    # build metric
    metric = Metric()

    epoch = 0
    iteration = 0
    best_f1 = 0.

    # load checkpoint if specified
    if args.checkpoint is not None:
        if not os.path.isfile(args.checkpoint):
            raise RuntimeError('checkpoint {} not found'.format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint)
        model.module.load_state_dict(checkpoint['model']['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer']['state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler']['state_dict'])
        epoch = checkpoint['optimizer']['epoch']
        iteration = checkpoint['optimizer']['iteration']
        best_f1 = checkpoint['metric']['f1']
        logging.info('load checkpoint {} with f1={:.4f}'.format(args.checkpoint, best_f1))

    # train - validation loop
    while True:
        epoch += 1
        if epoch > CFG.EPOCHS:
            writer.close()
            return

        lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('lr-epoch', lr, epoch)

        # train
        model.train()
        metric.reset()  # reset metric
        train_bar = tqdm(train_dataloader, desc='train', ascii=True)
        for sample in train_bar:
            iteration += 1
            if iteration > CFG.ITERATIONS:
                writer.close()
                return

            lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('lr-iteration', lr, iteration)

            x, gt = sample['x'], sample['gt']
            x, gt = x.cuda(), gt.cuda()
            y = model(x)
            # pred = (torch.sigmoid(y) > 0.5)
            pred = torch.argmax(y, dim=1)

            loss = criterion(y, gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metric.add(loss.item(), pred.data.cpu().numpy(), gt.data.cpu().numpy())

            writer.add_scalar('train/loss-iteration', loss.item(), iteration)
            writer.add_scalar('train/precision-iteration', metric.precision(), iteration)
            writer.add_scalar('train/recall-iteration', metric.recall(), iteration)
            writer.add_scalar('train/f1-iteration', metric.f1(), iteration)
            train_bar.set_postfix({
                'lr': f'{lr:.4f}',
                'loss': f'{metric.loss():.4f}',
                'precision': f'{metric.precision():.2f}',
                'recall': f'{metric.recall():.2f}',
                'f1': f'{metric.f1():.2f}',
            })

            if CFG.SCHEDULER.BY_ITERATION:
                scheduler.step()

        writer.add_scalar('train/loss-epoch', metric.loss(), epoch)
        writer.add_scalar('train/precision-epoch', metric.precision(), epoch)
        writer.add_scalar('train/recall-epoch', metric.recall(), epoch)
        writer.add_scalar('train/f1-epoch', metric.f1(), epoch)
        writer.add_scalar('train/seconds-epoch', metric.seconds(), epoch)

        if CFG.SCHEDULER.BY_EPOCH:
            scheduler.step()

        # save checkpoint
        checkpoint = {
            'model': {
                'state_dict': model.module.state_dict(),
            },
            'optimizer': {
                'state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'iteration': iteration,
            },
            'scheduler': {
                'state_dict': scheduler.state_dict(),
            },
            'metric': {
                'loss': metric.loss(),
                'precision': metric.precision(),
                'recall': metric.recall(),
                'f1': metric.f1(),
            },
        }
        torch.save(checkpoint, os.path.join(path, 'last.pth'))
        torch.save(checkpoint, os.path.join(path, f'{epoch}.pth'))
        if checkpoint['metric']['f1'] > best_f1:
            torch.save(checkpoint, os.path.join(path, 'best.pth'))


if __name__ == '__main__':
    main()

import argparse
import datetime
import logging
import os

import torch
import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm

from dataset import load_data
from models.modeling import DSTC
from utils import set_seed, gen_confusion_matrix, eval_metrics


def train(model, optimizer, criterion, train_loader):
    model.train()
    confusion_matrix = torch.zeros([NUM_CLASSES, NUM_CLASSES]).cuda()

    for data, label, name in train_loader:
        data, label = data.cuda(), label.long().cuda()
        optimizer.zero_grad()

        pred, train_label, spix_map = model(data, label)
        loss = criterion(pred.permute(0, 2, 1), train_label)
        loss.backward()
        optimizer.step()

        for i in range(pred.shape[0]):
            j = torch.arange(pred.shape[1]).cuda()
            mask = (spix_map[i].unsqueeze(-1) == j).float().cuda()

            sal_result = (mask @ pred[i]).argmax(dim=-1)
            confusion_matrix_tmp = gen_confusion_matrix(NUM_CLASSES, sal_result, label[i])
            confusion_matrix += confusion_matrix_tmp

    confusion_matrix = confusion_matrix.cpu().detach().numpy()
    return eval_metrics(confusion_matrix, mode='tr')


def valid(model, val_loader):
    with torch.no_grad():
        model.eval()
        confusionmat = torch.zeros([NUM_CLASSES, NUM_CLASSES]).cuda()

        for data, label, name in val_loader:
            data, label = data.cuda(), label.long().cuda()
            pred, _, spix_map = model(data, label)

            for i in range(pred.shape[0]):
                j = torch.arange(pred.shape[1]).cuda()
                mask = (spix_map[i].unsqueeze(-1) == j).float().cuda()

                sal_result = (mask @ pred[i]).argmax(dim=-1)
                confusionmat_tmp = gen_confusion_matrix(NUM_CLASSES, sal_result, label[i])
                confusionmat = confusionmat + confusionmat_tmp

        confusionmat = confusionmat.cpu().detach().numpy()
        return eval_metrics(confusionmat, mode='val')


def main():
    min_f1 = 0

    model = DSTC(cfg)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.cuda()

    spix_nums = (
            cfg['cluster']['proposal'] * cfg['cluster']['proposal'] *
            cfg['cluster']['fold'] * cfg['cluster']['fold']
    )
    print(f"number of superpixels: {spix_nums}")
    print(f"pixels per superpixels: {int(cfg['img_size'] * cfg['img_size'] / spix_nums)}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0001)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch_num, eta_min=0.1 * args.lr)

    train_loader = load_data(args, 'tr')
    val_loader = load_data(args, 'val')

    with tqdm(total=args.epoch_num) as pbar:
        for epoch in range(args.epoch_num):
            train_f1 = train(model, optimizer, criterion, train_loader)
            val_f1 = valid(model, val_loader)
            if val_f1 > min_f1:
                min_f1 = val_f1
                torch.save(model.state_dict(), f"DataStorage/{args.exp_name}/best_model.pth")
            scheduler.step()
            sw.add_scalar('f1/train', train_f1, epoch)
            sw.add_scalar('f1/val', val_f1, epoch)
            sw.add_scalar('lr', optimizer.state_dict()['param_groups'][0]['lr'], epoch)
            pbar.update(1)
            logger.info('Epoch: %d, train F1: %.4f, val F1: %.4f' % (epoch, train_f1, val_f1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch WHU_OHS')
    parser.add_argument('--config', default='models/yamls/resnet.yaml', type=str,
                        help='config file path (default: config.yaml)')
    parser.add_argument('--exp_name', type=str, help='exp name', required=True)
    # Dataset
    parser.add_argument('--data_root', default='', type=str, help='data root')
    # Training
    parser.add_argument('--batch_size', default=8, type=int, help='mini-batch size (default: 4)')
    parser.add_argument('--epoch_num', default=100, type=int, help='epoch number (default: 200)')
    parser.add_argument('--lr', default=5e-4, type=float, help='initial learning rate (default: 2e-4)')
    args = parser.parse_args()
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

    os.makedirs(f"DataStorage/{args.exp_name}/valid", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    NUM_CLASSES = cfg['num_classes']
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    set_seed(233)

    if args.exp_name == '':
        log_filename = f'logs/{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.log'
    else:
        log_filename = f'logs/{args.exp_name}.log'
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
    logger = logging.getLogger('training_logger')

    logger.info(cfg)
    logger.info(args)
    sw = SummaryWriter(log_dir=f'runs/{args.exp_name}')
    main()

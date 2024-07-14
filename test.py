import argparse
import logging
import os
import time

import cv2
import torch
import yaml
from tqdm import tqdm

from dataset import load_data
from models.modeling import DSTC
from utils import gen_confusion_matrix, eval_metrics, map_to_color, set_seed


def test(model, test_loader, device, save_path, num_classes, batch_size):
    with torch.no_grad():
        model.eval()
        confusion_matrix = torch.zeros([num_classes, num_classes], device=device)
        total_time = []

        for data, label, name in tqdm(test_loader):
            data, label = data.to(device), label.long().to(device)

            start_time = time.time()
            pred, _, spix_map = model(data, label)
            end_time = time.time()
            total_time.append(end_time - start_time)

            for i in range(pred.shape[0]):
                j = torch.arange(pred.shape[1]).cuda()
                mask = (spix_map[i].unsqueeze(-1) == j).float().cuda()

                sal_result = (mask @ pred[i]).argmax(dim=-1)
                sal_result_img = map_to_color(sal_result.cpu().numpy(), label[i].cpu().numpy())
                cv2.imwrite(os.path.join(save_path, f"{name[i]}.jpg"), sal_result_img)

                sal_gt = map_to_color(label[i].cpu().numpy())
                cv2.imwrite(os.path.join(save_path, f"{name[i]}_gt.jpg"), sal_gt)

                confusion_matrix_tmp = gen_confusion_matrix(num_classes, sal_result, label[i])
                confusion_matrix += confusion_matrix_tmp

        avg_time = sum(total_time) / len(total_time)
        print("Average time: ", avg_time)
        print("FPS: ", batch_size / avg_time)

        confusion_matrix = confusion_matrix.cpu().detach().numpy()
        return confusion_matrix


def main():
    parser = argparse.ArgumentParser(description='PyTorch WHU_OHS Dataset Test')
    parser.add_argument('--config', default='models/yamls/resnet.yaml', type=str,
                        help='Config file path (default: config.yaml)')
    parser.add_argument('--log_path', default='logs/resnet.log', type=str, help='Log path')
    parser.add_argument('--data_root', default='', type=str, help='Data root')
    parser.add_argument('--batch_size', default=8, type=int, help='Mini-batch size (default: 8)')
    parser.add_argument('--pretrained_model', default='DataStorage/resnet/best_model.pth', type=str,
                        help='Pretrained model path')
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, 'r'))
    device = torch.device('cuda')
    num_classes = cfg['num_classes']

    save_path = os.path.join(os.path.dirname(args.pretrained_model), 'test')
    os.makedirs(save_path, exist_ok=True)
    os.makedirs('./logs/', exist_ok=True)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    set_seed(233)

    logging.basicConfig(filename=args.log_path, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s: %(message)s')

    model = DSTC(cfg)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.cuda()
    model.load_state_dict(torch.load(args.pretrained_model), strict=False)

    test_loader = load_data(args, 'ts')
    confusion_matrix = test(model, test_loader, device, save_path, num_classes, args.batch_size)
    mean_f1, oa, kappa, miou, f1 = eval_metrics(confusion_matrix, mode='ts')

    print('mean_F1: {:.4f}, OA: {:.4f}, Kappa: {:.4f}, mIoU: {:.4f}'.format(mean_f1, oa, kappa, miou))
    print('class F1: ', f1)

    logging.info('mean_F1: {:.4f}, OA: {:.4f}, Kappa: {:.4f}, mIoU: {:.4f}'.format(mean_f1, oa, kappa, miou))
    logging.info(f'class F1: {f1}')


if __name__ == '__main__':
    main()

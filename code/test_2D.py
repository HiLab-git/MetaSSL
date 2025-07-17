import argparse
import os
import shutil

import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm
from networks.vision_transformer import SwinUnet as ViT_seg
# from networks.efficientunet import UNet
from networks.net_factory import net_factory
from config import get_config

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/mnt/data3/semi_zwr/SSL4MIS-master/data/ACDC', help='Name of Experiment')



parser.add_argument('--labeled_num', type=int, default=7,
                    help='labeled data')

parser.add_argument('--exp', type=str,
                    default='ACDC/FixMatch_standard_augs', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet_urpc', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=40000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=8,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument(
    '--cfg', type=str, default="../code/configs/swin_tiny_patch4_window7_224_lite.yaml", help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true',
                    help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                    'full: cache all data, '
                    'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int,
                    help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true',
                    help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true',
                    help='Test throughput only')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=4,
                    help='labeled_batch_size per gpu')

# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')


def calculate_metric_percase_ja(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    # asd = metric.binary.asd(pred, gt)
    #hd95 = metric.binary.hd95(pred, gt)
    jaccard = metric.binary.jc(pred, gt)
    return jaccard
    # , hd95
    # , asd

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    # asd = metric.binary.asd(pred, gt)
    #return dice
    if np.any(pred > 0) and np.any(gt > 0):
        hd95 = metric.binary.hd95(pred, gt)
        return hd95
    else:
        # 如果没有值大于零，可以选择返回一个特定的值，比如0，或者抛出一个异常
        print("Warning: No binary objects found in the input arrays.")
        return 50  # 或者抛出异常: raise ValueError("Input arrays do not contain any binary objects.")
    #jaccard = metric.binary.jc(pred, gt)
    #return hd95


def test_single_volume(case, net, test_save_path, FLAGS):
    h5f = h5py.File(FLAGS.root_path + "/data/{}.h5".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (256 / x, 256/ y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            if FLAGS.model == "unet_urds":
                out_main, _, _, _ = net(input)
            else:
                out_main, _, _, _ = net(input)
            out = torch.argmax(torch.softmax(
                out_main, dim=1), dim=1).squeeze(0)




            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / 256, y / 256), order=0)
            prediction[ind] = pred

    first_metric = calculate_metric_percase(prediction == 1, label == 1)
    second_metric = calculate_metric_percase(prediction == 2, label == 2)
    third_metric = calculate_metric_percase(prediction == 3, label == 3)
    first_ja = calculate_metric_percase_ja(prediction == 1, label == 1)
    second_ja = calculate_metric_percase_ja(prediction == 2, label == 2)
    third_ja = calculate_metric_percase_ja(prediction == 3, label == 3)

    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing((1, 1, 10))
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.SetSpacing((1, 1, 10))
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.SetSpacing((1, 1, 10))
    sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
    sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
    sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
    return first_metric, second_metric, third_metric,first_ja,second_ja,third_ja



def Inference(FLAGS,config):
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0]
                         for item in image_list])
    
    # snapshot_path = "../model/{}_{}_labeled/{}".format(
    #/mnt/data3/semi_zwr/SSL4MIS-master/model19/ACDC/Cross_Pseudo_Supervision_7/unet/model19/ACDC/Cross_Pseudo_Supervision_7/unet/model1_iter_7400_dice_0.8127.pth
    #/mnt/data3/semi_zwr/SSL4MIS-master/model127alpha = 0.05_noconfig=0.1
    snapshot_path = "/mnt/data3/semi_zwr/SSL_ours/t7_new_ours_1%_(e^3)_(e=2)/ACDC/Cross_Pseudo_Supervision_1/unet/model2_iter_23400_dice_1.pth"
    # test_save_path = "../model/{}_{}_labeled/{}_predictions/".format(
    test_save_path = "../test_1%/".format(        
        FLAGS.exp, FLAGS.labeled_num, FLAGS.model)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    
    def create_model(ema=False):
        # Network definition
        model = net_factory(net_type=args.model, in_chns=1,
                            class_num=4)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model
    
    net = create_model()
    print("1")
    save_mode_path = "/mnt/data3/semi_zwr/SSL_ours/z_6_50*label_40*unlabel_87.40%/ACDC/Cross_Pseudo_Supervision_3/unet/model2_iter_22600_dice_1.pth"
    print(save_mode_path)
    net.load_state_dict(torch.load(save_mode_path,map_location='cuda:0'), strict=False)
    print("init weight from {}".format(save_mode_path))
    net.eval()
    
    first_total = 0.0
    second_total = 0.0
    third_total = 0.0
    first_ja_total = 0.0
    second_ja_total = 0.0
    third_ja_total = 0.0

    first_metrics = []
    second_metrics = []  
    third_metrics = []
    first_ja_metrics = []
    second_ja_metrics = []
    third_ja_metrics = []
    for case in tqdm(image_list):
        print(case)
        first_metric, second_metric, third_metric,first_ja,second_ja,third_ja = test_single_volume(
            case, net, test_save_path, FLAGS)
        first_total += np.asarray(first_metric)
        second_total += np.asarray(second_metric)
        third_total += np.asarray(third_metric)
        print("ave",(first_metric+second_metric+third_metric)/3)
        first_ja_total += np.asarray(first_ja)
        second_ja_total += np.asarray(second_ja)
        third_ja_total += np.asarray(third_ja)
        first_ja_metrics.append(first_ja)
        second_ja_metrics.append(second_ja)
        third_ja_metrics.append(third_ja)
        first_metrics.append(first_metric)
        second_metrics.append(second_metric)
        third_metrics.append(third_metric)
    first_std = np.std(first_metrics)
    second_std = np.std(second_metrics)
    third_std = np.std(third_metrics)  
    first_ja_std = np.std(first_ja_metrics)
    second_ja_std = np.std(second_ja_metrics)
    third_ja_std = np.std(third_ja_metrics)
    avg_metric_ja = [first_ja_total / len(image_list), second_ja_total /
                    len(image_list), third_ja_total / len(image_list)]
    ave_ja = (first_ja_total/ len(image_list) + second_ja_total/ len(image_list) + third_ja_total/ len(image_list)) / 3
    print("avg_metric_ja",avg_metric_ja)
    print("ave_ja_std",ave_ja)
    ja_std=[first_ja_std,second_ja_std,third_ja_std]
    avg_ja_std = (first_ja_std + second_ja_std + third_ja_std) / 3
    print("ja_std",ja_std)
    print("ja_std",avg_ja_std)
    avg_metric = [first_total / len(image_list), second_total /
                  len(image_list), third_total / len(image_list)]
    ave_hd95 = (first_total / len(image_list) + second_total /
                  len(image_list) + third_total / len(image_list)) / 3
    
    std=[first_std,second_std,third_std]
    avg_std = (first_std + second_std + third_std) / 3
    print("ave_hd95",ave_hd95)
    print("avg_metric",avg_metric)
    print("avg_std",avg_std)
    print("std",std)
    return avg_metric,std



if __name__ == '__main__':
    args = parser.parse_args()
    config = get_config(args)
    torch.cuda.set_device(1)
    metric,std= Inference(args,config)
    np.savetxt("metric.txt", metric)
    np.savetxt("std.txt", std)
    print(metric)
    print((metric[0]+metric[1]+metric[2])/3)
    print(std)
    print((std[0]+std[1]+std[2])/3)


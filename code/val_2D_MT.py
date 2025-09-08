import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import SimpleITK as sitk
import os
import shutil


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    else:
        return 0, 0

def vote_threshold_label_selection_class_2D(pred1, pred2, cla1, cla2):
    """ 
        input:
            pred1 & pred2: logits with per-class prediction probability: B, C, H, W
            threshold: confident predictions

        output:
            label: one label for all three branches
            mask: confident or not
    """
    #threshold1=0.95
    #threshold2=0.95
    #print('threshold*****************************: ', threshold)
    # only vote
    # same prediction: weight = 1.0
    # different prediction: weight = 0.5
    #threshold = threshold.unsqueeze(1).unsqueeze(2) 
    #new_tensor = torch.zeros([24, 256, 256])
    #threshold = threshold.repeat(1, 256, 256)
    #print('threshold##############################: ', threshold)
    pred1=torch.softmax(pred1, dim=1)
    pred2=torch.softmax(pred2, dim=1)
    pred1_confidence, pred1_label = pred1.max(dim=1)
    pred2_confidence, pred2_label = pred2.max(dim=1)
    same_pred = (pred1_label == pred2_label)
    different_pred = (pred1_label != pred2_label)

    alpha = 1
    cla1_0=cla1[0]
    cla1_1=cla1[1]
    cla1_2=cla1[2]
    cla1_3=cla1[3]
    cla2_0=cla2[0]
    cla2_1=cla2[1]
    cla2_2=cla2[2]
    cla2_3=cla2[3]
    cla1_0=(cla1[0]*(cla1_0<=alpha)+alpha*(cla1_0>alpha))
    cla1_1=(cla1[1]*(cla1_1<=alpha)+alpha*(cla1_1>alpha))
    cla1_2=(cla1[2]*(cla1_2<=alpha)+alpha*(cla1_2>alpha))
    cla1_3=(cla1[3]*(cla1_3<=alpha)+alpha*(cla1_3>alpha))
    cla2_0=(cla2[0]*(cla2_0<=alpha)+alpha*(cla2_0>alpha))
    cla2_1=(cla2[1]*(cla2_1<=alpha)+alpha*(cla2_1>alpha))
    cla2_2=(cla2[2]*(cla2_2<=alpha)+alpha*(cla2_2>alpha))
    cla2_3=(cla2[3]*(cla2_3<=alpha)+alpha*(cla2_3>alpha))
    
    cla1_0=cla1_0[0]
    cla1_1=cla1_1[0]
    cla1_2=cla1_2[0]
    cla1_3=cla1_3[0]
    cla2_0=cla2_0[0]
    cla2_1=cla2_1[0]
    cla2_2=cla2_2[0]
    cla2_3=cla2_3[0]
    
    visualize1 = 1 * np.ones((1,256,256))
    visualize2 = 1 * np.ones((1,256,256))
    visualize1 = torch.from_numpy(visualize1).cuda()
    visualize2 = torch.from_numpy(visualize2).cuda()
    visualize1_diff_same=1* np.ones((1,256,256))
    visualize1_diff_same = torch.from_numpy(visualize1_diff_same).cuda()
    visualize1_confident=1* np.ones((1,256,256))
    visualize1_confident = torch.from_numpy(visualize1_confident).cuda()
    visualize2_confident=1* np.ones((1,256,256))
    visualize2_confident = torch.from_numpy(visualize2_confident).cuda()
    visualize1_diff_confident=1* np.ones((1,256,256))
    visualize1_diff_confident = torch.from_numpy(visualize1_diff_confident).cuda()
    visualize1_diff_noconfident=1* np.ones((1,256,256))
    visualize1_diff_noconfident = torch.from_numpy(visualize1_diff_noconfident).cuda()
    visualize2_diff_confident=1* np.ones((1,256,256))
    visualize2_diff_confident = torch.from_numpy(visualize2_diff_confident).cuda()
    visualize2_diff_noconfident=1* np.ones((1,256,256))
    visualize2_diff_noconfident = torch.from_numpy(visualize2_diff_noconfident).cuda()
    visualize1_same_confident=1* np.ones((1,256,256))
    visualize1_same_confident = torch.from_numpy(visualize1_same_confident).cuda()
    visualize1_same_noconfident=1* np.ones((1,256,256))
    visualize1_same_noconfident = torch.from_numpy(visualize1_same_noconfident).cuda()
    visualize2_same_confident=1* np.ones((1,256,256))
    visualize2_same_confident = torch.from_numpy(visualize2_same_confident).cuda()
    visualize2_same_noconfident=1* np.ones((1,256,256))
    visualize2_same_noconfident = torch.from_numpy(visualize2_same_noconfident).cuda()
    visualize1_back1=1* np.ones((1,256,256))
    visualize1_back1 = torch.from_numpy(visualize1_back1).cuda()
    visualize2_back2=1* np.ones((1,256,256))
    visualize2_back2 = torch.from_numpy(visualize2_back2).cuda()

    
    pixel_thresholds1 = 1 * np.ones((1,256,256))
    pixel_thresholds2 = 1 * np.ones((1,256,256))
    pixel_thresholds1 = torch.from_numpy(pixel_thresholds1).cuda() 
    pixel_thresholds2 = torch.from_numpy(pixel_thresholds2).cuda()
    
    #print('pred1_confidence: ', pred1_confidence.shape)
    #print('threshold1: ', threshold1.shape)
    threshold_1=(pixel_thresholds1*cla1_0*(pred1_label==0)+pixel_thresholds1*cla1_1*(pred1_label==1)+pixel_thresholds1*cla1_2*(pred1_label==2)+pixel_thresholds1*cla1_3*(pred1_label==3))
    threshold_2=(pixel_thresholds2*cla2_0*(pred2_label==0)+pixel_thresholds2*cla2_1*(pred2_label==1)+pixel_thresholds2*cla2_2*(pred2_label==2)+pixel_thresholds2*cla2_3*(pred2_label==3))

    
    
    #print('different_pred: ', different_pred.shape)
    same_pred_confident1 = same_pred * (pred1_confidence > threshold_1)*(pred1_label != 0)
    same_pred_confident2 = same_pred * (pred2_confidence > threshold_2)*(pred1_label != 0)
    back1=same_pred* (pred1_confidence > threshold_1)*(pred1_label == 0)
    back2=same_pred* (pred2_confidence > threshold_2)*(pred2_label == 0)
    same_pred_noconfident1 = same_pred * (pred1_confidence <= threshold_1)
    same_pred_noconfident2 = same_pred * (pred2_confidence <= threshold_2)
    
    #print('same_pred_confident1: ', same_pred_confident1.sum())
   #print('same_pred_confident1_back: ', same_pred_confident1_back.sum())
    #print('same_pred_confident2_back: ', same_pred_confident2_back.sum())
    confident1=pred1_confidence > threshold_1
    confident2=pred2_confidence > threshold_2
    no_confident1=pred1_confidence <= threshold_1
    no_confident2=pred2_confidence <= threshold_2
    different1_confident = different_pred * (pred1_confidence > threshold_1)
    #print('different1_confident: ', different1_confident.shape)
    #different1_else = ~different1_confident
    different2_confident = different_pred * (pred2_confidence > threshold_2)
    #different2_else = ~different2_confident
    different_noconfident1 = different_pred * (pred1_confidence <= threshold_1)
    different_noconfident2 = different_pred * (pred2_confidence <= threshold_2)
    #other1 = ~(different1_confident | same_pred_confident1)
    #other2 = ~(different2_confident | same_pred_confident2)
    #我想把same_pred_confident1，back1，same_pred_noconfident1，different1_confident，different_noconfident1这五个不同的区域表示在一张图片foul_label1上，然后再把这张图片保存下来
    #我想把same_pred_confident2，back2，same_pred_noconfident2，different2_confident，different_noconfident2这五个不同的区域表示在一张图片foul_label2上，然后再把这张图片保存下来
    # Define regions on foul_label1
    visualize1[same_pred_confident1] = 1
    visualize1[back1] = 2
    visualize1[same_pred_noconfident1] = 3
    visualize1[different1_confident] = 4
    visualize1[different_noconfident1] = 5

    # Define regions on foul_label2
    visualize2[same_pred_confident2] = 1
    visualize2[back2] = 2
    visualize2[same_pred_noconfident2] = 3
    visualize2[different2_confident] = 4
    visualize2[different_noconfident2] = 5

    visualize1_diff_same[same_pred] = 1
    visualize1_diff_same[different_pred]=2

    visualize1_confident[confident1] = 1
    visualize1_confident[no_confident1]=2

    visualize2_confident[confident2] = 1
    visualize2_confident[no_confident2]=2

    visualize1_diff_confident[different1_confident] = 1
    visualize1_diff_confident[(different1_confident==0)]=2
    visualize1_diff_noconfident[different_noconfident1]=1
    visualize1_diff_noconfident[(different_noconfident1==0)]=2
    visualize2_diff_confident[different2_confident] = 1
    visualize2_diff_confident[(different2_confident==0)]=2
    visualize2_diff_noconfident[different_noconfident2]=1
    visualize2_diff_noconfident[(different_noconfident2==0)]=2
    visualize1_same_confident[same_pred_confident1] = 1
    visualize1_same_confident[(back1==0)]=1
    visualize1_same_confident[(same_pred_confident1==0)]=2
    visualize1_same_noconfident[same_pred_noconfident1]=1
    visualize1_same_noconfident[(same_pred_noconfident1==0)]=2
    visualize2_same_confident[same_pred_confident2] = 1
    visualize2_same_confident[(back2==0)]=1
    visualize2_same_confident[(same_pred_confident2==0)]=2
    visualize2_same_noconfident[same_pred_noconfident2]=1
    visualize2_same_noconfident[(same_pred_noconfident2==0)]=2
    visualize1_back1[back1]=1
    visualize1_back1[(back1==0)]=2
    visualize2_back2[back2]=1
    visualize2_back2[(back2==0)]=2

    


    return visualize1,visualize2,visualize1_diff_same,visualize1_confident,visualize2_confident,visualize1_diff_confident,visualize1_diff_noconfident,visualize2_diff_confident,visualize2_diff_noconfident,visualize1_same_confident,visualize1_same_noconfident,visualize2_same_confident,visualize2_same_noconfident,visualize1_back1,visualize2_back2

def test_single_volume1(image, label, net, classes, patch_size=[256, 256]):
    #改成滑动窗口预测
    print("+++++++++++++++++++++++++++++++++++++",image.shape)
    print("+++++++++++++++++++++++++++++++++++++",label.shape)
    
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    print("+++++++++++++++++++++++++++++++++++++",image.shape)
    print("+++++++++++++++++++++++++++++++++++++",label.shape)
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        #val_outputs = sliding_window_inference(val_images, [args.input_size, args.input_size], 4, model, overlap=overlap)
        with torch.no_grad():
            print("input.shape",input.shape)
            out1=net(input)
            print("out1.shape",out1.shape)
            out = torch.argmax(torch.softmax(
                out1, dim=1), dim=1).squeeze(0)
            #print("out.shape",out.shape)
            out = out.cpu().detach().numpy()
            print("out.shape",out.shape)
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list

def test_single_volume2(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        
        with torch.no_grad():
            out1=net(input)
            out = torch.argmax(torch.softmax(
                out1, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list

def test_single_volume_CCVC(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        
        with torch.no_grad():
            out1,_=net(input)
            out = torch.argmax(torch.softmax(
                out1, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list

def test_single_volume3(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        
        with torch.no_grad():
            out1,_=net(input)
            out = torch.argmax(torch.softmax(
                out1, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list

def test_single_volume4(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        
        with torch.no_grad():
            _,out1=net(input)
            out = torch.argmax(torch.softmax(
                out1, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list


def test_single_volume(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        
        with torch.no_grad():
            out1=net(input)
            out = torch.argmax(torch.softmax(
                out1, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list

def test_single_volume_foul(image, label, net,net1,cla1,cla2,ilter_num,test_save_path,count,classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    prediction1 = np.zeros_like(label)
    foul_Label1 = np.zeros_like(label)
    foul_Label2 = np.zeros_like(label)
    foul_same_diff=np.zeros_like(label)
    foul_confident1=np.zeros_like(label)
    foul_confident2=np.zeros_like(label)
    ze1_diff_confident=np.zeros_like(label)
    ze1_diff_noconfident=np.zeros_like(label)
    ze2_diff_confident=np.zeros_like(label)
    ze2_diff_noconfident=np.zeros_like(label)
    ze1_same_confident=np.zeros_like(label)
    ze1_same_noconfident=np.zeros_like(label)
    ze2_same_confident=np.zeros_like(label)
    ze2_same_noconfident=np.zeros_like(label)
    ze1_back=np.zeros_like(label)
    ze2_back=np.zeros_like(label)



    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out_other=net1(input)
            out = torch.argmax(torch.softmax(
                out_other, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction1[ind] = pred
            

        with torch.no_grad():
            out1=net(input)
            out = torch.argmax(torch.softmax(
                out1, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
            visualize1,visualize2,visualize1_diff_same,visualize1_confident,visualize2_confident,visualize1_diff_confident,visualize1_diff_noconfident,visualize2_diff_confident,visualize2_diff_noconfident,visualize1_same_confident,visualize1_same_noconfident,visualize2_same_confident,visualize2_same_noconfident,visualize1_back1,visualize2_back2=vote_threshold_label_selection_class_2D(out1,out_other,cla1,cla2)
            visualize1=visualize1.squeeze(0)
            visualize2=visualize2.squeeze(0)
            visualize1=visualize1.cpu().detach().numpy()
            visualize2=visualize2.cpu().detach().numpy()
            visualize1_diff_same=visualize1_diff_same.squeeze(0)
            visualize1_diff_same=visualize1_diff_same.cpu().detach().numpy()
            visualize1_confident=visualize1_confident.squeeze(0)
            visualize1_confident=visualize1_confident.cpu().detach().numpy()
            visualize2_confident=visualize2_confident.squeeze(0)
            visualize2_confident=visualize2_confident.cpu().detach().numpy()
            visualize1_diff_confident=visualize1_diff_confident.squeeze(0)
            visualize1_diff_confident=visualize1_diff_confident.cpu().detach().numpy()
            visualize1_diff_noconfident=visualize1_diff_noconfident.squeeze(0)
            visualize1_diff_noconfident=visualize1_diff_noconfident.cpu().detach().numpy()
            visualize2_diff_confident=visualize2_diff_confident.squeeze(0)
            visualize2_diff_confident=visualize2_diff_confident.cpu().detach().numpy()
            visualize2_diff_noconfident=visualize2_diff_noconfident.squeeze(0)
            visualize2_diff_noconfident=visualize2_diff_noconfident.cpu().detach().numpy()
            visualize1_same_confident=visualize1_same_confident.squeeze(0)
            visualize1_same_confident=visualize1_same_confident.cpu().detach().numpy()
            visualize1_same_noconfident=visualize1_same_noconfident.squeeze(0)
            visualize1_same_noconfident=visualize1_same_noconfident.cpu().detach().numpy()
            visualize2_same_confident=visualize2_same_confident.squeeze(0)
            visualize2_same_confident=visualize2_same_confident.cpu().detach().numpy()
            visualize2_same_noconfident=visualize2_same_noconfident.squeeze(0)
            visualize2_same_noconfident=visualize2_same_noconfident.cpu().detach().numpy()
            visualize1_back1=visualize1_back1.squeeze(0)
            visualize1_back1=visualize1_back1.cpu().detach().numpy()
            visualize2_back2=visualize2_back2.squeeze(0)
            visualize2_back2=visualize2_back2.cpu().detach().numpy()

        
            visualize1 = zoom(visualize1, (x / patch_size[0], y / patch_size[1]), order=0)
            visualize2 = zoom(visualize2, (x / patch_size[0], y / patch_size[1]), order=0)
            visualize1_diff_same = zoom(visualize1_diff_same, (x / patch_size[0], y / patch_size[1]), order=0)
            visualize1_confident = zoom(visualize1_confident, (x / patch_size[0], y / patch_size[1]), order=0)
            visualize2_confident = zoom(visualize2_confident, (x / patch_size[0], y / patch_size[1]), order=0)
            visualize1_diff_confident = zoom(visualize1_diff_confident, (x / patch_size[0], y / patch_size[1]), order=0)
            visualize1_diff_noconfident = zoom(visualize1_diff_noconfident, (x / patch_size[0], y / patch_size[1]), order=0)
            visualize2_diff_confident = zoom(visualize2_diff_confident, (x / patch_size[0], y / patch_size[1]), order=0)
            visualize2_diff_noconfident = zoom(visualize2_diff_noconfident, (x / patch_size[0], y / patch_size[1]), order=0)
            visualize1_same_confident = zoom(visualize1_same_confident, (x / patch_size[0], y / patch_size[1]), order=0)
            visualize1_same_noconfident = zoom(visualize1_same_noconfident, (x / patch_size[0], y / patch_size[1]), order=0)
            visualize2_same_confident = zoom(visualize2_same_confident, (x / patch_size[0], y / patch_size[1]), order=0)
            visualize2_same_noconfident = zoom(visualize2_same_noconfident, (x / patch_size[0], y / patch_size[1]), order=0)
            visualize1_back1 = zoom(visualize1_back1, (x / patch_size[0], y / patch_size[1]), order=0)
            visualize2_back2 = zoom(visualize2_back2, (x / patch_size[0], y / patch_size[1]), order=0)
            foul_Label1[ind]=visualize1
            foul_Label2[ind]=visualize2
            foul_same_diff[ind]=visualize1_diff_same
            foul_confident1[ind]=visualize1_confident
            foul_confident2[ind]=visualize2_confident
            ze1_diff_confident[ind]=visualize1_diff_confident
            ze1_diff_noconfident[ind]=visualize1_diff_noconfident
            ze2_diff_confident[ind]=visualize2_diff_confident
            ze2_diff_noconfident[ind]=visualize2_diff_noconfident
            ze1_same_confident[ind]=visualize1_same_confident
            ze1_same_noconfident[ind]=visualize1_same_noconfident
            ze2_same_confident[ind]=visualize2_same_confident
            ze2_same_noconfident[ind]=visualize2_same_noconfident
            ze1_back[ind]=visualize1_back1
            ze2_back[ind]=visualize2_back2
            
        
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    
    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing((1, 1, 10))
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.SetSpacing((1, 1, 10))
    prd_itk1 = sitk.GetImageFromArray(prediction1.astype(np.float32))
    prd_itk1.SetSpacing((1, 1, 10))
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.SetSpacing((1, 1, 10))
    foul_Label1_itk = sitk.GetImageFromArray(foul_Label1.astype(np.float32))
    foul_Label1_itk.SetSpacing((1, 1, 10))
    foul_Label2_itk = sitk.GetImageFromArray(foul_Label2.astype(np.float32))
    foul_Label2_itk.SetSpacing((1, 1, 10))
    foul_same_diff_itk = sitk.GetImageFromArray(foul_same_diff.astype(np.float32))
    foul_same_diff_itk.SetSpacing((1, 1, 10))
    foul_confident1_itk = sitk.GetImageFromArray(foul_confident1.astype(np.float32))
    foul_confident1_itk.SetSpacing((1, 1, 10))
    foul_confident2_itk = sitk.GetImageFromArray(foul_confident2.astype(np.float32))
    foul_confident2_itk.SetSpacing((1, 1, 10))
    ze1_diff_confident_itk = sitk.GetImageFromArray(ze1_diff_confident.astype(np.float32))
    ze1_diff_confident_itk.SetSpacing((1, 1, 10))
    ze1_diff_noconfident_itk = sitk.GetImageFromArray(ze1_diff_noconfident.astype(np.float32))
    ze1_diff_noconfident_itk.SetSpacing((1, 1, 10))
    ze2_diff_confident_itk = sitk.GetImageFromArray(ze2_diff_confident.astype(np.float32))
    ze2_diff_confident_itk.SetSpacing((1, 1, 10))
    ze2_diff_noconfident_itk = sitk.GetImageFromArray(ze2_diff_noconfident.astype(np.float32))
    ze2_diff_noconfident_itk.SetSpacing((1, 1, 10))
    ze1_same_confident_itk = sitk.GetImageFromArray(ze1_same_confident.astype(np.float32))
    ze1_same_confident_itk.SetSpacing((1, 1, 10))
    ze1_same_noconfident_itk = sitk.GetImageFromArray(ze1_same_noconfident.astype(np.float32))
    ze1_same_noconfident_itk.SetSpacing((1, 1, 10))
    ze2_same_confident_itk = sitk.GetImageFromArray(ze2_same_confident.astype(np.float32))
    ze2_same_confident_itk.SetSpacing((1, 1, 10))
    ze2_same_noconfident_itk = sitk.GetImageFromArray(ze2_same_noconfident.astype(np.float32))
    ze2_same_noconfident_itk.SetSpacing((1, 1, 10))
    ze1_back_itk = sitk.GetImageFromArray(ze1_back.astype(np.float32))
    ze1_back_itk.SetSpacing((1, 1, 10))
    ze2_back_itk = sitk.GetImageFromArray(ze2_back.astype(np.float32))
    ze2_back_itk.SetSpacing((1, 1, 10))


    sitk.WriteImage(foul_same_diff_itk, test_save_path  + str(count)+"_foul_same_diff.nii.gz")
    sitk.WriteImage(foul_confident1_itk, test_save_path  + str(count)+"_foul_confident1.nii.gz")
    sitk.WriteImage(foul_confident2_itk, test_save_path  + str(count)+"_foul_confident2.nii.gz")
    sitk.WriteImage(foul_Label1_itk, test_save_path  + str(count)+"_foul_Label1.nii.gz")
    sitk.WriteImage(foul_Label2_itk, test_save_path  +str(count)+ "_foul_Label2.nii.gz")
    sitk.WriteImage(ze1_diff_confident_itk, test_save_path  + str(count)+"_ze1_diff_confident.nii.gz")
    sitk.WriteImage(ze1_diff_noconfident_itk, test_save_path  + str(count)+"_ze1_diff_noconfident.nii.gz")
    sitk.WriteImage(ze2_diff_confident_itk, test_save_path  + str(count)+"_ze2_diff_confident.nii.gz")
    sitk.WriteImage(ze2_diff_noconfident_itk, test_save_path  + str(count)+"_ze2_diff_noconfident.nii.gz")
    sitk.WriteImage(ze1_same_confident_itk, test_save_path  + str(count)+"_ze1_same_confident.nii.gz")
    sitk.WriteImage(ze1_same_noconfident_itk, test_save_path  + str(count)+"_ze1_same_noconfident.nii.gz")
    sitk.WriteImage(ze2_same_confident_itk, test_save_path  + str(count)+"_ze2_same_confident.nii.gz")
    sitk.WriteImage(ze2_same_noconfident_itk, test_save_path  + str(count)+"_ze2_same_noconfident.nii.gz")
    sitk.WriteImage(ze1_back_itk, test_save_path  + str(count)+"_ze1_back.nii.gz")
    sitk.WriteImage(ze2_back_itk, test_save_path  + str(count)+"_ze2_back.nii.gz")
    sitk.WriteImage(prd_itk, test_save_path  + str(count) + "_pred.nii.gz")
    sitk.WriteImage(prd_itk1, test_save_path  + str(count) + "_pred1.nii.gz")
    sitk.WriteImage(img_itk, test_save_path  + str(count) +"_img.nii.gz")
    sitk.WriteImage(lab_itk, test_save_path  + str(count) +"_gt.nii.gz")
    return metric_list

def test_single_volume_ds(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            output_main, _, _, _ = net(input)
            out = torch.argmax(torch.softmax(
                output_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list

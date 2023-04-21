# 时间：2023/3/23
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
import argparse
import os
import time, platform

import cv2
import torch
from model import DexiNed
from utils import (image_normalization, save_image_batch_to_disk,
                   visualize_result,count_parameters)

def transform(img,mean_bgr,img_height=512,img_width=512, gt=None):
    # gt[gt< 51] = 0 # test without gt discrimination
    # trans = transforms.Compose([transforms.ToTensor()])
    # img = cv2.resize(img, (self.img_width, self.img_height))

    # print(
    #     f"actual size: {img.shape}, target size: {(img_height, img_width,)}")
    img = cv2.resize(img, (img_width, img_height))
    gt = None

    # Make images and labels at least 512 by 512
    # elif img.shape[0] < 512 or img.shape[1] < 512:
    #     img = cv2.resize(img, (self.args.test_img_width, self.args.test_img_height))  # 512
    #     gt = cv2.resize(gt, (self.args.test_img_width, self.args.test_img_height))  # 512
    #
    # # Make sure images and labels are divisible by 2^4=16
    # elif img.shape[0] % 16 != 0 or img.shape[1] % 16 != 0:
    #     img_width = ((img.shape[1] // 16) + 1) * 16
    #     img_height = ((img.shape[0] // 16) + 1) * 16
    #     img = cv2.resize(img, (img_width, img_height))
    #     gt = cv2.resize(gt, (img_width, img_height))
    # else:
    #     img_width = self.args.test_img_width
    #     img_height = self.args.test_img_height
    #     img = cv2.resize(img, (img_width, img_height))
    #     gt = cv2.resize(gt, (img_width, img_height))

    # if self.yita is not None:
    #     gt[gt >= self.yita] = 1
    img = np.array(img, dtype=np.float32)
    # if self.rgb:
    #     img = img[:, :, ::-1]  # RGB->BGR
    # img=cv2.resize(img, (400, 464))
    img -= mean_bgr
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img.copy()).float()


    gt = np.zeros((img.shape[:2]))
    gt = torch.from_numpy(np.array([gt])).float()

    return img, gt


def output(img,img_hw):
    tensor2 = None
    tmp_img2 = None
    edge_maps = []
    image_shape=img_hw

    tensor=img
    for i in tensor:
        tmp = torch.sigmoid(i).cpu().detach().numpy()
        edge_maps.append(tmp)
    tensor = np.array(edge_maps)
    # print(f"tensor shape: {tensor.shape}")
    # image_shape = [x.cpu().detach().numpy() for x in img_shape]
    # (H, W) -> (W, H)
    # image_shape = [[y, x] for x, y in zip(image_shape[0], image_shape[1])]
    idx = 0
    tmp = tensor[:, idx, ...]
    tmp2 = tensor2[:, idx, ...] if tensor2 is not None else None
        # tmp = np.transpose(np.squeeze(tmp), [0, 1, 2])
    tmp = np.squeeze(tmp)
    tmp2 = np.squeeze(tmp2) if tensor2 is not None else None
    i_shape=image_shape
    # Iterate our all 7 NN outputs for a particular image
    preds = []
    for i in range(tmp.shape[0]):
        tmp_img = tmp[i]
        tmp_img = np.uint8(image_normalization(tmp_img))
        tmp_img = cv2.bitwise_not(tmp_img)
            # tmp_img[tmp_img < 0.0] = 0.0
            # tmp_img = 255.0 * (1.0 - tmp_img)
            # Resize prediction to match input image size
        if not tmp_img.shape[1] == i_shape[0] or not tmp_img.shape[0] == i_shape[1]:
            tmp_img = cv2.resize(tmp_img, (i_shape[1], i_shape[0]))
            tmp_img2 = cv2.resize(tmp_img2, (i_shape[0], i_shape[1])) if tmp2 is not None else None

        else:
            preds.append(tmp_img)
        if i == 6:
            fuse = tmp_img
            fuse = fuse.astype(np.uint8)
            if tmp_img2 is not None:
                fuse2 = tmp_img2
                fuse2 = fuse2.astype(np.uint8)
                # fuse = fuse-fuse2
                fuse_mask = np.logical_and(fuse > 128, fuse2 < 128)
                fuse = np.where(fuse_mask, fuse2, fuse)
    idx += 1
    return fuse


def test(checkpoint_path,img, model,img_hw):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint filte note found: {checkpoint_path}")
    # print(f"Restoring weights from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path,
                                     map_location=torch.device('cpu')
                                     ))
    model.eval()

    with torch.no_grad():
        preds = model(img)
        preds=output(preds,img_hw)

    return preds

add_loss=st.sidebar.selectbox(
    "请选择损失",
    ("CATS_LOSS","BSCN_LOSS2")
)
add_epochs=st.sidebar.slider(
    "请选择清晰程度（越清晰线条细节越少）",
    1,7
)
uploaded_file = st.file_uploader("上传您的图片", type=["jpg", "jpeg", "png"])
res="1.png"
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image)
    cv_image= cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    # pil_image =Image.open(uploaded_file).convert('RGB')
    # st.image(pil_image)
    # 将PIL.Image格式的图片转换为torch.tensor格式
    img_hw=cv_image.shape[0:2]
    target_size=[512,512]
    trans = transforms.Compose([transforms.ToTensor()])
    # tensor_image=trans(pil_image)

#     tensor_image,gt=transform(cv_image,[167.15, 146.07, 124.62],img_height=target_size[0],img_width=target_size[1])
#     # tensor_image=trans(tensor_image)
#     tensor_image = tensor_image.unsqueeze(0)
#     model = DexiNed()
#     checkpoint_path=f'checkpoints/{add_loss}/{add_epochs}/{add_epochs}_model.pth'
    # checkpoint_path="checkpoints/CATS_LOSS/3/3_model.pth"
#     pre=test(checkpoint_path,tensor_image,model,img_hw)
    # pre=pre.squeeze(0)
    # trans2=transforms.ToPILImage()
    # image = trans2(pre)
    # # 显示图片
    st.image(res)
#     num_param = count_parameters(model)






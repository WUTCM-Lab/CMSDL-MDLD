
import torch
import cv2
from torch.utils.data import Dataset
import torchvision.transforms as Transforms
import numpy as np
import random
import scipy.io as io
import argparse


def to_standardize(image):
    mean = np.mean(image)
    var = np.mean(np.square(image - mean))
    image = (image - mean) / (np.sqrt(var) + 1e-12)
    return image


def to_normalization(image):
    image = image.astype(np.float32)
    image_max = np.max(image)
    image_min = np.min(image)
    if image_max - image_min < 1e-12:
        return np.zeros_like(image, dtype=np.float32)
    image = (image - image_min) / (image_max - image_min)
    return image


def idx2coordinates(idx, col):
    i = idx // col
    j = idx % col
    return i, j


def ensure_hwc(img):

    if img.ndim == 2:
        img = img[..., np.newaxis]
    return img


def align_spatial_size(t1, t2, gt):

    h = min(t1.shape[0], t2.shape[0], gt.shape[0])
    w = min(t1.shape[1], t2.shape[1], gt.shape[1])

    t1 = t1[:h, :w, ...]
    t2 = t2[:h, :w, ...]
    gt = gt[:h, :w, ...]

    return t1, t2, gt


def pad_image(img, pad):

    if img.ndim == 2:
        return cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    elif img.ndim == 3:
        return cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    else:
        raise ValueError(f"Unsupported img ndim: {img.ndim}")


def Load_data(data_name, t1_path, t2_path, gt_path, patch_size, mode):
    ps = patch_size
    pad = ps // 2

    if data_name == "bastrop":
        mat = io.loadmat(t1_path)
        t1 = mat['t1_L5'][:, :, 3]
        t2 = mat["t2_ALI"][:, :, 5]
        t1 = to_normalization(t1)
        t2 = to_normalization(t2)
        gt = mat["ROI_1"]

    elif data_name == "california_mat":
        mat = io.loadmat(t1_path)
        t1 = mat['t1_L5'][:, :, 0]
        t2 = mat["t2_L8"][:, :, 3]
        t1 = to_normalization(t1)
        t2 = to_normalization(t2)
        gt = mat["ROI_2"]

    elif data_name == "Liyukou":
        t1mat = io.loadmat(t1_path)
        t2mat = io.loadmat(t2_path)
        gtmat = io.loadmat(gt_path)
        t1 = t1mat['T1'][:, :, 0]
        t2 = t2mat['T2'][:, :, 3]
        t1 = to_normalization(t1)
        t2 = to_normalization(t2)
        gt = gtmat['a']

    elif data_name == "california":
        t1 = cv2.imread(t1_path, cv2.IMREAD_COLOR)
        t2 = cv2.imread(t2_path, cv2.IMREAD_COLOR)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

    elif data_name == "italy":
        t1 = cv2.imread(t1_path, cv2.IMREAD_COLOR)
        t2 = cv2.imread(t2_path, cv2.IMREAD_COLOR)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

    elif data_name == "yellow":
        t1 = cv2.imread(t1_path, cv2.IMREAD_COLOR)
        t2 = cv2.imread(t2_path, cv2.IMREAD_COLOR)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

    elif data_name == "shuguang":
        t1 = cv2.imread(t1_path, cv2.IMREAD_COLOR)
        t2 = cv2.imread(t2_path, cv2.IMREAD_COLOR)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

    elif data_name == "gloucester2":
        t1 = cv2.imread(t1_path, cv2.IMREAD_COLOR)
        t2 = cv2.imread(t2_path, cv2.IMREAD_COLOR)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

    elif data_name == "gloucester1":
        t1 = cv2.imread(t1_path, cv2.IMREAD_COLOR)
        t2 = cv2.imread(t2_path, cv2.IMREAD_COLOR)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

    elif data_name == "france":
        t1 = cv2.imread(t1_path, cv2.IMREAD_COLOR)
        t2 = cv2.imread(t2_path, cv2.IMREAD_COLOR)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

    elif data_name == "Mexico":
        t1 = cv2.imread(t1_path, cv2.IMREAD_COLOR)
        t2 = cv2.imread(t2_path, cv2.IMREAD_COLOR)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

    elif data_name == "Texas":
        t1 = cv2.imread(t1_path, cv2.IMREAD_COLOR)
        t2 = cv2.imread(t2_path, cv2.IMREAD_COLOR)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

    elif data_name == "Italy":
        t1 = cv2.imread(t1_path, cv2.IMREAD_COLOR)
        t2 = cv2.imread(t2_path, cv2.IMREAD_COLOR)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    else:
        raise ValueError(f"Unsupported data_name: {data_name}")

    t1 = ensure_hwc(t1)
    t2 = ensure_hwc(t2)

    if gt.ndim == 2:
        gt = gt[..., np.newaxis]


    t1, t2, gt = align_spatial_size(t1, t2, gt)

    gt = gt[:, :, 0]

    t1_expand = pad_image(t1, pad)
    t2_expand = pad_image(t2, pad)

    h, w, c = t1.shape

    t1_list = []
    t2_list = []
    gt_list = []

    if mode == 'train':
        for i in range(0, h, ps):
            for j in range(0, w, ps):
                patch1 = t1_expand[i:i + ps, j:j + ps]
                patch2 = t2_expand[i:i + ps, j:j + ps]


                if patch1.shape[0] != ps or patch1.shape[1] != ps:
                    patch1 = cv2.copyMakeBorder(
                        patch1,
                        0, ps - patch1.shape[0],
                        0, ps - patch1.shape[1],
                        cv2.BORDER_REFLECT
                    )
                if patch2.shape[0] != ps or patch2.shape[1] != ps:
                    patch2 = cv2.copyMakeBorder(
                        patch2,
                        0, ps - patch2.shape[0],
                        0, ps - patch2.shape[1],
                        cv2.BORDER_REFLECT
                    )

                t1_list.append(patch1)
                t2_list.append(patch2)
                gt_list.append(gt[i, j] / 255.0)

    elif mode == 'test':
        for i in range(h):
            for j in range(w):
                patch1 = t1_expand[i:i + ps, j:j + ps]
                patch2 = t2_expand[i:i + ps, j:j + ps]

                if patch1.shape[0] != ps or patch1.shape[1] != ps:
                    patch1 = cv2.copyMakeBorder(
                        patch1,
                        0, ps - patch1.shape[0],
                        0, ps - patch1.shape[1],
                        cv2.BORDER_REFLECT
                    )
                if patch2.shape[0] != ps or patch2.shape[1] != ps:
                    patch2 = cv2.copyMakeBorder(
                        patch2,
                        0, ps - patch2.shape[0],
                        0, ps - patch2.shape[1],
                        cv2.BORDER_REFLECT
                    )

                t1_list.append(patch1)
                t2_list.append(patch2)
                gt_list.append(gt[i, j] / 255.0)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return t1_list, t2_list, gt_list


class Data_Loader(Dataset):
    def __init__(self, data_name, t1_path, t2_path, gt_path, patch_size, mode, transform=None):
        self.data_name = data_name
        self.t1_path = t1_path
        self.t2_path = t2_path
        self.gt_path = gt_path
        self.ps = patch_size
        self.mode = mode
        self.transform = transform

        self.t1, self.t2, self.gt = Load_data(
            self.data_name,
            self.t1_path,
            self.t2_path,
            self.gt_path,
            self.ps,
            self.mode
        )

    def augment(self, image, flipCode):
        flip = cv2.flip(image, flipCode)
        return flip

    def __getitem__(self, index):
        patch1 = self.t1[index]
        patch2 = self.t2[index]

        if self.transform is not None:
            patch1 = self.transform(patch1)
            patch2 = self.transform(patch2)
        else:
            patch1 = torch.from_numpy(np.transpose(patch1, (2, 0, 1))).float()
            patch2 = torch.from_numpy(np.transpose(patch2, (2, 0, 1))).float()

        label = torch.tensor(self.gt[index], dtype=torch.float32)

        return patch1, patch2, label

    def __len__(self):
        return len(self.t1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', default='Texas', type=str)
    parser.add_argument('--t1_path', default='./data/Texas/im1.bmp', type=str)
    parser.add_argument('--t2_path', default='./data/Texas/im2.bmp', type=str)
    parser.add_argument('--gt_path', default='./data/Texas/reference.bmp', type=str)
    parser.add_argument('--patch_size', default=11, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--mode', default='test', type=str)
    args = parser.parse_args()

    dataset = Data_Loader(
        data_name=args.data_name,
        t1_path=args.t1_path,
        t2_path=args.t2_path,
        gt_path=args.gt_path,
        patch_size=args.patch_size,
        mode=args.mode,
        transform=Transforms.ToTensor()
    )

    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    for i, (image1, image2, label) in enumerate(loader):
        print(i, image1.shape, image2.shape, label.shape)
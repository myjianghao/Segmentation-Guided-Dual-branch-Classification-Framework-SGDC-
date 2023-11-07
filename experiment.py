import os
from torch import optim
import pytorch_lightning as pl
import torchvision.utils as vutils
from torch import nn
import torch
from typing import TypeVar
import numpy as np
from torch.autograd import Variable
from torch.nn import functional as F
from evaluation import *
from torchvision.utils import save_image
from torchvision import transforms

Tensor = TypeVar('torch.tensor')


def c_evaluation(y_pred, y_true):
    l_prec = []
    l_recall = []
    l_F1 = []
    for i in range(4):
        TP = torch.sum((y_pred == i) * (y_true == i)).item()
        FP = torch.sum((y_pred == i) * (y_true != i)).item()
        FN = torch.sum((y_pred != i) * (y_true == i)).item()
        TN = torch.sum((y_pred != i) * (y_true != i)).item()
        l_prec.append(TP / (TP + FP))
        l_recall.append(TP / (TP + FN))
        l_F1.append((2 * l_prec[i] * l_recall[i]) / (l_prec[i] + l_recall[i] + 1e-6))
    return sum(l_prec) / len(l_prec), sum(l_recall) / len(l_recall), sum(l_F1) / len(l_F1)


# 正太分布概率表
norm_prob = {
    '0.7': 0.758,
    '0.8': 0.7881,
    '0.9': 0.8159,
    '1.0': 0.8413,
    '1.1': 0.8643,
    '1.2': 0.8849,
    '1.3': 0.9032,
    '1.4': 0.9192,
    '1.5': 0.9332,
    '1.6': 0.9452,
    '1.7': 0.9554,
    '1.8': 0.9641,
    '1.9': 0.9713,
    '2.0': 0.9772,
    '2.1': 0.9821,
    '2.2': 0.9861,
    '2.3': 0.9893,
}


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


from model.Unet import U_Net
import pytorch_lightning as pl


class se_net(pl.LightningModule):
    def __init__(self):
        super(se_net, self).__init__()
        self.model = U_Net()

    def forward(self, input):
        return self.model(input)


def dilate(bin_img, ksize=3):
    pad = (ksize - 1) // 2
    bin_img = F.pad(bin_img, pad=[pad, pad, pad, pad], mode='reflect')
    out = F.max_pool2d(bin_img, kernel_size=ksize, stride=1, padding=0)
    return out


class Resnetperiment(pl.LightningModule):

    def __init__(self,
                 Resnet_model,
                 loss_array,
                 batch_len,
                 is_mixup,
                 is_regression: bool,
                 is_camloss: bool,
                 params: dict) -> None:
        super(Resnetperiment, self).__init__()
        self.model = Resnet_model
        self.params = params
        if is_regression:
            self.loss_function = F.mse_loss
        else:
            self.loss_function = nn.CrossEntropyLoss()
        self.curr_device = None
        self.loss_array = loss_array
        self.batch_len = batch_len
        self.threshold_v_top = 0
        self.threshold_v_bottom = 0
        self.last_mean = 1
        self.x_factor = 2.0
        self.is_mixup = is_mixup
        self.is_regression = is_regression
        self.is_camloss = is_camloss
        # self.U_net = se_net.load_from_checkpoint("./checkpoints/U_Net/version_6/epoch=69-step=17009.ckpt").cuda()

    def forward(self, input: Tensor, is_camloss: bool, **kwargs) -> Tensor:
        # input = torch.cat((dilate((self.U_net(input) > 0.5).float()), input), dim=1)
        return self.model(input, is_camloss, **kwargs)

    def training_step(self, batch, batch_idx):
        real_img, labels = batch
        self.curr_device = real_img.device
        if self.is_regression:
            labels = labels.view(-1, 1)
        else:
            labels = labels - 1

        if self.is_mixup:
            real_img, targets_a, targets_b, lam = mixup_data(real_img, labels)
            real_img, targets_a, targets_b = map(Variable, (real_img,
                                                            targets_a, targets_b))
        outputs, cam = self.forward(real_img, self.is_camloss)
        mean = np.zeros(4)
        std = np.zeros(4)
        # 判断是否要更新
        if self.current_epoch > 12:
            # if 0:
            if self.is_mixup:
                train_loss = mixup_criterion(self.loss_function, outputs, targets_a, targets_b, lam)
            else:
                train_loss = self.loss_function(outputs, labels)
            if self.current_epoch % 2:
                if self.is_regression:
                    predicts = torch.round(outputs)
                else:
                    predicts = torch.argmax(outputs, dim=1)
                accuracy = torch.sum(predicts == labels) / len(labels)
                self.log_dict({"train_loss": train_loss.item(), "train_accuracy": accuracy.item()}, sync_dist=True)
                train_loss = train_loss * 0
                for i in range(len(outputs)):
                    """if self.is_regression:
                        self.loss_array[batch_idx * self.batch_len + i] = self.loss_function(outputs[i].reshape(-1),
                                                                                             labels[i].reshape(
                                                                                                 -1)).item()
                    else:
                        self.loss_array[batch_idx * self.batch_len + i] = self.loss_function(outputs[i].reshape(1, -1),
                                                                                             labels[i].reshape(
                                                                                                 1)).item()"""
                    mask_indicator = torch.zeros(len(outputs[i]), dtype=bool)
                    mask_indicator[labels[i]] = 1
                    self.loss_array[labels[i], batch_idx * self.batch_len + i] = sum(
                        outputs[i].cpu() * mask_indicator) - max(
                        outputs[i].cpu() * (~mask_indicator))
            else:
                compare_unit = []
                for i in range(len(outputs)):
                    mask_indicator = torch.zeros(len(outputs[i]), dtype=bool)
                    mask_indicator[labels[i]] = 1
                    compare_unit.append(
                        sum(outputs[i].cpu() * mask_indicator) - max(outputs[i].cpu() * (~mask_indicator)))
                if batch_idx == 0:
                    for i in range(len(self.loss_array)):
                        mean[i] = np.mean(self.loss_array[i][np.logical_not(np.isnan(self.loss_array[i]))])
                        std[i] = np.std(self.loss_array[i][np.logical_not(np.isnan(self.loss_array[i]))])
                loss_total = train_loss * 0
                for i in range(len(outputs)):
                    if (compare_unit[i] < mean[labels[i]] + self.x_factor * std[labels[i]]) & (
                            compare_unit[i] > mean[labels[i]] - self.x_factor * std[labels[i]]):
                        loss_total += self.loss_function(outputs[i].reshape(1, -1), labels[i].reshape(1)) * (
                                2 * norm_prob[str(self.x_factor)] - 1)
                    else:
                        loss_total += self.loss_function(outputs[i].reshape(1, -1), labels[i].reshape(1)) * 2 * (
                                1 - norm_prob[str(self.x_factor)])
                train_loss = loss_total / len(outputs)
                if self.is_regression:
                    predicts = torch.round(outputs)
                else:
                    predicts = torch.argmax(outputs, dim=1)
                accuracy = torch.sum(predicts == labels) / len(labels)
                self.log_dict({"train_loss": train_loss.item(), "train_accuracy": accuracy.item()}, sync_dist=True)
                """
                loss_unit = []
                for i in range(len(outputs)):
                    if self.is_regression:
                        loss_unit.append(self.loss_function(outputs[i].reshape(-1), labels[i].reshape(-1)).item())
                    else:
                        loss_unit.append(self.loss_function(outputs[i].reshape(1, -1), labels[i].reshape(1)).item())
                if batch_idx == 0:
                    mean_s = np.mean(self.loss_array[self.loss_array > 0])
                """
                """if (self.current_epoch > 50) & (self.last_mean < mean_s):
                        if str(self.x_factor - 0.1) in norm_prob:
                            self.x_factor -= 0.1
                    self.last_mean = mean_s
                    """
                """
                    std_s = np.std(self.loss_array[self.loss_array > 0])
                    self.threshold_v_top = mean_s + self.x_factor * std_s
                    self.threshold_v_bottom = mean_s - self.x_factor * std_s
                loss_com = np.array(loss_unit)
                if (np.all(loss_com < self.threshold_v_top)) & (np.all(loss_com > self.threshold_v_bottom)):
                    # if np.all(loss_com < self.threshold_v_top):
                    if self.is_regression:
                        predicts = torch.round(outputs)
                    else:
                        predicts = torch.argmax(outputs, dim=1)
                    accuracy = torch.sum(predicts == labels) / len(labels)
                    self.log_dict({"train_loss": train_loss.item(), "train_accuracy": accuracy.item()}, sync_dist=True)
                    train_loss = train_loss * (2 * norm_prob[str(self.x_factor)] - 1)
                else:
                    if self.is_regression:
                        predicts = torch.round(outputs)
                    else:
                        predicts = torch.argmax(outputs, dim=1)
                    accuracy = torch.sum(predicts == labels) / len(labels)
                    self.log_dict({"train_loss": train_loss.item(), "train_accuracy": accuracy.item()}, sync_dist=True)
                    train_loss = train_loss * 2 * (1 - norm_prob[str(self.x_factor)])
                """
        else:
            if self.is_mixup:
                train_loss = mixup_criterion(self.loss_function, outputs, targets_a, targets_b, lam)
            else:
                train_loss = self.loss_function(outputs, labels)
            if self.is_regression:
                predicts = torch.round(outputs)
            else:
                predicts = torch.argmax(outputs, dim=1)
            accuracy = torch.sum(predicts == labels) / len(labels)
            self.log_dict({"train_loss": train_loss.item(), "train_accuracy": accuracy.item()}, sync_dist=True)
        """if train_loss.item() < 0.2:
            loss_unit = []
            for i in range(len(outputs)):
                loss_unit.append(self.loss_function(outputs[i].reshape(1, -1), labels[i].reshape(1)).item())
            loss_array = np.array(loss_unit)
            if np.std(loss_array) > 0.15:
                fin_loss = train_loss * 0.3"""
        return train_loss

    def validation_step(self, batch, batch_idx):
        real_img, labels = batch
        self.curr_device = real_img.device
        if self.is_regression:
            labels = labels.view(-1, 1)
        else:
            labels = labels - 1
        outputs, cam = self.forward(real_img, self.is_camloss)
        val_loss = self.loss_function(outputs, labels)
        if self.is_regression:
            predicts = torch.round(outputs)
        else:
            predicts = torch.argmax(outputs, dim=1)
        accuracy = torch.sum(predicts == labels) / len(labels)
        self.log_dict({"val_loss": val_loss.item(), "val_accuracy": accuracy.item()}, on_step=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        real_img, labels, filename_list = batch
        self.curr_device = real_img.device
        if self.is_regression:
            labels = labels.view(-1, 1)
        else:
            labels = labels - 1
        outputs, cam = self.forward(real_img, self.is_camloss)
        test_loss = self.loss_function(outputs, labels)
        predicts = torch.argmax(outputs, dim=1)
        accuracy = torch.sum(predicts == labels) / len(labels)
        precision, recall, F1 = c_evaluation(predicts, labels)
        print({"test_loss": test_loss.item(), "test_accuracy": accuracy.item(), "precision": precision, "recall": recall,
             "F1": F1})
        self.log_dict(
            {"test_loss": test_loss.item(), "test_accuracy": accuracy.item(), "precision": precision, "recall": recall,
             "F1": F1}, on_step=True)

    def configure_optimizers(self):

        optims = []
        scheds = []
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])

        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model, self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

            try:
                if self.params['scheduler_gamma'] is not None:
                    scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                                 gamma=self.params['scheduler_gamma'])
                    scheds.append(scheduler)

                    # Check if another scheduler is required for the second optimizer
                    try:
                        if self.params['scheduler_gamma_2'] is not None:
                            scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                          gamma=self.params['scheduler_gamma_2'])
                            scheds.append(scheduler2)
                    except:
                        pass
                    return optims, scheds
            except:
                return optims


class Unetexperiment(pl.LightningModule):
    def __init__(self,
                 Unet_model,
                 loss_array,
                 batch_len,
                 is_mixup,
                 is_regression: bool,
                 is_camloss: bool,
                 params: dict) -> None:
        super(Unetexperiment, self).__init__()
        self.model = Unet_model
        self.params = params
        self.loss_function = nn.BCELoss()
        self.curr_device = None
        self.loss_array = loss_array
        self.batch_len = batch_len
        self.is_mixup = is_mixup
        self.is_regression = is_regression
        self.is_camloss = is_camloss

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx):
        real_img, labels = batch
        self.curr_device = real_img.device
        if self.is_mixup:
            real_img, targets_a, targets_b, lam = mixup_data(real_img, labels)
            real_img, targets_a, targets_b = map(Variable, (real_img,
                                                            targets_a, targets_b))
        outputs = self.forward(real_img)
        outputs_probs = torch.sigmoid(outputs)
        outputs_flat = outputs_probs.view(outputs_probs.size(0), -1)
        if self.is_mixup:
            targets_a = targets_a.view(labels.size(0), -1)
            targets_b = targets_b.view(labels.size(0), -1)
            train_loss = mixup_criterion(self.loss_function, outputs_flat, targets_a, targets_b, lam)
        else:
            labels_flat = labels.view(labels.size(0), -1)
            train_loss = self.loss_function(outputs_flat, labels_flat)
        acc = get_accuracy(outputs, labels)
        SE = get_sensitivity(outputs, labels)
        SP = get_specificity(outputs, labels)
        PC = get_precision(outputs, labels)
        F1 = get_F1(outputs, labels)
        JS = get_JS(outputs, labels)
        DC = get_DC(outputs, labels)
        self.log_dict(
            {"train_loss": train_loss.item(), "accuracy": acc, "SE": SE, "SP": SP,
             "PC": PC, "F1": F1, "JS": JS, "DC": DC}, sync_dist=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        real_img, labels = batch
        self.curr_device = real_img.device
        outputs = self.forward(real_img)
        outputs_probs = torch.sigmoid(outputs)
        outputs_flat = outputs_probs.view(outputs_probs.size(0), -1)
        labels_flat = labels.view(labels.size(0), -1)
        train_loss = self.loss_function(outputs_flat, labels_flat)
        acc = get_accuracy(outputs, labels)
        SE = get_sensitivity(outputs, labels)
        SP = get_specificity(outputs, labels)
        PC = get_precision(outputs, labels)
        F1 = get_F1(outputs, labels)
        JS = get_JS(outputs, labels)
        DC = get_DC(outputs, labels)
        self.log_dict(
            {"val_loss": train_loss.item(), "val_accuracy": acc, "val_SE": SE, "val_SP": SP,
             "val_PC": PC, "val_F1": F1, "val_JS": JS, "val_DC": DC}, sync_dist=True)

    def test_step(self, batch, batch_idx):
        real_img, labels = batch
        self.curr_device = real_img.device
        outputs = self.forward(real_img)
        outputs_probs = torch.sigmoid(outputs)
        outputs_flat = outputs_probs.view(outputs_probs.size(0), -1)
        labels_flat = labels.view(labels.size(0), -1)
        train_loss = self.loss_function(outputs_flat, labels_flat)
        acc = get_accuracy(outputs, labels)
        print(batch_idx, ":train_loss:", train_loss, "acc:", acc)
        to_pil = transforms.ToPILImage()
        save_image(real_img,
                   os.path.join("./Reconstructions",
                                str(batch_idx) + ".png"),
                   nrow=1)
        # resize = transforms.Resize((640, 896))
        pil_image = to_pil((outputs[0] > 0.5).float())
        # pil_image.show()
        pil_image.save(os.path.join("./Reconstructions",
                                    str(batch_idx) + "_m.png"))
        pil_image = to_pil(labels[0])
        pil_image.save(os.path.join("./Reconstructions",
                                    str(batch_idx) + "_lm.png"))

    def configure_optimizers(self):

        optims = []
        scheds = []
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])

        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model, self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

            try:
                if self.params['scheduler_gamma'] is not None:
                    scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                                 gamma=self.params['scheduler_gamma'])
                    scheds.append(scheduler)

                    # Check if another scheduler is required for the second optimizer
                    try:
                        if self.params['scheduler_gamma_2'] is not None:
                            scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                          gamma=self.params['scheduler_gamma_2'])
                            scheds.append(scheduler2)
                    except:
                        pass
                    return optims, scheds
            except:
                return optims


import cv2


class Unet_Resnetperiment(pl.LightningModule):

    def __init__(self,
                 U_Resmodel,
                 loss_array,
                 batch_len,
                 is_mixup,
                 is_regression: bool,
                 is_camloss: bool,
                 params: dict) -> None:
        super(Unet_Resnetperiment, self).__init__()
        self.model = U_Resmodel
        self.params = params
        self.loss_function1 = nn.BCELoss()
        if is_regression:
            self.loss_function2 = F.mse_loss
        else:
            self.loss_function2 = nn.CrossEntropyLoss()
        self.curr_device = None
        self.loss_array = loss_array
        self.batch_len = batch_len
        self.threshold_v_top = 0
        self.threshold_v_bottom = 0
        self.last_mean = 1
        self.x_factor = 2.0
        self.is_mixup = is_mixup
        self.is_regression = is_regression
        self.is_camloss = is_camloss

    def forward(self, input: Tensor, is_camloss: bool, s_labels: Tensor, **kwargs) -> Tensor:
        return self.model(input, is_camloss, s_labels, **kwargs)

    def training_step(self, batch, batch_idx):
        real_img, s_labels, c_labels = batch
        self.curr_device = real_img.device
        if self.is_regression:
            c_labels = c_labels.view(-1, 1)
        else:
            c_labels = c_labels - 1

        if self.is_mixup:
            real_img, targets_a, targets_b, lam = mixup_data(real_img, c_labels)
            real_img, targets_a, targets_b = map(Variable, (real_img,
                                                            targets_a, targets_b))
        Sm_outputs, C_outputs, cam = self.forward(real_img, self.is_camloss, s_labels)
        # 判断是否要更新
        Sm_outputs_probs = torch.sigmoid(Sm_outputs)
        Sm_outputs_flat = Sm_outputs_probs.view(Sm_outputs_probs.size(0), -1)
        s_labels_flat = s_labels.view(s_labels.size(0), -1)
        mean = np.zeros(4)
        std = np.zeros(4)
        if self.current_epoch > 16:
            # if 0:
            if self.is_mixup:
                loss_s = self.loss_function1(Sm_outputs_flat, s_labels_flat)
                loss_c = mixup_criterion(self.loss_function2, C_outputs, targets_a, targets_b, lam)
                train_loss = loss_s + loss_c
                if self.is_camloss:
                    label_resize = transforms.Resize((cam.shape[-2], cam.shape[-1]))
                    label_cam = label_resize(s_labels)
                    loss_cam = torch.sum((1 - label_cam) * cam) / torch.sum(cam)
                    train_loss += loss_cam ** 2
            else:
                loss_s = self.loss_function1(Sm_outputs_flat, s_labels_flat)
                loss_c = self.loss_function2(C_outputs, c_labels)
                train_loss = loss_s + loss_c
                if self.is_camloss:
                    label_resize = transforms.Resize((cam.shape[-2], cam.shape[-1]))
                    label_cam = label_resize(s_labels)
                    loss_cam = torch.sum((1 - label_cam) * cam) / torch.sum(cam)
                    train_loss += loss_cam ** 2
            if self.current_epoch % 2:
                if self.is_regression:
                    predicts = torch.round(C_outputs)
                else:
                    predicts = torch.argmax(C_outputs, dim=1)
                accuracy = torch.sum(predicts == c_labels) / len(c_labels)
                self.log_dict(
                    {"train_loss_s": loss_s.item(), "train_loss_c": loss_c.item(), "train_loss": train_loss.item(),
                     "train_accuracy": accuracy.item()}, sync_dist=True)
                train_loss = train_loss * 0
                for i in range(len(C_outputs)):
                    mask_indicator = torch.zeros(len(C_outputs[i]), dtype=bool)
                    mask_indicator[c_labels[i]] = 1
                    self.loss_array[c_labels[i], batch_idx * self.batch_len + i] = sum(
                        C_outputs[i].cpu() * mask_indicator) - max(
                        C_outputs[i].cpu() * (~mask_indicator))
            else:
                compare_unit = []
                for i in range(len(C_outputs)):
                    mask_indicator = torch.zeros(len(C_outputs[i]), dtype=bool)
                    mask_indicator[c_labels[i]] = 1
                    compare_unit.append(
                        sum(C_outputs[i].cpu() * mask_indicator) - max(C_outputs[i].cpu() * (~mask_indicator)))
                if batch_idx == 0:
                    for i in range(len(self.loss_array)):
                        mean[i] = np.mean(self.loss_array[i][np.logical_not(np.isnan(self.loss_array[i]))])
                        std[i] = np.std(self.loss_array[i][np.logical_not(np.isnan(self.loss_array[i]))])
                loss_total = loss_c * 0
                for i in range(len(C_outputs)):
                    if (compare_unit[i] < mean[c_labels[i]] + self.x_factor * std[c_labels[i]]) & (
                            compare_unit[i] > mean[c_labels[i]] - self.x_factor * std[c_labels[i]]):
                        loss_total += self.loss_function2(C_outputs[i].reshape(1, -1), c_labels[i].reshape(1)) * (
                                2 * norm_prob[str(self.x_factor)] - 1)
                    else:
                        loss_total += self.loss_function2(C_outputs[i].reshape(1, -1), c_labels[i].reshape(1)) * 2 * (
                                1 - norm_prob[str(self.x_factor)])
                loss_c = loss_total / len(C_outputs)
                train_loss = loss_s + loss_c
                if self.is_regression:
                    predicts = torch.round(C_outputs)
                else:
                    predicts = torch.argmax(C_outputs, dim=1)
                accuracy = torch.sum(predicts == c_labels) / len(c_labels)
                self.log_dict(
                    {"train_loss_s": loss_s.item(), "train_loss_c": loss_c.item(), "train_loss": train_loss.item(),
                     "train_accuracy": accuracy.item()}, sync_dist=True)
            """if self.current_epoch % 2:
                train_loss = train_loss * 0
                for i in range(len(C_outputs)):
                    if self.is_regression:
                        self.loss_array[batch_idx * self.batch_len + i] = self.loss_function2(
                            C_outputs[i].reshape(-1),
                            c_labels[i].reshape(-1)).item()
                    else:
                        self.loss_array[batch_idx * self.batch_len + i] = self.loss_function2(
                            C_outputs[i].reshape(1, -1),
                            c_labels[i].reshape(1)).item()
            else:
                loss_unit = []
                for i in range(len(C_outputs)):
                    if self.is_regression:
                        loss_unit.append(
                            self.loss_function2(C_outputs[i].reshape(-1), c_labels[i].reshape(-1)).item())
                    else:
                        loss_unit.append(
                            self.loss_function2(C_outputs[i].reshape(1, -1), c_labels[i].reshape(1)).item())
                if batch_idx == 0:
                    mean_s = np.mean(self.loss_array[self.loss_array > 0])
                    std_s = np.std(self.loss_array[self.loss_array > 0])
                    self.threshold_v_top = mean_s + self.x_factor * std_s
                    self.threshold_v_bottom = mean_s - self.x_factor * std_s
                loss_com = np.array(loss_unit)
                if (np.all(loss_com < self.threshold_v_top)) & (np.all(loss_com > self.threshold_v_bottom)):
                    # if np.all(loss_com < self.threshold_v_top):
                    if self.is_regression:
                        predicts = torch.round(C_outputs)
                    else:
                        predicts = torch.argmax(C_outputs, dim=1)
                    accuracy = torch.sum(predicts == c_labels) / len(c_labels)
                    self.log_dict(
                        {"train_loss_s": loss_s.item(), "train_loss_c": loss_c.item(), "train_loss": train_loss.item(),
                         "train_accuracy": accuracy.item()}, sync_dist=True)
                    loss_c = loss_c * (2 * norm_prob[str(self.x_factor)] - 1)
                    train_loss = loss_s + loss_c
                    if self.is_camloss:
                        label_resize = transforms.Resize((cam.shape[-2], cam.shape[-1]))
                        label_cam = label_resize(s_labels)
                        loss_cam = torch.sum((1 - label_cam) * cam) / torch.sum(cam)
                        train_loss += loss_cam ** 2
                else:
                    if self.is_regression:
                        predicts = torch.round(C_outputs)
                    else:
                        predicts = torch.argmax(C_outputs, dim=1)
                    accuracy = torch.sum(predicts == c_labels) / len(c_labels)
                    self.log_dict(
                        {"train_loss_s": loss_s.item(), "train_loss_c": loss_c.item(), "train_loss": train_loss.item(),
                         "train_accuracy": accuracy.item()}, sync_dist=True)
                    loss_c = loss_c * 2 * (1 - norm_prob[str(self.x_factor)])
                    train_loss = loss_s + loss_c
                    if self.is_camloss:
                        label_resize = transforms.Resize((cam.shape[-2], cam.shape[-1]))
                        label_cam = label_resize(s_labels)
                        loss_cam = torch.sum((1 - label_cam) * cam) / torch.sum(cam)
                        train_loss += loss_cam ** 2"""
        else:
            if self.is_mixup:
                loss_s = self.loss_function1(Sm_outputs_flat, s_labels_flat)
                loss_c = mixup_criterion(self.loss_function2, C_outputs, targets_a, targets_b, lam)
                train_loss = loss_s + loss_c
                if self.is_camloss:
                    label_resize = transforms.Resize((cam.shape[-2], cam.shape[-1]))
                    label_cam = label_resize(s_labels)
                    loss_cam = torch.sum((1 - label_cam) * cam) / torch.sum(cam)
                    train_loss += loss_cam ** 2
            else:
                loss_s = self.loss_function1(Sm_outputs_flat, s_labels_flat)
                loss_c = self.loss_function2(C_outputs, c_labels)
                train_loss = loss_s + loss_c
                if self.is_camloss:
                    label_resize = transforms.Resize((cam.shape[-2], cam.shape[-1]))
                    label_cam = label_resize(s_labels)
                    loss_cam = torch.sum((1 - label_cam) * cam) / torch.sum(cam)
                    train_loss += loss_cam ** 2
            if self.is_regression:
                predicts = torch.round(C_outputs)
            else:
                predicts = torch.argmax(C_outputs, dim=1)
            accuracy = torch.sum(predicts == c_labels) / len(c_labels)
            self.log_dict(
                {"train_loss_s": loss_s.item(), "train_loss_c": loss_c.item(), "train_loss": train_loss.item(),
                 "train_accuracy": accuracy.item()}, sync_dist=True)
        """if train_loss.item() < 0.2:
            loss_unit = []
            for i in range(len(outputs)):
                loss_unit.append(self.loss_function(outputs[i].reshape(1, -1), labels[i].reshape(1)).item())
            loss_array = np.array(loss_unit)
            if np.std(loss_array) > 0.15:
                fin_loss = train_loss * 0.3"""
        return train_loss

    def validation_step(self, batch, batch_idx):
        real_img, s_labels, c_labels = batch
        self.curr_device = real_img.device
        if self.is_regression:
            c_labels = c_labels.view(-1, 1)
        else:
            c_labels = c_labels - 1
        Sm_outputs, C_outputs, cam = self.forward(real_img, self.is_camloss, s_labels)
        Sm_outputs_probs = torch.sigmoid(Sm_outputs)
        Sm_outputs_flat = Sm_outputs_probs.view(Sm_outputs_probs.size(0), -1)
        s_labels_flat = s_labels.view(s_labels.size(0), -1)
        loss_s = self.loss_function1(Sm_outputs_flat, s_labels_flat)
        loss_c = self.loss_function2(C_outputs, c_labels)
        val_loss = loss_s + loss_c
        if self.is_camloss:
            label_resize = transforms.Resize((cam.shape[-2], cam.shape[-1]))
            label_cam = label_resize(s_labels)
            loss_cam = torch.sum((1 - label_cam) * cam) / torch.sum(cam)
            val_loss += loss_cam ** 2
        if self.is_regression:
            predicts = torch.round(C_outputs)
        else:
            predicts = torch.argmax(C_outputs, dim=1)
        accuracy = torch.sum(predicts == c_labels) / len(c_labels)
        self.log_dict({"val_loss_s": loss_s.item(), "val_loss_c": loss_c.item(), "val_loss": val_loss.item(),
                       "val_accuracy": accuracy.item()}, on_step=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        real_img, s_labels, c_labels = batch
        self.curr_device = real_img.device
        if self.is_regression:
            c_labels = c_labels.view(-1, 1)
        else:
            c_labels = c_labels - 1
        Sm_outputs, C_outputs, cam = self.forward(real_img, self.is_camloss, s_labels)
        Sm_outputs_probs = torch.sigmoid(Sm_outputs)
        Sm_outputs_flat = Sm_outputs_probs.view(Sm_outputs_probs.size(0), -1)
        s_labels_flat = s_labels.view(s_labels.size(0), -1)
        loss_s = self.loss_function1(Sm_outputs_flat, s_labels_flat)
        loss_c = self.loss_function2(C_outputs, c_labels)
        val_loss = loss_s + loss_c
        if self.is_camloss:
            label_resize = transforms.Resize((cam.shape[-2], cam.shape[-1]))
            label_cam = label_resize(s_labels)
            loss_cam = torch.sum((1 - label_cam) * cam) / torch.sum(cam)
            val_loss += loss_cam ** 2
        if self.is_regression:
            predicts = torch.round(C_outputs)
        else:
            predicts = torch.argmax(C_outputs, dim=1)
        accuracy = torch.sum(predicts == c_labels) / len(c_labels)
        precision, recall, F1 = c_evaluation(predicts, c_labels)
        """restore_transforms = transforms.Resize((mask_output.shape[-2], mask_output.shape[-1]))
        Sm_pig = 255 * (restore_transforms(Sm_outputs)[0][0] > 0.5).float().numpy()
        # Sm_pig = cv2.applyColorMap(np.uint8(Sm_pig), cv2.COLORMAP_JET)
        # cv2.imwrite("./heatmap/" + str(batch_idx) + "_s.png", Sm_pig)
        accuracy = torch.sum(predicts == c_labels) / len(c_labels)
        pred_edge_kk = mask_output[0][0].numpy()
        pred_edge_kk = (pred_edge_kk - pred_edge_kk.min()) / (pred_edge_kk.max() - pred_edge_kk.min() + 1e-8)
        pred_edge_kk = 255 * pred_edge_kk
        # heatmap = cv2.applyColorMap(np.uint8(pred_edge_kk), cv2.COLORMAP_JET)
        # cv2.imwrite("./heatmap/" + str(batch_idx) + "_h.png", heatmap)
        if (Sm_pig != pred_edge_kk).any():
            print(batch_idx)"""
        print(
            {"test_loss": loss_c.item(), "test_accuracy": accuracy.item(), "precision": precision, "recall": recall,
             "F1": F1})
        self.log_dict({"test_loss": loss_c.item(), "test_accuracy": accuracy.item(), "precision": precision, "recall": recall,
             "F1": F1}, on_step=True)

    def configure_optimizers(self):
        optims = []
        scheds = []
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])

        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model, self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

            try:
                if self.params['scheduler_gamma'] is not None:
                    scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                                 gamma=self.params['scheduler_gamma'])
                    scheds.append(scheduler)

                    # Check if another scheduler is required for the second optimizer
                    try:
                        if self.params['scheduler_gamma_2'] is not None:
                            scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                          gamma=self.params['scheduler_gamma_2'])
                            scheds.append(scheduler2)
                    except:
                        pass
                    return optims, scheds
            except:
                return optims


class CSFTperiment(pl.LightningModule):

    def __init__(self,
                 Resnet_model,
                 loss_array,
                 batch_len,
                 is_mixup,
                 is_regression: bool,
                 is_camloss: bool,
                 params: dict) -> None:
        super(CSFTperiment, self).__init__()
        self.model = Resnet_model
        self.params = params
        if is_regression:
            self.loss_function = F.mse_loss
        else:
            # self.loss_function = nn.CrossEntropyLoss(weight=torch.FloatTensor([230 / 1069, 435 / 1069, 201 / 1069, 203 / 1069]).cuda())
            self.loss_function = nn.CrossEntropyLoss()
        self.curr_device = None
        self.loss_array = loss_array
        self.batch_len = batch_len
        self.threshold_v_top = 0
        self.threshold_v_bottom = 0
        self.last_mean = 1
        self.x_factor = 2.0
        self.is_mixup = is_mixup
        self.is_regression = is_regression
        self.is_camloss = is_camloss

    def forward(self, input: Tensor, input2: Tensor, input3: Tensor, **kwargs) -> Tensor:
        return self.model(input, input2, input3, **kwargs)

    def training_step(self, batch, batch_idx):
        real_img, real_img2, real_img3, labels = batch
        self.curr_device = real_img.device
        if self.is_regression:
            labels = labels.view(-1, 1)
        else:
            labels = labels - 1
        outputs = self.forward(real_img, real_img2, real_img3)
        train_loss = self.loss_function(outputs, labels)
        if self.is_regression:
            predicts = torch.round(outputs)
        else:
            predicts = torch.argmax(outputs, dim=1)
        accuracy = torch.sum(predicts == labels) / len(labels)
        self.log_dict({"train_loss": train_loss.item(), "train_accuracy": accuracy.item()}, sync_dist=True)
        """if train_loss.item() < 0.2:
            loss_unit = []
            for i in range(len(outputs)):
                loss_unit.append(self.loss_function(outputs[i].reshape(1, -1), labels[i].reshape(1)).item())
            loss_array = np.array(loss_unit)
            if np.std(loss_array) > 0.15:
                fin_loss = train_loss * 0.3"""
        return train_loss

    def validation_step(self, batch, batch_idx):
        real_img, real_img2, real_img3, labels = batch
        self.curr_device = real_img.device
        if self.is_regression:
            labels = labels.view(-1, 1)
        else:
            labels = labels - 1
        outputs = self.forward(real_img, real_img2, real_img3)
        val_loss = self.loss_function(outputs, labels)
        if self.is_regression:
            predicts = torch.round(outputs)
        else:
            predicts = torch.argmax(outputs, dim=1)
        accuracy = torch.sum(predicts == labels) / len(labels)
        self.log_dict({"val_loss": val_loss.item(), "val_accuracy": accuracy.item()}, on_step=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        real_img, real_img2, real_img3, labels = batch
        self.curr_device = real_img.device
        if self.is_regression:
            labels = labels.view(-1, 1)
        else:
            labels = labels - 1
        outputs = self.forward(real_img, real_img2, real_img3)
        test_loss = self.loss_function(outputs, labels)
        predicts = torch.argmax(outputs, dim=1)
        accuracy = torch.sum(predicts == labels) / len(labels)
        self.log_dict({"test_loss": test_loss.item(), "test_accuracy": accuracy.item()}, on_step=True)

    def configure_optimizers(self):

        optims = []
        scheds = []
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])

        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model, self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

            try:
                if self.params['scheduler_gamma'] is not None:
                    scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                                 gamma=self.params['scheduler_gamma'])
                    scheds.append(scheduler)

                    # Check if another scheduler is required for the second optimizer
                    try:
                        if self.params['scheduler_gamma_2'] is not None:
                            scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                          gamma=self.params['scheduler_gamma_2'])
                            scheds.append(scheduler2)
                    except:
                        pass
                    return optims, scheds
            except:
                return optims

        """optims = []
        optimizer = optim.SGD(self.model.parameters(),
                                    lr=5e-4,  # self.params['LR']
                                    momentum=0.9,
                                    weight_decay=0.0005  # self.params['weight_decay']
                                    )

        optims.append(optimizer)
        return optims"""

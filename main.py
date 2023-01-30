from model import Model
from train import Train
from test import Test
import torch
import torch.optim as optim
from data_loader import DataLoader
import utils as ut

train_loader, test_loader = DataLoader().return_loaders()
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
EPOCHS = 1


def train_bn_l1():
    group_norm = False
    layer_norm = False
    model = Model(group_norm, layer_norm).to(device)
    ut.print_model_summary(model)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    train = Train()
    test = Test()
    for epoch in range(EPOCHS):
        print("EPOCH:", epoch)
        train.train_model(model, device, train_loader, optimizer, epoch, L1=True)
        test_losses, test_acc, pred, target, data = test.test_model(model, device, test_loader)
    return test_losses, test_acc, pred, target, data


def train_gn():
    group_norm = True
    layer_norm = False
    model = Model(group_norm, layer_norm).to(device)
    ut.print_model_summary(model)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    train = Train()
    test = Test()
    for epoch in range(EPOCHS):
        print("EPOCH:", epoch)
        train.train_model(model, device, train_loader, optimizer, epoch, L1=False)
        test_losses, test_acc, pred, target, data = test.test_model(model, device, test_loader)
    return test_losses, test_acc, pred, target, data


def train_ln():
    group_norm = False
    layer_norm = True
    model = Model(group_norm, layer_norm).to(device)
    ut.print_model_summary(model)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    train = Train()
    test = Test()
    for epoch in range(EPOCHS):
        print("EPOCH:", epoch)
        train.train_model(model, device, train_loader, optimizer, epoch, L1=False)
        test_losses, test_acc, pred, target, data = test.test_model(model, device, test_loader)
    return test_losses, test_acc, pred, target, data


bn_test_losses, bn_test_acc, bn_pred, bn_target, bn_data = train_bn_l1()
gn_test_losses, gn_test_acc, gn_pred, gn_target, gn_data = train_gn()
ln_test_losses, ln_test_acc, ln_pred, ln_target, ln_data = train_ln()


ut.draw_train_test_acc_loss(bn_test_losses, bn_test_acc, gn_test_losses, gn_test_acc, ln_test_losses, ln_test_acc)
ut.draw_misclassified_images(bn_pred, bn_target, bn_data, "Batch Normalization with L1")
ut.draw_misclassified_images(gn_pred, gn_target, gn_data, "Group Normalization")
ut.draw_misclassified_images(ln_pred, ln_target, ln_data, "Layer Normalization")
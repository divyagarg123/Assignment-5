import torch
from torchsummary import summary
from model import Model
import matplotlib.pyplot as plt


def check_for_cuda():
    SEED = 1
    cuda = torch.cuda.is_available()
    print("CUDA Available", cuda)
    torch.manual_seed(SEED)

    if cuda:
        torch.cuda.manual_seed(SEED)
    return cuda


def print_model_summary(model):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    summary(model, input_size=(1, 28, 28))


def draw_train_test_acc_loss(bn_test_losses, bn_test_acc, gn_test_losses, gn_test_acc, ln_test_losses, ln_test_acc):
    fig, axs = plt.subplots(3, 2, figsize=(15, 10))
    axs[0, 0].plot(bn_test_losses)
    axs[0, 0].set_title("BN Test Loss")
    axs[0, 1].plot(bn_test_acc)
    axs[0, 1].set_title("BN Test Accuracy")

    axs[1, 0].plot(gn_test_losses)
    axs[1, 0].set_title("GN Test Loss")
    axs[1, 1].plot(gn_test_acc)
    axs[1, 1].set_title("GN Test Accuracy")

    axs[2, 0].plot(ln_test_losses)
    axs[2, 0].set_title("LN Test Loss")
    axs[2, 1].plot(ln_test_acc)
    axs[2, 1].set_title("LN Test Accuracy")


def draw_misclassified_images(pred, target, data, main_title):
    fig = plt.figure(figsize=(10, 8))
    index = 1
    for i in range(len(target)):
        plt.subplot(5, 2, index)
        plt.axis('off')
        plt.imshow(data[i].squeeze(), cmap='gray_r')
        title = "Target=" + str(target[i]) + "  Pred=" + str(pred[i])
        plt.gca().set_title(title)
        index += 1
    fig.suptitle(main_title, size=20)
    plt.show()

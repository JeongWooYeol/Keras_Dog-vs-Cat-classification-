import numpy as np
import torch, torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, ConcatDataset
import torch.nn as nn
import os
import data
import model

def train(dataloader, model):
    device = 'cuda'
    print(model)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500, 1000, 1500], gamma=0.5)

    epochs = 50
    itr = 1
    p_itr = 200
    model.train()
    total_loss = 0
    loss_list = []
    acc_list = []
    for epoch in range(epochs):
        for samples, labels in dataloader:
            samples, labels = samples.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(samples)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            scheduler.step()

            if itr % p_itr == 0:
                pred = torch.argmax(output, dim=1)
                correct = pred.eq(labels)
                acc = torch.mean(correct.float())
                print(
                    '[Epoch {}/{}] Iteration {} -> Train Loss: {:.4f}, Accuracy: {:.3f}'.format(epoch + 1, epochs, itr,
                                                                                                total_loss / p_itr,
                                                                                                acc))
                loss_list.append(total_loss / p_itr)
                acc_list.append(acc)
                total_loss = 0

            itr += 1

    return model


if __name__ == "__main__":

    train_dir = './train/train/'
    test_dir = './test1/test1/'
    train_files = os.listdir(train_dir)
    test_files = os.listdir(test_dir)
    batch_size = 128

    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ColorJitter(),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(128),
        transforms.ToTensor()
    ])

    cat_files = [tf for tf in train_files if 'cat' in tf]
    dog_files = [tf for tf in train_files if 'dog' in tf]

    cats = data.CatDogDataset(cat_files, train_dir, transform=data_transform)
    dogs = data.CatDogDataset(dog_files, train_dir, transform=data_transform)

    catdogs = ConcatDataset([cats, dogs])

    dataloader = DataLoader(catdogs, batch_size=32, shuffle=True, num_workers=4)

    samples, labels = iter(dataloader).next()
    plt.figure(figsize=(16, 24))
    grid_imgs = torchvision.utils.make_grid(samples[:24])
    np_grid_imgs = grid_imgs.numpy()
    # in tensor, image is (batch, width, height), so you have to transpose it to (width, height, batch) in numpy to show it.
    plt.imshow(np.transpose(np_grid_imgs, (1, 2, 0)))

    best_model = train(dataloader, model.resnet18())
    torch.save(best_model.state_dict(), "ResNet18_CatDog.pth")


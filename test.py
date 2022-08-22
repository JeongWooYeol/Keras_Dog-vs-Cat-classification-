import pandas as pd
import numpy as np
import torch, torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os
from data import CatDogDataset
import model


if __name__ == '__main__':
    filename_pth = 'ResNet18_CatDog.pth'
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((128, 128)),
        torchvision.transforms.ToTensor()
    ])
    test_dir = './test1/test1/'
    test_files = os.listdir(test_dir)

    testset = CatDogDataset(test_files, test_dir, mode = 'test', transform = test_transform)
    test_dl = DataLoader(testset, batch_size=32, shuffle=False, num_workers=4)

    model = model.resnet18().cuda()
    model.load_state_dict((torch.load(filename_pth, map_location = torch.device('cuda'))), strict = False)
    print(model)

    device = 'cuda'
    model.eval()

    fn_list = []
    pred_list = []
    for x, fn in test_dl:
        with torch.no_grad():
            x = x.to(device)
            output = model(x)
            pred = torch.argmax(output, dim=1)
            fn_list += [n[:-4] for n in fn]
            pred_list += [p.item() for p in pred]

    submission = pd.DataFrame({"id": fn_list, "label": pred_list})
    submission.to_csv('preds_ResNet18.csv', index=False)

    samples, _ = iter(test_dl).next()
    samples = samples.to(device)
    fig = plt.figure(figsize=(24, 16))
    fig.tight_layout()
    output = model(samples[:24])
    pred = torch.argmax(output, dim=1)
    pred = [p.item() for p in pred]
    ad = {0: 'cat', 1: 'dog'}
    for num, sample in enumerate(samples[:24]):
        plt.subplot(4, 6, num + 1)
        plt.title(ad[pred[num]])
        plt.axis('off')
        sample = sample.cpu().numpy()
        plt.imshow(np.transpose(sample, (1, 2, 0)))
    plt.show()
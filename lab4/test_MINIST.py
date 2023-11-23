import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as data_utils
import torchvision.datasets as dataset
import torchvision.transforms as transforms

from network import NET

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, default=r'model/best.pth')
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()

    test_data = dataset.MNIST(root="mnist",
                                   train=False,
                                   transform=transforms.ToTensor(),
                                   download=False)
    test_dataloader = data_utils.DataLoader(dataset=test_data, shuffle=False, batch_size=args.batch_size)

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = NET().to(device)

    model.load_state_dict(torch.load(args.weights_file))
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    model.eval()
    acc_num = 0

    for data in test_dataloader:
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            preds = model(inputs)
            preds = preds.argmax(dim=1)

        acc_num += (preds == labels).sum()

    print('test accuracy: {:.2%}'.format(acc_num/len(test_data)))

import os
import torch
import time
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader
from models import *
from dataset import prepare_data, Dataset
from utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

parser = argparse.ArgumentParser(description="PPNet")
parser.add_argument("--block_nums", type=int, default=3, help='numbers of blocks')
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batch_size", type=int, default=256, help="Training batch size")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=20, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
parser.add_argument("--mode", type=str, default="S", help='with known noise level (S) or blind training (B)')
parser.add_argument("--noiseL", type=int, default=25, help='noise level; ignored when mode=B')
parser.add_argument("--val_noiseL", type=int, default=25, help='noise level used on validation set')
opt = parser.parse_args()


def main():
    # Load dataset
    print('Loading dataset ...\n')
    dataset_train = Dataset(train=True, mode=opt.mode)
    dataset_val = Dataset(train=False, mode=opt.mode)
    loader_train = DataLoader(dataset=dataset_train, batch_size=opt.batch_size, shuffle=True, num_workers=4)
    print("# of training samples: %d\n" % int(len(dataset_train)))
    # Build model
    net = PPNet(in_channels=3, block_nums=opt.block_nums)

    criterion = my_loss()

    model = net.to("cuda")
    device_ids = [0, 1]
    model = nn.DataParallel(model, device_ids=device_ids)
    if opt.mode == "S":
        base_path = os.path.join(opt.outf, f"{opt.mode}{opt.noiseL}", f"block{opt.block_nums}")
    else:
        base_path = os.path.join(opt.outf, opt.mode, f"block{opt.block_nums}")

    if os.path.exists(os.path.join(base_path, "checkpoint.txt")):
        with open(os.path.join(base_path, "checkpoint.txt"), 'r') as f1:
            checkpoint = int(f1.read())
            print("Resuming from checkpoint %d" % checkpoint)
            model.load_state_dict(torch.load(os.path.join(base_path, "net.pth")))
    else:
        os.makedirs(base_path)
        checkpoint = -1

    with open(f'{base_path}/psnr_val.txt', 'a') as f:
        f.write(f'Block Nums: {opt.block_nums}, Total Parameters: {sum(x.numel() for x in net.parameters())}, Noise Level: {opt.noiseL}, Val Noise Level: {opt.val_noiseL}')
        f.write('\n')

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    # training
    step = 0
    noiseL_B = [0, 55] # ingnored when opt.mode=='S'

    best_psnr = 0

    for epoch in range(checkpoint + 1, opt.epochs):
        start = time.time()
        if epoch <= opt.milestone:
            current_lr = opt.lr
        elif epoch > opt.milestone and epoch <= 40:
            current_lr = opt.lr / 10.
        elif epoch > 40 and epoch <= 50:
            current_lr = opt.lr / 100.
        else:
            current_lr = opt.lr / 1000.

        # set learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('learning rate %f' % current_lr)
        # train
        for i, img_train in enumerate(loader_train, 0):
            # training step
            model.train()
            optimizer.zero_grad()
            if opt.mode == 'S':
                noise = torch.FloatTensor(img_train.size()).normal_(mean=0, std=opt.noiseL/255.)
            elif opt.mode == 'B':
                noise = torch.zeros(img_train.size())
                stdN = np.random.uniform(noiseL_B[0], noiseL_B[1], size=noise.size()[0])
                for n in range(noise.size()[0]):
                    sizeN = noise[0, :, :, :].size()
                    noise[n, :, :, :] = torch.FloatTensor(sizeN).normal_(mean=0, std=stdN[n]/255.)

            imgn_train = img_train + noise
            img_train, imgn_train = img_train.cuda(), imgn_train.cuda()
            out_train = model(imgn_train)
            loss = criterion(out_train, img_train)
            loss.backward()
            optimizer.step()
            # results
            model.eval()
            with torch.no_grad():
                out_train = torch.clamp(model(imgn_train), 0., 1.)
                psnr_train = batch_PSNR(out_train, img_train, 1.)
                print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" %
                    (epoch, i+1, len(loader_train), loss.item(), psnr_train))

            step += 1
        ## the end of each epoch
        model.eval()
        # validate
        psnr_val = 0
        with torch.no_grad():
            for k in range(len(dataset_val)):
                img_val = torch.unsqueeze(dataset_val[k], 0)
                torch.manual_seed(11)  # set the seed
                noise = torch.FloatTensor(img_val.size()).normal_(mean=0, std=opt.val_noiseL/255.)
                imgn_val = img_val + noise
                img_val, imgn_val = img_val.cuda(), imgn_val.cuda()
                out_val = torch.clamp(model(imgn_val), 0., 1.)
                psnr_val += batch_PSNR(out_val, img_val, 1.)
            psnr_val /= len(dataset_val)
            print("\n[epoch %d] PSNR_val: %.4f" % (epoch, psnr_val))
            with open(f'{base_path}/psnr_val.txt', 'a') as f:
                f.write(f'Epoch: {epoch}, psnr: {psnr_val}, time usage: {time.time() - start}')
                f.write('\n')

        if psnr_val > best_psnr:
            best_psnr = psnr_val
            with open(os.path.join(base_path, "checkpoint.txt"), 'w') as f1:
                f1.write(str(epoch))
            # save model
            torch.save(model.state_dict(), os.path.join(base_path, 'net.pth'))


if __name__ == "__main__":
    if opt.preprocess:
        if opt.mode == 'S':
            # adnet
            prepare_data(data_path='../../data', patch_size=50, stride=40, aug_times=1, mode=opt.mode)
        if opt.mode == 'B':
            prepare_data(data_path='../../data', patch_size=50, stride=10, aug_times=2, mode=opt.mode)
    main()

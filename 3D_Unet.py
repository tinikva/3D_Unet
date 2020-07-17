import matplotlib
matplotlib.use('Agg')
import numpy as np  
import pandas as pd  
np.random.seed(1234)
import torch
from torch.autograd import Variable
#from imageio import imread, imsave
from torch.nn import functional as F
from torch.nn import init
from skimage.morphology import label
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import sys
import os
import pickle
import time
import skimage.morphology as mph
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
from skimage import io


# Type in output folder, epoch number, and initial learning rate
output = sys.argv[1]
eps = sys.argv[2]
LR = sys.argv[3]

# Make directory if not exist
if not os.path.exists('../' + output):
    os.makedirs('../' + output)
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def label_converter(mask_data):
    new_masks = np.zeros(shape=mask_data.shape, dtype=np.uint8)
    for i in range(len(mask_data)):
        new_masks[i] = np.where(mask_data[i] > 0, 1, 0)

    return new_masks

def dataloader(handles, mode = 'train'):
    
    images = {}
    images['Image'] = []
    images['Label'] = []
    images['Gap'] = []
    images['ID'] = []
    
    for idx, row in handles.iterrows():
        im = io.imread(row['Image'])
        images['ID'].append(row['ID'])
        if mode != 'test':
            la = io.imread(row['Label'])
            la = label_converter(la)
            gap = io.imread(row['Gap'])
            images['Label'].append(lb)
            images['Gap'].append(gap)
        
        im = im / 255 # Normalization
         #convert labels
        images['Image'].append(im)  

    return images

# UNet model
class UNet_down_block(torch.nn.Module):
    def __init__(self, input_channel, output_channel, down_size):
        super(UNet_down_block, self).__init__()
        self.conv1 = torch.nn.Conv3d(input_channel, output_channel, 3, padding=(2,1,1))
        self.bn1 = torch.nn.BatchNorm3d(output_channel)
        self.conv2 = torch.nn.Conv3d(output_channel, output_channel, 3, padding=(2,1,1))
        self.bn2 = torch.nn.BatchNorm3d(output_channel)
        self.conv3 = torch.nn.Conv3d(output_channel, output_channel, 3, padding=(2,1,1))
        self.bn3 = torch.nn.BatchNorm3d(output_channel)
        self.max_pool = torch.nn.MaxPool3d(2, 2)
        self.relu = torch.nn.ReLU()
        self.down_size = down_size

    def forward(self, x):
        if self.down_size:
            x = self.max_pool(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        return x


class UNet_up_block(torch.nn.Module):
    def __init__(self, prev_channel, input_channel, output_channel):
        super(UNet_up_block, self).__init__()
        self.up_sampling = torch.nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv1 = torch.nn.Conv3d(prev_channel + input_channel, output_channel, 3, padding=(2,1,1))
        self.bn1 = torch.nn.BatchNorm3d(output_channel)
        self.conv2 = torch.nn.Conv3d(output_channel, output_channel, 3, padding=(2,1,1))
        self.bn2 = torch.nn.BatchNorm3d(output_channel)
        self.conv3 = torch.nn.Conv3d(output_channel, output_channel, 3, padding=(2,1,1))
        self.bn3 = torch.nn.BatchNorm3d(output_channel)
        self.relu = torch.nn.ReLU()

    def forward(self, prev_feature_map, x):
        x = self.up_sampling(x)
        x = torch.cat((x, prev_feature_map), dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        return x


class UNet(torch.nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.down_block1 = UNet_down_block(3, 16, False)
        self.down_block2 = UNet_down_block(16, 32, True)
        self.down_block3 = UNet_down_block(32, 64, True)
        self.down_block4 = UNet_down_block(64, 128, True)
        self.down_block5 = UNet_down_block(128, 256, True)
        self.down_block6 = UNet_down_block(256, 512, True)
        self.down_block7 = UNet_down_block(512, 1024, True)

        self.mid_conv1 = torch.nn.Conv3d(1024, 1024, 3, padding=(2,1,1))
        self.bn1 = torch.nn.BatchNorm3d(1024)
        self.mid_conv2 = torch.nn.Conv3d(1024, 1024, 3, padding=(2,1,1))
        self.bn2 = torch.nn.BatchNorm3d(1024)
        self.mid_conv3 = torch.nn.Conv3d(1024, 1024, 3, padding=(2,1,1))
        self.bn3 = torch.nn.BatchNorm3d(1024)

        self.up_block1 = UNet_up_block(512, 1024, 512)
        self.up_block2 = UNet_up_block(256, 512, 256)
        self.up_block3 = UNet_up_block(128, 256, 128)
        self.up_block4 = UNet_up_block(64, 128, 64)
        self.up_block5 = UNet_up_block(32, 64, 32)
        self.up_block6 = UNet_up_block(16, 32, 16)

        self.last_conv1 = torch.nn.Conv3d(16, 16, 3, padding=(2,1,1))
        self.last_bn = torch.nn.BatchNorm3d(16)
        self.last_conv2 = torch.nn.Conv3d(16, 1, 1, padding=0)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        self.x1 = self.down_block1(x)
        self.x2 = self.down_block2(self.x1)
        self.x3 = self.down_block3(self.x2)
        self.x4 = self.down_block4(self.x3)
        self.x5 = self.down_block5(self.x4)
        self.x6 = self.down_block6(self.x5)
        self.x7 = self.down_block7(self.x6)
        self.x7 = self.relu(self.bn1(self.mid_conv1(self.x7)))
        self.x7 = self.relu(self.bn2(self.mid_conv2(self.x7)))
        self.x7 = self.relu(self.bn3(self.mid_conv3(self.x7)))
        x = self.up_block1(self.x6, self.x7)
        x = self.up_block2(self.x5, x)
        x = self.up_block3(self.x4, x)
        x = self.up_block4(self.x3, x)
        x = self.up_block5(self.x2, x)
        x = self.up_block6(self.x1, x)
        x = self.relu(self.last_bn(self.last_conv1(x)))
        x = self.last_conv2(x)
        return x
    
# Initial weights
def init_weights(module):
    for name, param in module.named_parameters():
        if name.find('weight') != -1:
            if len(param.size()) == 1:
                init.uniform(param.data, 1).type(torch.DoubleTensor).to(device)
            else:
                init.xavier_uniform(param.data).type(torch.DoubleTensor).to(device)
        elif name.find('bias') != -1:
            init.constant(param.data, 0).type(torch.DoubleTensor).to(device)

# Early stop function
def losscp (list):
    newlist = np.sort(list)
    if np.array_equal(np.array(list), np.array(newlist)):
        return 1
    else:
        return 0
    
# Dice loss function
def dice_loss(input, target):
    smooth = 1.
    iflat = input.view(-1).cpu()
    tflat = target.view(-1).cpu()
    intersection = (iflat * tflat).sum()
    return 1.0 - (((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)))


# F1 score
def Fscore(y_pred, target):
    pred = (y_pred.view(-1) > 0.5).type(torch.FloatTensor).to(device)
    target_vec = target.view(-1).type(torch.FloatTensor).to(device)
    label = target_vec.sum().cpu().data.numpy()
    tp = (pred * target_vec).sum().cpu().data.numpy()
    predicted = pred.sum().cpu().data.numpy()
    recall = tp / predicted
    precision = tp / label
    F = 2 * precision * recall / (precision + recall)
    return F


# PPV metric function
def metric(y_pred, target):
    pred = (y_pred.view(-1) > 0.5).type(torch.FloatTensor).to(device)
    target_vec = target.view(-1).type(torch.FloatTensor).to(device)
    label = target_vec.sum().cpu().data.numpy()
    tp = (pred * target_vec).sum().cpu().data.numpy()
    predicted = pred.sum().cpu().data.numpy()
    ppv = (tp) / (predicted + label - tp)
    return ppv

# Training and validation method
def train(bs, sample, vasample, ep, ilr, mode):
    # Initialize learning rate decay and learning rate
    # model
    model = UNet()
    # initialize weight
    init_weights(model)
    model.to(device) 
    # optimizer
    opt = torch.optim.Adam(model.parameters(), lr=ilr)
    opt.zero_grad()
    # train and validation samples
    rows_trn = len(sample['Label'])
    rows_val = len(vasample['Label'])
    # Batch per epoch
    batches_per_epoch = rows_trn // bs
    losslists = []
    vlosslists = []
    Fscorelist = []
    PPVlist = []
    
    #data, target = data.to(device), target.to(device)
        
    for epoch in range(ep):
        print(epoch, flush=True)
        # Learning rate
        #lr = init_lr * lr_dec
        order = np.arange(rows_trn)
        losslist = []
        tr_metric_list = []
        va_metric_list = []
        tr_F_list = []
        va_F_list = []
        for itr in range(batches_per_epoch):
            rows = order[itr * bs: (itr + 1) * bs]
            if itr + 1 == batches_per_epoch:
                rows = order[itr * bs:]
            # read in a batch
            trim = sample['Image'][rows[0]]
            trla = sample['Label'][rows[0]]
            trga = sample['Gap'][rows[0]]
            if mode == 'nuke':
                # Cuda and tensor inputs and label
                x = Variable(torch.from_numpy(trim).type(torch.FloatTensor)).to(device)
                y = Variable(torch.from_numpy(trla).type(torch.FloatTensor)).to(device)

                # Prediction
                pred_mask = model(x)
                # BCE and dice loss
                loss = loss_fn(pred_mask, y).cpu() + dice_loss(F.sigmoid(pred_mask), y)
                losslist.append(loss.data.numpy())
                loss.backward()
                # ppv metric
                tr_metric = metric(F.sigmoid(pred_mask), y)
                tr_metric_list.append(tr_metric)
                tr_F = Fscore(F.sigmoid(pred_mask), y)
                tr_F_list.append(tr_F)
            elif mode == 'gap':
                # Cuda and tensor inputs and label
                x = Variable(torch.from_numpy(trim).type(torch.FloatTensor)).to(device)
                y = Variable(torch.from_numpy(trga / 255).type(torch.FloatTensor)).to(device)
                # Prediction
                pred_mask = model(x)
                # BCE and dice loss
                loss = loss_fn(pred_mask, y).cpu() + dice_loss(F.sigmoid(pred_mask), y)
                losslist.append(loss.data.numpy())
                loss.backward()
                # ppv metric
                tr_metric = metric(F.sigmoid(pred_mask), y)
                tr_metric_list.append(tr_metric)
                tr_F = Fscore(F.sigmoid(pred_mask), y)
                tr_F_list.append(tr_F)
            opt.step()
            opt.zero_grad()

        vlosslist = []
        # For validation set
        for itr in range(rows_val):
            vaim = vasample['Image'][itr]
            vala = vasample['Label'][itr]
            vaga = vasample['Gap'][itr]
            if mode == 'nuke':         
            # cuda and tensor sample
                xv = Variable(torch.from_numpy(vaim).type(torch.FloatTensor)).to(device)
                yv = Variable(torch.from_numpy(vala).type(torch.FloatTensor)).to(device)
                # prediction
                pred_maskv = model(xv)
                # dice and BCE loss
                vloss = loss_fn(pred_maskv, yv).cpu() + dice_loss(F.sigmoid(pred_maskv), yv)
                vlosslist.append(vloss.data.numpy())
                # ppv metric
                va_metric = metric(F.sigmoid(pred_maskv), yv)
                va_metric_list.append(va_metric)
                va_F = Fscore(F.sigmoid(pred_maskv), yv)
                va_F_list.append(va_F)
            elif mode == 'gap':
                # cuda and tensor sample
                xv = Variable(torch.from_numpy(vaim).type(torch.FloatTensor)).to(device)
                yv = Variable(torch.from_numpy(vaga / 255).type(torch.FloatTensor)).to(device)
                # prediction
                pred_maskv = model(xv)
                # dice and BCE loss
                vloss = loss_fn(pred_maskv, yv).cpu() + dice_loss(F.sigmoid(pred_maskv), yv)
                vlosslist.append(vloss.data.numpy())
                # ppv metric
                va_metric = metric(F.sigmoid(pred_maskv), yv)
                va_metric_list.append(va_metric)
                va_F = Fscore(F.sigmoid(pred_maskv), yv)
                va_F_list.append(va_F)
        lossa = np.mean(losslist)
        vlossa = np.mean(vlosslist)
        tr_score = np.mean(tr_metric_list)
        va_score = np.mean(va_metric_list)
        tr_F_list = np.nan_to_num(tr_F_list)
        va_F_list = np.nan_to_num(va_F_list)
        tr_Fscore = np.mean(tr_F_list)
        va_Fscore = np.mean(va_F_list)
        # Print epoch summary
        print(
            'Epoch {:>3} |lr {:>1.5f} | Loss {:>1.5f} | VLoss {:>1.5f} | Train F1 {:>1.5f} | Val F1 {:>1.5f} | Train PPV {:>1.5f} | Val PPV {:>1.5f}'.format(
                epoch + 1, lr, lossa, vlossa, tr_Fscore, va_Fscore, tr_score, va_score))
        losslists.append(lossa)
        vlosslists.append(vlossa)
        Fscorelist.append(va_Fscore)
        PPVlist.append(va_score)

        for param_group in opt.param_groups:
            param_group['lr'] = lr
        # Save models
        if vlossa == np.min(vlosslists):
            print('Min loss found:')
            print(vlossa)
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': opt.state_dict(),
            }
            torch.save(checkpoint, '../' + output + '/' + mode + 'loss_unet')
        if va_Fscore == np.max(Fscorelist):
            print('Max F found:')
            print(va_Fscore)
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': opt.state_dict(),
            }
            torch.save(checkpoint, '../' + output + '/' + mode + 'F_unet')

        if va_score == np.max(PPVlist):
            print('Max PPV found:')
            print(va_score)
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': opt.state_dict(),
            }
            torch.save(checkpoint, '../' + output + '/' + mode + 'PPV_unet')

        # if no change or increase in loss for consecutive 6 epochs, decrease learning rate by 10 folds
        if epoch > 6:
            if losscp(losslists[-5:]) or losscp(vlosslists[-5:]):
                lr_dec = lr_dec / 10
        # if no change or increase in loss for consecutive 15 epochs, save validation predictions and stop training
        if epoch > 15:
            if losscp(losslists[-15:]) or losscp(vlosslists[-15:]) or epoch+1 == ep:
                for itr in range(rows_val):
                    vaim = vasample['Image'][itr]
                    xv = Variable(torch.from_numpy(vaim).type(torch.FloatTensor)).to(device)
                    pred_maskv = model(xv)
                    pred_np = (F.sigmoid(pred_maskv)).cpu().data.numpy()
                    ppp = pred_np[0,0,:,:]
                    pred_np = pred_np.round().astype(np.uint8)
                    pred_np = pred_np[0, 0, :, :]
                    pww = pred_np
                    if not os.path.exists('../' + output + '/' + mode + 'validation/'):
                        os.makedirs('../' + output + '/' + mode + 'validation/')
                    if mode == 'nuke':
                        pred_np = mph.remove_small_objects(pred_np.astype(bool), min_size=30,
                                                           connectivity=2).astype(np.uint8)
                        pred_np = mph.remove_small_holes(pred_np.astype(bool), min_size=30, connectivity=2)
                    elif mode == 'gap':
                        pred_np = mph.remove_small_objects(pred_np.astype(bool), min_size=15,
                                                           connectivity=2).astype(np.uint8)
                        selem = mph.disk(1)
                        pred_np = mph.erosion(pred_np, selem)
                    if np.max(pred_np) == np.min(pred_np):
                        print('1st_BOOM!')
                        print(vasample['ID'][itr])
                        if np.max(pww) == np.min(pww):
                            print('2nd_BOOM!')
                            if ppp.max() == 0 or ppp.min() == 1:
                                print('3rd_BOOM!')
                                save_image('../' + output + '/' + mode + 'validation/' + vasample['ID'][itr] + '.png',
                                       ppp.astype(np.uint8))
                            else:
                                ppp = (ppp / ppp.max()) * 1
                                ppp = (ppp > 0.95).astype(np.uint8)
                                save_image('../' + output + '/' + mode + 'validation/' + vasample['ID'][itr] + '.png',
                                       ((ppp / ppp.max()) * 255).astype(np.uint8))
                        else:
                            save_image('../' + output + '/' + mode + 'validation/' + vasample['ID'][itr] + '.png',
                                   ((pww / pww.max()) * 255).astype(np.uint8))
                    else:
                        save_image('../' + output + '/' + mode + 'validation/' + vasample['ID'][itr] + '.png',
                               ((pred_np / pred_np.max()) * 255).astype(np.uint8))
                break

    # Loss figures
    plt.plot(losslists)
    plt.plot(vlosslists)
    plt.title('Train & Validation Loss')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.savefig('../' + output + '/'+mode+'_loss.png')


def vatest(vasample):
    if not os.path.exists('../' + output + '/validation'):
        os.makedirs('../' + output + '/validation')
    for itr in range(len(vasample['ID'])):
        vaid = vasample['ID'][itr]
        a = io.imread('../' + output + '/nukevalidation/' + vaid + '.png')
        b = io.imread('../' + output + '/gapvalidation/' + vaid + '.png')
        out = np.clip(a - b, 0, None)
        out = mph.remove_small_objects(out, min_size=30, connectivity=1)
        out = mph.remove_small_holes(out, min_size=30, connectivity=2)
        save_image('../' + output + '/validation/' + vaid + '_pred.png',
               ((out / out.max()) * 255).astype(np.uint8))


def cbtest(tesample, group):
    test_ids = []
    rles = []
    if not os.path.exists('../' + output + '/final_' + group):
        os.makedirs('../' + output + '/final_' + group)
    for itr in range(len(tesample['ID'])):
        teid = tesample['ID'][itr]
        a = io.imread('../' + output + '/' + group + '/' + teid + '_nuke_pred.png')
        b = io.imread('../' + output + '/' + group + '/' + teid + '_gap_pred.png')
        out = np.clip(a - b, 0, None)
        out = mph.remove_small_objects(out, min_size=30, connectivity=1)
        out = mph.remove_small_holes(out, min_size=30, connectivity=2)
        out = ((out / out.max()) * 255).astype(np.uint8)
        save_image('../' + output + '/final_' + group + '/' + teid + '_pred.png',
               ((out / out.max()) * 255).astype(np.uint8))
        # vectorize mask
        rle = list(prob_to_rles(out))
        rles.extend(rle)
        test_ids.extend([teid] * len(rle))
    # save vectorize masks as CSV
    sub = pd.DataFrame()
    sub['ImageId'] = test_ids
    sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))

    return sub

if __name__ == '__main__':
    # Read in files containing paths to training, validation, and testing images
    tr = pd.read_csv('samples_train.csv', header=0,
                           usecols=['Type', 'Image', 'Label', 'Gap', 'Width', 'Height','Depth', 'ID'])
 
    va = pd.read_csv('samples_val.csv', header=0,
                           usecols=['Type', 'Image', 'Label', 'Gap', 'Width', 'Height', 'Depth', 'ID'])
   
    # Load in images
    trsample = dataloader(tr, 'train')
    vasample = dataloader(va, 'val')

    # training
    train(1, trsample, vasample, int(eps), float(LR), 'nuke')
    train(1, trsample, vasample, int(eps), float(LR), 'gap')
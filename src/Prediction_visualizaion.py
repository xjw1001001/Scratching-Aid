
import argparse
from datetime import datetime
from torch.utils.data import DataLoader
import torch.optim as optim
from mouse_data_prediction import *
import crnn
import numpy as np
import grad_cam
from common_tools import *
import pandas as pd
from typing import List
import torch

def saliency(input_tensor, model):
    # we don't need gradients w.r.t. weights for a trained model
    for param in model.parameters():
        param.requires_grad = False

    # set model in eval mode
    model.eval()
    # transoform input PIL image to torch.Tensor and normalize
    input = input_tensor

    # we want to calculate gradient of higest score w.r.t. input
    # so set requires_grad to True for input
    input.requires_grad = True
    # forward pass to calculate predictions
    preds = model(input)
    score, indices = torch.max(preds, 1)
    # backward pass to get gradients of score predicted class w.r.t. input image
    score.backward()
    # get max along channel axis
    slc, _ = torch.max(torch.abs(input.grad[0]), dim=0)
    # normalize to [0..1]
    slc = (slc - slc.min()) / (slc.max() - slc.min())

    # apply inverse transform on image
    with torch.no_grad():
        input_img = inv_normalize(input[0])
    # plot image and its saleincy map
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(np.transpose(input_img.detach().numpy(), (1, 2, 0)))
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1, 2, 2)
    plt.imshow(slc.numpy(), cmap=plt.cm.hot)
    plt.xticks([])
    plt.yticks([])
    plt.show()

def saliency11(input_tensor, model):
    # we don't need gradients w.r.t. weights for a trained model
    for param in model.parameters():
        param.requires_grad = False

    # set model in eval mode
    model.eval()
    # transoform input PIL image to torch.Tensor and normalize
    input = input_tensor

    # we want to calculate gradient of higest score w.r.t. input
    # so set requires_grad to True for input
    input.requires_grad = True
    # forward pass to calculate predictions
    preds = model(input)
    score, indices = torch.max(preds, 1)
    # backward pass to get gradients of score predicted class w.r.t. input image
    score.backward()
    # get max along channel axis
    slc, _ = torch.max(torch.abs(input.grad[0]), dim=1)
    # normalize to [0..1]
    slc = (slc - slc.min()) / (slc.max() - slc.min())

    return slc.numpy()

def label_lis(dat_frame):
    dat_frame = dat_frame.reset_index()
    label_lis = []
    for i in range(len(dat_frame)):
        label_lis=list(range(int(dat_frame['Start_frame'][i]),int(dat_frame['End_frame'][i])+1))+label_lis
    label_lis.sort(reverse=False)
    return label_lis

if __name__ == "__main__":

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    dataset_params = {
        'batch_size': 32,
        'shuffle': True,
        'num_workers': 4,
        'pin_memory': True,
        'frame_step': 1
    }

    cnn_encoder_params = {
        'cnn_out_dim': 256,  # 256
        'drop_prob': 0.20,
        'bn_momentum': 0.01
    }

    rnn_decoder_params = {
        'use_gru': True,
        'cnn_out_dim': 256,  # 256
        'rnn_hidden_layers': 2,
        'rnn_hidden_nodes': 512, # 512
        'num_classes': 2,
        'drop_prob': 0.20,
        'bidirectional': True
    }

    window_size = 45
    remove_bound = False
    positive_frame_thres = 23
    model = nn.Sequential(
        crnn.CNNEncoder(**cnn_encoder_params),
        crnn.RNNDecoder(**rnn_decoder_params)
    )

    # change model path here, same with the model result path
    path_checkpoint = 'C:/Users/11764/Desktop/automatic_itch/results/'
    train_dir = "C:/Users/11764/Desktop/automatic_itch"
    #"/media/yhs/8608ebc4-123e-4bc8-803b-d3d75187dbd2/Automatic_itch_project/results/04-28_15-05Window45, no interval no discard, 1-32 train/"
    # change the video to predict here, must be an int
    predict_video = 8


    from timeit import default_timer as timer
    

    
    start = timer()
    path_checkpoint  = path_checkpoint + "checkpoint_best.pkl"
    check = torch.load(path_checkpoint)
    print("video", predict_video)
    model.load_state_dict(check["model_state_dict"])
    model.train()



    # predict frames
    predict_data = mouse_Dataset(data_dir=train_dir, window_size=window_size, validation=True,predict_new=True, frame_interval  = dataset_params['frame_step'],
                               predict_video = predict_video, remove_bound = False,
    sliding_size = 1,
    positive_frame_thres = positive_frame_thres)
    predict_loader = DataLoader(dataset=predict_data, batch_size=1,
                              num_workers=dataset_params['num_workers'])
    labels = [0,1]
    num_labels = 2
    criterion = nn.CrossEntropyLoss()

    model.eval()
    model.to("cpu")
    inputs, labels = predict_data[1234]
    inputs = inputs.to("cpu")
    label_true = labels

    inputs = inputs.reshape([-1,window_size,1,256,256])

    outputs = model(inputs)
    _, predicted = torch.max(outputs.data, 1)
    label_predicted = predicted.cpu().detach().numpy()[0]

    l1 = saliency11(inputs, model)

    end = timer()
    print("time used:",end - start)



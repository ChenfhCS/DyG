import torch
import numpy as np
import argparse
import torch.nn as nn
import os
import warnings
import pandas as pd

from data_loader import get_loader
from mlp import MLP_Predictor


warnings.simplefilter("ignore")

current_path = os.getcwd()

class My_loss(nn.Module):
    def __init__(self):
        super(My_loss, self).__init__()
    def forward(self, x, y):
        return torch.mean(torch.abs((x - y)/y))

def save_log(x, name):
    df_loss=pd.DataFrame(data=x)
    df_loss.to_csv('./experiment_results/{}.csv'.format(name), header=False)

def run_train(args, model, train_loader, test_loader, flag):
    loss_func = My_loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'], weight_decay=args['weight_decay'])
    loss_log = []
    for epoch in range(args['epochs']):
        Loss = []
        for step, (batch_x, batch_y) in enumerate(train_loader):
            model.train()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            out = model(batch_x)
            # print(batch_x, batch_y, out.squeeze(dim=-1))
            loss = loss_func(out.squeeze(dim=-1), batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            Loss.append(loss.item())
        loss_log.append(np.mean(Loss))
        print('Epoch {}, Loss: {}'.format(epoch, np.mean(Loss)))
        # if epoch % args['test_freq'] == 0:

    for test_data in test_loader:
        model.eval()
        test_x, test_y = test_data
        test_x = test_x.to(device)
        test_y = test_y.to(device)
        predict = model(test_x)
        print('predictions: {}| measured: {}'.format(predict[0:10], test_y[0:10]))
        pred_log = predict.view(-1).tolist()
        test_log = test_y.tolist()
        # print(pred_log, test_log)
    
    save_log(loss_log, 'loss')
    save_log(pred_log, 'prediction')
    save_log(test_log, 'measured')
    torch.save(model.state_dict(), './model/{}_{}.pt'.format(flag, args['timesteps']))


def _get_args():
    import json
    parser = argparse.ArgumentParser(description='Test parameters')
    parser.add_argument('--json-path', type=str, required=True,
                        help='the path of hyperparameter json file')
    # parser.add_argument('--test-type', type=str, required=True, choices=['local', 'dp', 'ddp'],
    #                     help='method for DGNN training')
    
    # for experimental configurations
    parser.add_argument('--timesteps', type=int, nargs='?', default=8,
                    help="total time steps used for train, eval and test")
    parser.add_argument('--epochs', type=int, nargs='?', default=100,
                    help="total number of epochs")
    parser.add_argument('--world_size', type=int, default=1,
                        help='method for DGNN training')
    parser.add_argument('--dataset', type=str, default='Epinion',
                        help='method for DGNN training')
    parser.add_argument('--encoder', type=str, nargs='?', default="str",
                    help='Which encoder needs to be predicted')


    args = vars(parser.parse_args())
    with open(args['json_path'],'r') as load_f:
        para = json.load(load_f)
    args.update(para)

    return args

if __name__ == '__main__':
    args = _get_args()
    args['device'] = torch.device("cuda")
    device = args['device']
    flag = args['encoder']
    train_loader, test_loader = get_loader(args, flag)

    model = MLP_Predictor(in_feature = 2)
    model = model.to(device)

    run_train(args, model, train_loader, test_loader, flag)
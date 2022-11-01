import torch
import argparse
import warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm
from dataset.Epinion.epinion import EpinionDatasetLoader
from nn import DySAT
from nn import classifier

class My_Model(torch.nn.Module):
    def __init__(self, args, node_features):
        super(My_Model, self).__init__()
        self.args = args
        self.dgnn = DySAT(args, num_features = node_features)
        self.classifier = classifier(in_feature = 128)

    def forward(self, snapshots):
        final_emb = self.dgnn(snapshots)
        outputs = []
        for time, snapshot in enumerate(snapshots):
            emb = final_emb[:, time, :].to(self.args['device'])
            sample = snapshot.train_samples
            # get target embeddings
            source_id = sample[:, 0]
            target_id = sample[:, 1]
            source_emb = emb[source_id]
            target_emb = emb[target_id]
            input_emb = source_emb.mul(target_emb)
            outputs.append(self.classifier(input_emb))
        return outputs

def _get_args():
    parser = argparse.ArgumentParser(description='example settings')
    
    # for experimental configurations
    parser.add_argument('--timesteps', type=int, nargs='?', default=8,
                    help="total time steps used for train, eval and test")
    parser.add_argument('--epochs', type=int, nargs='?', default=100,
                    help="total number of epochs")
    parser.add_argument('--world_size', type=int, default=1,
                        help='method for DGNN training')
    parser.add_argument('--dataset', type=str, default='Epinion',
                        help='method for DGNN training')
    args = vars(parser.parse_args())
    return args

if __name__ == '__main__':
    args = _get_args()
    args['rank'] = 0
    args['device'] = torch.device("cuda")
    loader = EpinionDatasetLoader(timesteps = args['timesteps'])
    dataset = loader.get_dataset()
    snapshots = [snapshot for snapshot in dataset]
    model = My_Model(args, node_features = 2).to(args['device'])

    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)

    pbar = tqdm(range(200), leave=False)
    for epoch in pbar:
        loss = 0
        outputs = model(snapshots)
        for time, snapshot in enumerate(snapshots):
            y = outputs[time]
            label = snapshot.train_labels
            error = loss_func(y.squeeze(dim=-1), label)
            loss += error
        loss = loss/len(snapshots)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        pbar.set_description('epoch: {} loss: {}'.format(epoch, loss.item()))






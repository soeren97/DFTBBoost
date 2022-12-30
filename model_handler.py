from tqdm import tqdm
import os
import time
import shutil
import utils

import pandas as pd
import numpy as np
import torch

from torch_geometric.loader import DataLoader as GNNDataloader
from torchmetrics import MeanAbsolutePercentageError as MAPE
from torchmetrics import MeanSquaredError as MSE
from torchmetrics import MeanAbsoluteError as MAE
from torchmetrics import MeanSquaredLogError as MSLE

from models import GNN, CNN, NN, GNN_plus
from CostumDataset import CostumDataset

        
class ModelTrainer():
    def __init__(self):
        self.model = None
        self.device = None
        self.dataloader = None
        
        self.loss_fn = None
        self.best_loss = 10e2
        self.patience = 50
        self.reset_patience = None
        self.early_stopping = False
        self.lr = 0
        self.decay_rate = 0
        
        self.epochs = None
        self.batch_size = None

        self.train_loader = None
        self.test_loader = None

        self.optimizer = None

        self.config = None
        
        self.data = None
        self.train_set = None
        self.test_set = None
        self.valid_set = None
        
        self.model_folder = None
        self.save_model = None
    
    def setup_data(self):
        model_name = self.model.__class__.__name__
        
        if model_name in ['GNN', 'GNN_plus']:
            self.loader = GNNDataloader
        else:
            self.loader = torch.utils.data.DataLoader
        
        self.data = CostumDataset(ml_method = model_name)
        
        self.train_set, self.test_set = utils.random_split(self.data, [.8,.2])

    def evaluate_early_stopping(self, loss):
        if not self.best_loss >= loss:
            self.patience -= 1
        else:
            self.best_loss = loss
            self.patience = self.reset_patience
        
        if self.patience == 0:
            self.early_stopping = True         

    def train(self):  
        for data in self.train_loader:
            self.optimizer.zero_grad() 

            if self.model.__class__.__name__ in ['GNN', 'GNN_plus']:
                data.to(self.device)  
                
                pred = self.model(data)
                
                Y = data.y.reshape(-1, 3)
                
            else:
                X = data['x'].to(self.device)
                Y = data['y'].to(self.device)
                
                pred = self.model(X.float())
            
            # Calculate HOMO, LUMO and gap
            pred = utils.find_homo_lumo(pred)
            
            loss = self.loss_fn(pred, Y)
            
            loss.backward()  
            
            # Update using the gradients

            self.optimizer.step()
 
        return loss
    
    def test(self):
        for data in self.test_loader:
            # Use GPU if available
            data.to(self.device)  

            # Reset gradients
            self.optimizer.zero_grad() 
            
            pred = self.model(data)
            
            pred = utils.find_homo_lumo(pred)
            Y = data.y.reshape(-1, 3)
            
            # Calculating the loss and gradients
            loss = self.loss_fn(pred, Y)   
        return loss
        
    def train_model(self):
        losses_train = []
        losses_test = []
        
        self.train_loader = self.loader(self.train_set, 
                            batch_size = self.batch_size, 
                            shuffle = True,
                            )
        
        self.test_loader = self.loader(self.test_set, 
                            batch_size = self.batch_size, 
                            shuffle = True
                            )
        
        for epoch in (pbar:= tqdm(range(self.epochs),  
                                  total = self.epochs,
                                  desc = 'Training',
                                  leave = False)):            
            loss_train = self.train()
            losses_train.append(loss_train.cpu().detach())
            
            loss_test = self.test()
            losses_test.append(loss_test.cpu().detach())

            pbar.set_description(f'Test loss {loss_test:.2E}')

            self.evaluate_early_stopping(loss_test)

            if self.early_stopping and self.save_model:
                print(f' Best loss {self.best_loss}')
                print(self.model_folder)
                
                losses = np.array([losses_train,
                            losses_test]).T
        
                return pd.DataFrame(losses, 
                                    columns = [
                                        'Train_loss',
                                        'Test_loss'])
        
            elif self.early_stopping:
                losses = np.array([losses_train,
                            losses_test]).T
        
                return pd.DataFrame(losses, 
                                    columns = [
                                        'Train_loss',
                                        'Test_loss'])

        losses = np.array([losses_train,
                            losses_test]).T
        
        return pd.DataFrame(losses, 
                            columns = [
                                'Train_loss',
                                'Test_loss'])

    def main(self):
        self.config = utils.load_config()
        self.epochs = self.config['epochs']
        self.batch_size = int(self.config['batch_size'] / 32)
        self.decay_rate = float(self.config['decay_rate'])
        self.lr = float(self.config['lr'])
        self.reset_patience = self.config['start_patience']
        self.model = eval(self.config['model'])().to(self.device)
        
        self.save_model = True
        self.model_folder = 'Models/m' + str(time.time())[:-8] + '/'

        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                    lr = self.lr, 
                                    weight_decay=self.decay_rate
                                    )

        self.loss_fn = eval(self.config['loss_fn'])().to(self.device)
        
        self.data_intervals = os.listdir('Data/datasets')
        
        self.setup_data()

        loss_df = self.train_model()
        
        if self.save_model:
            os.mkdir(self.model_folder)
            torch.save(self.model, self.model_folder + 'model.pkl')
            loss_df.to_pickle(self.model_folder + 'losses.pkl')
            shutil.copy('model_config/config.yaml', self.model_folder + 'config.yaml' )
         
if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer.main()    
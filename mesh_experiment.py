import torch
import numpy as np

class Experiment:
    
    def __init__(   self,
                    model,
                    optimizer,
                    loss_fn,
                    train_loader,
                    test_loader,
                    epochs,
                    rand_inits,
                    OUT_DIR,
                    model_info,
                    rand_seed=11 ):

        """Constructor

        Builds an experiment object for training and evaluating a GINR model with 
        a single configuration.

        Args:
            model (torch.nn.module) Torch model representing MLP in GINR
            optimizer (torch.optim) Optimizer for training
            loss_fn (torch.nn) Objective function for training
            train_loader (torch.utils.data.DataLoader) Dataloader for train data
            test_loader (torch.utils.data.DataLoader) Dataloader for test data
            epochs (int) Number of epochs
            rand_inits (int) Number of random inits for model
            OUT_DIR (str) Experiment directory named with datetime and training dataset.
                Format: MM_DD_YY_HH:MM:SS_datasetname
            model_info (dict) Dictionary with model param and dataset information.
            rand_seed (int) Random seed for torch and numpy default=11

        Returns:
            self
        """

        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.rand_inits = rand_inits
        self.OUT_DIR = OUT_DIR
        torch.manual_seed(rand_seed)
        np.random.seed(rand_seed)

        self.exp_log = {
            'model': str(model),
            'optimizer': str(optimizer),
            'loss_fn': str(loss_fn),
            'epochs': epochs,
            'rand_inits': rand_inits,
            'rand_seed': rand_seed
            }
        self.exp_log.update(model_info)
        
        
    def train(self, run_num):
        """Trains the experiment model
        
        Trains the model and save model parameters with the best performance on evaluation data.

        Args:
            run_num (int) The iteration of random init for this particular training run

        Returns:
            dict of:
                best_loss (float) Best evaluation loss over training. This is the model saved as best model state dict
                training_losses (np.array)
                eval_losses (np.array)
        """

        self.model.train()
        print(f'started training for {self.epochs} epochs')
        training_losses = np.zeros((self.epochs, 1))
        eval_losses = np.zeros((self.epochs, 1))
        for i in range(self.epochs):
            best_model = self.model.state_dict()
            best_loss = 1e10
            losses=[]
            for batch in self.train_loader:
                self.optimizer.zero_grad()
                
                batchX = batch[0]
                batchY = batch[1]
                
                preds = self.model(batchX)
                
                loss = self.loss_fn(preds, batchY)
                losses.append(loss.detach().numpy())
                loss.backward()
                self.optimizer.step()
            
            mean_loss = np.array(losses).sum() / len(losses) # technically not quite right cuz partial batches will be weighted slightly higher (we will survive)
            training_losses[i] = mean_loss

            eval_loss = self.evaluate()
            eval_losses[i] = eval_loss
            if eval_loss < best_loss:
                best_model = self.model.state_dict()
                best_loss = eval_loss
            if i % 5 == 0:
                print(f'epoch {i} training loss: {mean_loss} eval loss: {eval_loss}')

        torch.save(best_model, f'{self.OUT_DIR}run_{run_num}/best-model-parameters.pt')
        return {
            'run' : run_num,
            'best_loss' : best_loss,
            'train_losses' : training_losses,
            'eval_losses' : eval_losses
            }
        
    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            
            correct = 0
            total = 0
            for batch in self.test_loader:
                batchX = batch[0]
                batchY = batch[1]
                
                preds = self.model(batchX)
                loss = self.loss_fn(preds, batchY)
            
            return loss.detach().numpy()
        
    def run(self):
        train_results = []
        for i in range(self.rand_inits):
            
            train_results.append(self.train(i))

        train_results = np.array(train_results)
        self.exp_log['train_results'] = train_results
        return self.exp_log
        
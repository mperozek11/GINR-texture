import torch
import numpy as np

class Experiment:
    
    def __init__(self, model, optimizer, loss_fn, train_loader, test_loader, epochs):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = epochs
        
    def train(self):
        self.model.train()
        for i in range(self.epochs):
            
            losses = []
            for batch in self.train_loader:
                self.optimizer.zero_grad()

                batchX = batch[0]
                batchY = batch[1]

                preds = self.model(batchX)

                loss = self.loss_fn(preds, batchY)
                losses.append(loss.detach().numpy())
                
                loss.backward()
                self.optimizer.step()
            
            
            if i % 100 == 0:
                mean_loss = np.array(losses).sum() / len(losses) # technically not quite right cuz partial batches will be weighted slightly higher (we will survive)
                print(f'epoch {i} loss: {mean_loss}')
        
    def evaluate(self, thresh=0.5):
        self.model.eval()
        with torch.no_grad():
            
            correct = 0
            total = 0
            for batch in self.test_loader:
                batchX = batch[0]
                batchY = batch[1]
                
                preds = self.model(batchX)
                inference = (preds > thresh)
                
                accs = (inference == batchY)
                
                correct += accs.sum()
                total += accs.shape[0]
                


            return correct / total
        
        
        
        
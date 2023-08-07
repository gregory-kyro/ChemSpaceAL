# Necessary import statements
from torch.utils.data.dataloader import DataLoader
from torch.cuda.amp import GradScaler
from tqdm import tqdm
import wandb
from .Configurations import *


class Trainer:
""" Class to train GPT model """

    def __init__(self, model, train_dataset, test_dataset=None):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = model.config
        self.stoi = train_dataset.stoi
        self.itos = train_dataset.itos

    def train(self, wandb):
        model, config = self.model, self.config
        optimizer = model.configure_optimizers(config)
        scaler = GradScaler() # Define variable used for gradient scaling in mixed-precision training
        self.tokens = 0 # Initialize a counter used for learning rate decay

        def run_epoch(split):
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=True, batch_size=config.batch_size, num_workers=config.num_workers)
            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            # For batch index, batch in progress bar:
            for it, (x, y) in pbar:
                # Move the input data tensor, target data tensor, property tensor, and scaffold tensor to GPU
                x, y = x.to(config.device), y.to(config.device)
                # Allow model to use lower-precision computations for improved memory usage
                if config.device == 'cuda':
                    with torch.cuda.amp.autocast():
                        with torch.set_grad_enabled(is_train):
                            logits, loss = model(x, y)
                            loss = loss.mean()
                            losses.append(loss.item())
                else:
                    with torch.cpu.amp.autocast():
                        with torch.set_grad_enabled(is_train):
                            logits, loss = model(x, y)
                            loss = loss.mean()
                            losses.append(loss.item())

                if is_train:
                    model.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer) # Unscale the gradients of the optimizer's parameters to their original values
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Clip gradients of model parameters to prevent them from exploding, setting maximum gradient norm to be 1.0
                    scaler.step(optimizer) # Update the optimizer's parameters based on calculated gradients
                    scaler.update() # Update the scale factor of the gradient scaler
                  
                    if config.lr_decay:
                        self.tokens += (y >= 0).sum() # Increment the number of processed tokens by the count of valid tokens (not padding or special tokens)
                        if config.lr_warmup and self.tokens < config.warmup_tokens:
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens)) # Perform a linear warm-up
                        else:
                            baseline = config.warmup_tokens if config.lr_warmup else 0
                            # Calculate the progress of training in terms of the number of tokens processed
                            progress = float(self.tokens - baseline) / float(max(1, config.final_tokens - baseline))
                            # Calculate the scaling factor for the learning rate (between 0.1 and 1.0) to gradually reduce learning rate as training progresses
                            lr_mult = max(0.1, 0.5 * (1.0 + np.cos(np.pi * progress)))
                        lr = config.learning_rate * lr_mult # Multiply the base learning rate by the scaling factor to obtain the updated learning rate
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate
                    if wandb is not None: # Log training progress using Weights & Biases
                        wandb.log({'step_train_loss': loss, 'train_step': it + epoch*len(loader), 'learning_rate': lr})
                    
                    # Update the description of the progress bar with epoch, iteration, and training loss
                    pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")
            
          return float(np.mean(losses))

        # Initialize best loss as infinity
        best_loss = float('inf')

        for epoch in range(config.epochs):
            print(f'{epoch=}')
            train_loss = run_epoch('train')
            log_dict = {'epoch_train_loss': train_loss, 'epoch': epoch + 1}
            
            if self.test_dataset is not None:
                test_loss = run_epoch('test')
                log_dict['epoch_valid_loss'] = test_loss
            
            if wandb is not None:
                wandb.log(log_dict)
            good_model = False
            
            if self.test_dataset is None:
                good_model = True
            
            else:
                if test_loss < best_loss:
                    best_loss = test_loss
                    good_model = True
            
            if good_model:
                torch.save(self.model.state_dict(), self.config.save_ckpt_path)
              

    @staticmethod
    def train_GPT(train_dataset, config_dict = config.get_config(), valid_dataset=None, load_ckpt=False):
        """
        Function to train GPT model
        """
      
        mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                          warmup_tokens=0.1*train_dataset.len_data*train_dataset.block_size,
                          final_tokens=config_dict["epochs"]*train_dataset.len_data*train_dataset.block_size,
                          loss_ignore_index=train_dataset.stoi['<'],
                          **config_dict)
        
        model = GPT(mconf)
        if load_ckpt:
            model.load_state_dict(torch.load(config_dict['load_ckpt_path']))
        model.to(config_dict["device"])
        torch.compile(model)
        trainer = Trainer(model, train_dataset, valid_dataset)

        %env WANDB_EXECUTABLE=python3
        wandb.init(project=config_dict["wandb_project"], name=config_dict["wandb_runname"])
        trainer.train(wandb=wandb)
        
        return model, trainer, wandb

"""
# Example usage
model = Trainer.train_GPT(train_dataset=train_dataset, config_dict=config.get_config(), valid_dataset=valid_dataset)
"""

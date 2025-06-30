import torch
import torchvision
import os

class SaveBestModel:
    """
    This class implement an easy to implement checkpoint to only save the best model.

    How it works:
        1. Initiate a SaveBestModel object with desired parameters.
        2. Call the object after each epoch and the class will determine based on initial setup parameters whether the
           model is good enough to be saved.  (Better than previous ones)
        It's that easy!

    Examples:
        >>># Before the training loop:

        >>>save_best_model = utils.SaveBestModel(

        >>>         config["model_dir"], metric_name="validation loss",

        >>>         model_name=config["model_name"], best_metric_val=float('inf'),

        >>>         evaluation_method='MIN')

        >>># In the training loop after each epoch:

        >>>save_best_model(np.array(v_loss).mean(), epoch, model, optimizer, criterion)
    """
    def __init__(self, save_dir, metric_name, model_name, best_metric_val, evaluation_method='MAX', verbose: bool = True):
        """
        Setup the SaveBestModel object
        :param save_dir: directory where to save the checkpoints
        :param metric_name: The name of the metric that we are using to measure the best model
        :param model_name: The name of the model to be saved
        :param best_metric_val: minimal metric value under which the model won't be saved.  Usually inf or -inf
        :param evaluation_method: 'MAX' or 'MIN' Whether the goal is to maximise the metric or minimize it.
        """
        self.best_metric_val = best_metric_val
        self.metric_name = metric_name
        self.model_name = model_name
        self.save_dir = save_dir
        self.evaluation_method = evaluation_method.upper()
        self.verbose = verbose
        assert self.evaluation_method == "MAX" or self.evaluation_method == "MIN"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def __call__(self, current_val, epoch, model, optimizer, criterion: torch.nn.Module, outputMessage=""):
        """
        Call after each epoch
        :param current_val: Current metric value
        :param epoch: the current epoch
        :param model: the model to save
        :param optimizer: the optimizer
        :param criterion: the loss object
        :param outputMessage: the message to output when the model is saved.
        :return: None
        """
        if self.evaluation_method == "MIN":
            if current_val < self.best_metric_val:
                self.best_metric_val = current_val
                if self.verbose:
                    print(f"Best {self.metric_name}: {self.best_metric_val}")
                    print(f"Saving best model for epoch: {epoch + 1} at {self.save_dir}")
                    print(outputMessage)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'activation': criterion.activation.state_dict(),
                    'metric':current_val,
                    'metric_name':self.metric_name,
                }, '{}/{}.pth'.format(self.save_dir, self.model_name))
        elif self.evaluation_method == "MAX":  # all other metrics should be maximized
            if current_val > self.best_metric_val:
                self.best_metric_val = current_val
                if self.verbose:
                    print(f"Best {self.metric_name}: {self.best_metric_val}")
                    print(f"Saving best model for epoch: {epoch + 1} at {self.save_dir}")
                    print(outputMessage)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'activation': criterion.activation.state_dict(),
                    'metric':current_val,
                    'metric_name':self.metric_name,
                }, '{}/{}.pth'.format(self.save_dir, self.model_name))
        else:
            raise ValueError(f"Invalid evaluation method: {self.evaluation_method}, Expected: 'MAX' or 'MIN'")


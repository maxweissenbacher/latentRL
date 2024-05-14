import numpy as np


class EarlyStopper:
    def __init__(self, patience: int, min_delta: float) -> None:
        """Early stopping mechanism to prevent overfitting.

        Attributes:
            patience (int): Number of epochs to wait before stopping.
            min_delta (float): Minimum change in validation loss to qualify as improvement.
            counter (int): Number of epochs with no improvement.
            min_validation_loss (float): Minimum observed validation loss.
        """

        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
        self.stop = False

    def track(self, validation_loss: float) -> bool:
        """Check if early stopping is necessary.
        Args:
            validation_loss (float): Current validation loss.
        Returns:
            bool: True if the model should be stopped, False otherwise.
        """

        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
                print(f"--- early stopping after {self.patience} epochs ---")
                return True
        return False

    def reset_counter(self):
        """Reset counter to 0"""
        self.counter = 0
        self.stop = False

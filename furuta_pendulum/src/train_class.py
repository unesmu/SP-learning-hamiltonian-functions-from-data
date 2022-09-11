from .plots import *
from .dynamics import *
from .trajectories import *
from .utils import *
from .train_helpers import *
from .train import *

class Training:
    """
    Training class
    """

    def __init__(self):
        pass

    def _init_model(self, model_name):
        pass

    def _init_dataset(self):
        pass

    def _save(self):
        pass

    def _output_training_stats(self, epoch, train_loss, test_loss, t0):
        """
        Output and save training stats every epoch or multiple of epochs
        """

        if epoch % self.print_every == 0 or epoch == self.num_epochs - 1:
            print(
                "[%d/%d]\t train loss: %.4f, test loss: %.4f, t: %2.3f"
                % (epoch, self.num_epochs, train_loss, test_loss, time.time() - t0)
            )

        return

    def _train_step(self):
        """
        basic training step of the model
        """
        return

    def _test_step(self):
        """
        basic training step of the model
        """
        return

    def train(self):
        """
        Training function. Uses the classes variables, and returns stats from the
        training procedure
        """
        return
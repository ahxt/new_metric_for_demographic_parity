import torch



# Define a new class called gap_reg that inherits from torch.nn.Module
class gap_reg(torch.nn.Module):
    # Define the constructor for the class
    def __init__(self, mode = "dp"):
        # Call the constructor of the parent class
        super(gap_reg, self).__init__()
        # Set the mode attribute to the value passed in as an argument
        self.mode = mode

    # Define the forward method for the class
    def forward(self, y_pred, s, y_gt):
        # Select the predicted values corresponding to s == 0
        y0 = y_pred[s == 0]
        # Select the predicted values corresponding to s == 1
        y1 = y_pred[s == 1]
        # Calculate the regularization loss as the absolute difference between the means of y0 and y1
        reg_loss = torch.abs(torch.mean(y0) - torch.mean(y1))
        # Return the regularization loss along with three tensors of zeros
        return reg_loss, torch.Tensor([0]), torch.Tensor([0]), torch.Tensor([0])


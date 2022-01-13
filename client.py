from collections import OrderedDict

import torch
from model import SiameseNetwork as Net
from utils import loadData, train, test


import flwr as fl

DEVICE = torch.device('cpu')
net = Net().to(DEVICE)
trainLoader, testLoader = loadData()


class CifarClient(fl.client.NumPyClient):
    def get_parameters(self):
        """
        Returns model weights as a list of NumPy ndarrays
        """
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        """
        (Optional) 
        Updates local model weights with the parameters received from the server
        """
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """
        1. Sets the local model weights
        2. Trains the local model
        3. Receives the updated local model weights
        """
        print("Fitting the client model")
        self.set_parameters(parameters)
        train(net, trainLoader, epochs=10, device=DEVICE)
        return self.get_parameters(), len(trainLoader), {}

    def evaluate(self, parameters, config):
        """
        Tests the local model
        """
        print("Evalauating the client model")
        self.set_parameters(parameters)
        loss = test(net, testLoader, DEVICE)
        print("Test Loss:", loss)
        return float(loss), len(testLoader), {"loss": float(loss)}


fl.client.start_numpy_client("localhost:8080", client=CifarClient())

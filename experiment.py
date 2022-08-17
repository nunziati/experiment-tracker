from datetime import datetime
import torch
import os
import shutil

import model
import dataset
import training_algorithm

class Experiment:
    def __init__(
        self,
        name = "",
        model = None,
        training_set = None,
        test_set = None,
        training_algorithm = None,
        metric = None,
        callback = None
    ):
        if name == "": name = "experiment_" + str(datetime.now()).replace(" ", "-")[:20]

        self.name = name
        self.model = model # instance of torch.nn.Module
        self.training_set = training_set # instance of ETDataset
        self.test_set = test_set # instance of ETDataset
        self.training_algorithm = training_algorithm # instance of ETTraining_algorithm

    def ready(self):
        return self.model != None and self.training_set != None and self.training_algorithm != None
    
    def run(self):
        if not self.ready():
            print("One of [model, dataset, training_algorithm], not properly set: experiment can't start.")
            return

        if os.path.isdir(f"experiments/{self.name}"):
            selection = input(f"Duplicated experiment name {self.name}, do you want to overwrite? (y/n) ")
            if selection != "y": return
            else: shutil.rmtree(f"experiments/{self.name}")
        
        os.makedirs(f"experiments/{self.name}")
        os.chdir(f"experiments/{self.name}")

        with open("model_description.txt", "a+") as f:
            if hasattr(self.model.__class__, "description") and callable(self.model.description):
                f.write(self.model.description())
            else:
                f.write("No description found for the model.")

        model, training_log = self.training_algorithm(self.model, self.training_set)

        with open("training.log", "a+") as f:
            f.write(training_log)

        torch.save(model.state_dict(), "model.pth")
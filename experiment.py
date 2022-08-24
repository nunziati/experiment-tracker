from datetime import datetime
import torch
import os
import shutil
from matplotlib.figure import Figure

import model
import dataset
import algorithm

class ETExperiment:
    def __init__(
        self,
        experiment_name = "",
        model = None,
        training_set = None,
        test_set = None,
        pipeline = None
    ):
        if experiment_name == "":
            experiment_name = "experiment_" + str(datetime.now()).replace(" ", "-")[:20]

        self.experiment_name = experiment_name
        self.model = model # instance of torch.nn.Module
        self.training_set = training_set # instance of ETDataset
        self.test_set = test_set # instance of ETDataset
        self.pipeline = pipeline # instance (or sequence of instances) of ETAlgorithm

    def ready(self):
        return self.model != None and self.training_set != None and self.pipeline != None

    def execute_pipeline(self):
        return self.pipeline(self)

    def save_results(self):
        if not hasattr(self, "results"):
            raise Exception("Results not yet calculated, run the pipeline first.")
        if not isinstance(self.results, dict):
            raise TypeError("Expected dict.")

        experiment_dir = os.getcwd()
        if os.path.isdir(f"results"):
            selection = input(f"Duplicated directory results/ do you want to overwrite? (y/n) ")
            if selection == "y":
                shutil.rmtree(f"results")
                os.makedirs(f"results")
        else:
            os.makedirs(f"results") 
        
        os.chdir(f"results")

        self.save_results_recursive(self.results)

        os.chdir(experiment_dir)

    def save_results_recursive(self, results):
        for k, v in results.items():
            if isinstance(v, str):
                with open(k + ".txt", "a+") as f: f.write(v)
            elif isinstance(v, Figure):
                v.savefig(k + ".jpg")
            elif isinstance(v, torch.nn.Module):
                torch.save(v.state_dict(), k + ".pth")
            elif isinstance(v, dict):
                current_dir = os.getcwd()
                if os.path.isdir(k):
                    selection = input(f"Duplicated directory {k}, do you want to overwrite? (y/n) ")
                    if selection == "y":
                        shutil.rmtree(k)
                        os.makedirs(k)
                else:
                    os.makedirs(k)

                os.chdir(k)
                self.save_results_recursive(v)
                os.chdir(current_dir)
            else:
                with open(k + ".txt", "a+") as f: f.write(str(v))

    def run(self):
        if not self.ready():
            print("One of [model, trainig_set, algorithm], not properly set: experiment can't start.")
            return

        if os.path.isdir(f"experiments/{self.experiment_name}"):
            selection = input(f"Duplicated experiment name {self.experiment_name}, do you want to overwrite? (y/n) ")
            if selection != "y": return
            else: shutil.rmtree(f"experiments/{self.experiment_name}")
        
        os.makedirs(f"experiments/{self.experiment_name}")
        os.chdir(f"experiments/{self.experiment_name}")

        # put it in the pipeline
        """with open("model_description.txt", "a+") as f:
            if hasattr(self.model.__class__, "description") and callable(self.model.description):
                f.write(self.model.description())
            else:
                f.write("No description found for the model.")"""

        self.results = self.execute_pipeline()

        self.save_results()
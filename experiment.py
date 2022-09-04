from datetime import datetime
import torch
import os
import shutil
from matplotlib.figure import Figure

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
            if v == None:
                continue
            elif isinstance(v, str):
                with open(k + ".txt", "a+") as f: f.write(v)
            elif isinstance(v, Figure):
                v.savefig(k + ".jpg")
            elif isinstance(v, torch.nn.Module):
                torch.save(v.state_dict(), k + ".pth")
            elif isinstance(v, torch.Tensor):
                torch.save(v, k + ".pth")
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

        self.results = self.execute_pipeline()

        if hasattr(self.model.__class__, "description") and callable(self.model.description):
            self.results["model_description"] = self.model.description()
        else:
            self.results["model_description"] = "No description found for the model."

        self.save_results()


def build_experiment(config):
    name = config["name"]
    model = config["model"]
    type = config["type"]
    dataset_name = config["dataset"]

    assert isinstance(name, str)
    assert isinstance(model, torch.nn.Module)
    assert type in ["joint", "task", "online"]
    assert dataset_name in ["cifar10", "cifar100"]

    predefined_datasets = dict(
        cifar10 = dataset.get_cifar10_dataset,
        cifar100 = dataset.get_cifar100_dataset
    )

    training_set, test_set = predefined_datasets[dataset_name]()

    if type == "joint":
        training_algorithm = algorithm.ETAlgorithm(algorithm.train, config)
        test_algorithm = algorithm.ETAlgorithm(algorithm.evaluate, config)
        pipeline = algorithm.ETPipeline("holdout", [training_algorithm, test_algorithm])
    elif type == "task":
        pipeline = algorithm.ETAlgorithm(algorithm.task_incremental_train, config)
    elif type == "online":
        pipeline = algorithm.ETAlgorithm(algorithm.single_pass_online_train, config)

    experiment = ETExperiment(name, model, training_set, test_set, pipeline)

    return experiment


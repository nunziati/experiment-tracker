from datetime import datetime
import torch
import os
from model import cifar10_cnn

class Experiment:
    predefined_collection = dict(
        model = {"cifar10_cnn": cifar10_cnn},
        dataset = {},
        training_algorithm = {},
        metric = {},
        callback = {}
    )

    def __init__(
        self,
        name = "",
        model = None,
        dataset = None,
        training_algorithm = None,
        metric = None,
        callback = None
    ):
        if name == "": name = "experiment_" + str(datetime.now()).replace(" ", "-")[:20]   
        if isinstance(model, dict): model = self.from_predefined(model, "model")
        if isinstance(dataset, dict): dataset = self.from_predefined_dataset(dataset, "dataset")
        if isinstance(training_algorithm, dict): training_algorithm = self.from_predefined(training_algorithm, "training_algorithm")
        if isinstance(metric, dict): metric = self.from_predefined_metric(metric, "metric")
        if isinstance(callback, dict): callback = self.from_predefined_callback(callback, "callback")

        self.name = name
        self.model = model
        self.dataset = dataset
        self.training_algorithm = training_algorithm
        self.metric = metric
        self.callback = callback

    def from_predefined(self, config, attribute_name):
        if config["name"] not in self.predefined_collection[attribute_name]:
            raise ValueError(f"The {attribute_name} {config['name']} is not present in the predefined {attribute_name} collection.")
        return self.predefined_collection[attribute_name](config)

    def ready(self):
        return (self.model != None and self.dataset != None and self.training_algorithm != None)
    
    def run(self):
        if not self.ready():
            print("One of [model, dataset, training_algorithm], not properly set: experiment can't start.")
            return

        os.mkdir(f"experiments/{self.name}")

        
        
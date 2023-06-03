import numpy as np
import pandas as pd

import json
import pickle

import warnings
warnings.filterwarnings("ignore")
import config


class MedicalInsurance():
    def __init__(self, age, sex, bmi, children, smoker, region):

        self.age = age
        self.sex = sex
        self.bmi = bmi
        self.children = children
        self.smoker = smoker

        # input for one-hot encodding
        self.region = "region_" + region

    def load_models(self):
        #fitted model
        with open(config.MODEL_FILE_PATH, "rb") as f:
            self.model = pickle.load(f)

        with open(config.JSON_FILE_PATH, "r") as f:
            self.json_data = json.load(f)

    def get_predicted_charges(self):

        self.load_models()   # Creating instance function of model and json_data
        
        # index of one-hot encodded columns
        column_list=list(self.json_data['columns'])
        region_index = column_list.index(self.region)

        test_array = np.zeros(len(self.json_data['columns']))

        # Note: put feature values in front of correct feature index
        test_array[0] = self.age
        test_array[1] = self.json_data['sex'][self.sex]
        test_array[2] = self.bmi
        test_array[3] = self.children
        test_array[4] = self.json_data['smoker'][self.smoker]
        test_array[region_index] = 1

        print("Test Array -->\n", test_array)

        charges = self.model.predict([test_array])[0]

        return round(charges,2)


if __name__ == "__main__":
    age = 26.0
    sex = "male"
    bmi = 28.6
    children = 2.0
    smoker = "no"      
    region = "northeast"

    med_ins = MedicalInsurance(age, sex, bmi, children, smoker, region)
    charges = med_ins.get_predicted_charges()
    print("Predicted Insurance Charges:", charges, "/- Rs.")

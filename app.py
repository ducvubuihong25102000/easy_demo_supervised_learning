from genericpath import isdir
from pickletools import optimize
import pandas
import wget
from pathlib import Path
from my_model import MyModel
import numpy as np
import math
import os


#download file for project
url1 =  "https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/graphing.py"
url2 = "https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/m0b_optimizer.py"
url3 = "https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/Data/seattleWeather_1948-2017.csv"
#directory name contain 3 file are download
directory = "download"

if not os.path.isdir(f"./{directory}"):
    os.mkdir(f"{directory}")
if not ( Path(f"./{directory}/graphing.py").is_file() or Path(f"./{directory}/m0b_optimizer.py").is_file() or Path(f"./{directory}/seattleWeather_1948-2017.csv").is_file()):
    wget.download(url1 ,out="download")
    wget.download(url2 ,out="download")
    wget.download(url3 ,out="download")

import download.graphing as gr
from download.m0b_optimizer import MyOptimizer

from CostFunction.cost_function import cost_function_squared
from CostFunction.cost_function_2 import cost_function_absolute
import training_method as tm


# Load a file that contains weather data for Seattle
data = pandas.read_csv('.\download\seattleWeather_1948-2017.csv', parse_dates=['date'])

# Keep only January temperatures
data = data[[(d.month == 1 )  for d in data.date]].copy()



gr.scatter_2D(data, label_x="date", label_y="min_temperature", title="January Temperatures (°F)",show=True)

#Normalize data
data["years_since_1982"] = [(d.year + d.timetuple().tm_yday / 365.25) - 1982 for d in data.date]

# Scale and offset temperature so that it has a smaller range of values
data["normalised_temperature"] = (data["min_temperature"] - np.mean(data["min_temperature"])) / np.std(data["min_temperature"])

gr.scatter_2D(data, label_x="years_since_1982", label_y="normalised_temperature", title="January Temperatures (Normalised)")


# Create our model ready to be trained
model = MyModel()
model_1 = MyModel()
my_optimize = MyOptimizer()
my_optimize_1 = MyOptimizer()

print(f"Model parameters before training:\t\t{model.intercept:.8f},\t{model.slope:.8f}")


# print("Model visualised before training:")
# gr.scatter_2D(data, "years_since_1982", "normalised_temperature", trendline=model.predict,show = True) 

data_1 = data

for i in range(0,10001):
    continue_loop, cost = tm.train_one_iteration(model_inputs = data["years_since_1982"],
                                                    true_temperatures = data["normalised_temperature"],
                                                    last_cost = math.inf, 
                                                    optimizer=my_optimize,
                                                    model= model
                                                    )
    
    # continue_loop_1, cost_1 = tm.train_one_iteration_1(model_inputs = data_1["years_since_1982"],
    #                                                 true_temperatures = data_1["normalised_temperature"],
    #                                                 last_cost = math.inf, 
    #                                                 optimizer=my_optimize_1,
    #                                                 model= model_1
    #                                                 )
    if i%500==0:
        print(f"Model parameters after {i} iteration of training:\t{model.intercept:.8f},\t{model.slope:.8f}")
        # print(f"Model parameters after {i} iteration of training:\t{model_1.intercept:.8f},\t{model_1.slope:.8f}")
        

print("Model visualised after training:")
gr.scatter_2D(data, "years_since_1982", "normalised_temperature", trendline=model.predict,show=True) 
# gr.scatter_2D(data_1, "years_since_1982", "normalised_temperature", trendline=model_1.predict,show=True) 
# gr.scatter_3D(data, "years_since_1982", "normalised_temperature",show=True) 


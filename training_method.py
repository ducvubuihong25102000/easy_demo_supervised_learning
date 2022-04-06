from pickletools import optimize
from my_model import MyModel
from CostFunction.cost_function import cost_function_squared
from CostFunction.cost_function_2 import cost_function_absolute

from download.m0b_optimizer import MyOptimizer


def train_one_iteration(model_inputs, true_temperatures, last_cost:float,optimizer : MyOptimizer, model : MyModel):
    '''
    Runs a single iteration of training.


    model_inputs: One or more dates to provide the model (dates)
    true_temperatues: Corresponding temperatures known to occur on those dates

    Returns:
        A Boolean, as to whether training should continue
        The cost calculated (small numbers are better)
    '''
    # === USE THE MODEL ===
    # Estimate temperatures for all data that we have
    estimated_temperatures = model.predict(model_inputs)

    # === OBJECTIVE FUNCTION ===
    # Calculate how well the model is working
    # Smaller numbers are better 
    difference, cost = cost_function_squared(true_temperatures, estimated_temperatures)

    # Decide whether to keep training
    # We'll stop if the training is no longer improving the model effectively
    if cost >= last_cost:
        # Stop training
        return False, cost
    else:
        # === OPTIMIZER ===
        # Calculate updates to parameters
        intercept_update, slope_update = optimizer.get_parameter_updates(model_inputs, cost, difference)

        # Change the model parameters
        model.slope += slope_update
        model.intercept += intercept_update

        return True, cost
    
def train_one_iteration_1(model_inputs, true_temperatures, last_cost:float,optimizer : MyOptimizer, model : MyModel):
    """"Train use cost_function as absolute"""

    estimated_temperatures = model.predict(model_inputs)

    difference, cost = cost_function_absolute(true_temperatures, estimated_temperatures)

    if cost >= last_cost:
        # Stop training
        return False, cost
    else:
        intercept_update, slope_update = optimizer.get_parameter_updates(model_inputs, cost, difference)
        model.slope += slope_update
        model.intercept += intercept_update

        return True, cost
    
    


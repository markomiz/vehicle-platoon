## To Run Different Scenarios
run scenarios.py - you can see the available arguments
tuning cantrollers can be done with tune_controller.py - but you have to manually update the gains once you perform the tuning

plotting is done in plotting.py
the 'world' which handles setup and passing messages beteen vehicles is in world.py

the vehicles are modelled as a system (vehicle_system.py)
of a set of estimators (vehicle_estimate.py) a controller (controller.py) and the 'ground truth' vehicle model (vehicle.py)



## Abstract
Cooperative driving systems, such as vehicle platooning, offer tremendous potential for improving road safety, traffic flow, and fuel efficiency. This paper demonstrates a simple and effective method for vehicle platooning, which improves state estimates and predictions through vehicle-to-vehicle communication of state estimates (obtained by combining several sensor measurements) and next controls. 
Each vehicle in the platoon solves an optimal control problem to maintain a time gap behind the vehicle ahead and follow the same path given its good prediction of the car ahead's future state.  

A comparison is given with a reactive control system, where the vehicles do not communicate. Despite good performance with a small number of cars in a platoon, the string instability of the non-communicative, reactive system leads to very dangerous outcomes for more than a few vehicles. In contrast, the communicative system maintains very good performance regardless of the number of vehicles in the platoon. The results demonstrate the significance of incorporating communication and optimal control to enhance the performance of vehicle platooning and the dangers of scaling reactive systems.

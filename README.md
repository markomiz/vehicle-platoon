## To Run Different Scenarios
run scenarios.py - you can see the available arguments

tuning cantrollers can be done with tune_controller.py - but you have to manually update the gains once you get the results from the genetic algorithm

plotting is done in plotting.py
the 'world' which handles setup and passing messages beteen vehicles is in world.py

the vehicles are modelled as a system (vehicle_system.py)
of a set of estimators (vehicle_estimate.py) a controller (controller.py) and the 'ground truth' vehicle model (vehicle.py)



## Abstract
Cooperative driving systems, such as vehicle platooning, offer tremendous potential for improving road safety, traffic flow, and fuel efficiency. This paper demonstrates a simple and effective method for vehicle platooning, which improves state estimates and predictions through vehicle-to-vehicle communication of state estimates (obtained by combining several sensor measurements) and next controls. 
Vehicles in the platoon solve an optimal control problems to maintain a time gap behind the vehicle ahead.  

A comparison is given with a reactive control system, where the vehicles do not communicate. Despite good performance once well tuned, the delay of propogation of speed changes can lead to dangerous behaviour in certain cases. In contrast, the communicative system can react as a unit and therefore perform more safely. The results demonstrate the significance of incorporating communication and optimal control to enhance the performance of vehicle platooning and the dangers of purely reactive systems.

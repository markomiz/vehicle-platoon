import numpy as np
import matplotlib.pyplot as plt
from vehicle import *
from vehicle_estimator import *
from helpers import *



def LoneCarEstimationTest():
    # Create a Vehicle
    initial_state = np.array([0., 0., 0., 10.])  # [x, y, theta, speed]
    wheelbase = 2.5
    vehicle = Vehicle(initial_state, wheelbase)

    # Create a Vehicle Estimator
    initial_estimate = np.array([0., 0., 0., 10.])  # [x, y, theta, speed]
    initial_covariance = np.diag([1, 1, 0.1, 1])  # Initial covariance matrix
    estimator = VehicleEstimator(initial_estimate, initial_covariance, wheelbase)

    # Parameters
    num_steps = 100
    timestep = 0.1
    steering_angle = np.radians(5)  # Gentle steering angle
    acceleration = 0.1  # Gentle acceleration

    gnss_uncertainty = 1.0  # Reasonable GNSS measurement uncertainty (in meters)
    speedometer_uncertainty = 0.5  # Reasonable speedometer measurement uncertainty (in m/s)
    compass_uncertainty = 0.01 # Reasonable compass measurement uncertainty

    # Arrays for tracking errors
    errors = np.zeros((num_steps, 4))
    covs = np.zeros((num_steps))

    reals = np.zeros((num_steps, 2))
    ests = np.zeros((num_steps, 2))

    # Run the estimation test loop
    for step in range(num_steps):
        # Simulate the vehicle dynamics
        vehicle.noisy_single_track_update(steering_angle, acceleration, timestep)

        # Perform estimation prediction step
        estimator.prediction(steering_angle, acceleration, timestep)

        # Simulate measurements


        # Measure GNSS position
        measured_position = vehicle.get_state()[:2] + np.random.randn(2) * gnss_uncertainty
        estimator.gnss_measurement(measured_position, gnss_uncertainty)

        # Measure speed
        measured_speed = vehicle.get_state()[3] + np.random.randn() * speedometer_uncertainty
        estimator.speedometer_measurement(measured_speed, speedometer_uncertainty)

        # Measure angle
        measured_heading = vehicle.get_state()[2] + np.random.randn() * compass_uncertainty
        estimator.compass_measurement(mod2pi(measured_heading), speedometer_uncertainty)

        # Calculate errors and accumulate
        true_state = vehicle.get_state()
        estimate_state = estimator.get_state()
        errors[step] = np.abs(true_state - estimate_state)
        errors[step,2] = np.abs(mod2pi(errors[step,2]))
        covs[step] = np.linalg.norm(estimator.covariance)

        reals[step] = true_state[:2]
        ests[step] = estimate_state[:2]

    # Plot the errors
    plt.figure()

    # plt.plot(reals[:,0], reals[:,1],label='real')
    # plt.plot(ests[:,0], ests[:,1],label='ests')
    plt.plot(errors[:, 0], label='X')
    plt.plot(errors[:, 1], label='Y')
    plt.plot(errors[:, 2], label='Theta')
    plt.plot(errors[:, 3], label='Speed')
    # plt.plot(covs, label="Cov norm")
    plt.xlabel('Time Steps')
    plt.ylabel('Error')
    plt.title('Estimation Errors')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Calculate mean errors for each element of the state
    mean_errors = np.mean(errors, axis=0)
    print(f"Mean Error - X: {mean_errors[0]}")
    print(f"Mean Error - Y: {mean_errors[1]}")
    print(f"Mean Error - Theta: {mean_errors[2]}")
    print(f"Mean Error - Speed: {mean_errors[3]}")

# Run the test
LoneCarEstimationTest()


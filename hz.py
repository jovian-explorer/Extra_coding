import math
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from mpl_toolkits.mplot3d import Axes3D
from aquarel import load_theme

theme = load_theme("boxy_dark")

# Define constants
L_sun = 3.828 * 10**26  # solar luminosity in Watts
sigma = 5.67 * 10**-8  # Stefan-Boltzmann constant in W/m^2/K^4
T_eff_sun = 5778  # effective temperature of the sun in K
R_sun = 6.957 * 10**8  # solar radius in meters
d_sun = 1.496 * 10**11  # distance from the sun to Earth in meters

# Define function for habitable zone calculation
def habitable_zone(L_star, T_eff):
    inner = math.sqrt(L_star/L_sun) * math.sqrt(T_eff_sun/T_eff) * 0.95
    outer = math.sqrt(L_star/L_sun) * math.sqrt(T_eff_sun/T_eff) * 1.37
    inner_radius = inner * R_sun
    outer_radius = outer * R_sun
    inner_distance = math.sqrt(L_star/L_sun) * math.sqrt(1/inner) * d_sun
    outer_distance = math.sqrt(L_star/L_sun) * math.sqrt(1/outer) * d_sun
    return (inner_radius, outer_radius, inner_distance, outer_distance)

# User inputs
L_star_min = 1e26 #float(input("Enter minimum star luminosity in Watts: "))
L_star_max = 1.2e26 #float(input("Enter maximum star luminosity in Watts: "))
T_eff_min = 2000 #float(input("Enter minimum effective temperature of the star in K: "))
T_eff_max = 4000 #float(input("Enter maximum effective temperature of the star in K: "))
num_points = 2000 #int(input("Enter number of points: "))

# Calculate range of parameter values
L_star_values = np.linspace(L_star_min, L_star_max, num_points)
T_eff_values = np.linspace(T_eff_min, T_eff_max, num_points)
L_star_mesh, T_eff_mesh = np.meshgrid(L_star_values, T_eff_values)

# Create a pool of worker processes
num_cores = mp.cpu_count()
pool = mp.Pool(num_cores)

# Calculate habitable zone parameters for range of parameter values
results = []
for i in range(num_points):
    for j in range(num_points):
        L_star = L_star_values[i]
        T_eff = T_eff_values[j]
        results.append(pool.apply_async(habitable_zone, args=(L_star, T_eff)))

# Get results from pool
inner_radius = np.zeros((num_points, num_points))
outer_radius = np.zeros((num_points, num_points))
inner_distance = np.zeros((num_points, num_points))
outer_distance = np.zeros((num_points, num_points))

for i in range(num_points):
    for j in range(num_points):
        index = i * num_points + j
        inner_radius[i, j], outer_radius[i, j], inner_distance[i, j], outer_distance[i, j] = results[index].get()

# Plot results
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(L_star_mesh/L_sun, T_eff_mesh, inner_radius/R_sun, alpha=0.8)
ax.plot_surface(L_star_mesh/L_sun, T_eff_mesh, outer_radius/R_sun, alpha=0.8)
ax.set_xlabel('Luminosity (L/L_sun)')
ax.set_ylabel('Effective Temperature (K)')
ax.set_zlabel('Habitable Zone Radius (R/R_sun)')
ax.set_title('Habitable Zone Parameter Space')
plt.show()

import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# from enum import Enum

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

z = 0

X, Y, Z = [], [], []


class Data:
    X = np.arange(-2, 3.5, 0.1, dtype=np.float64)
    Y = np.arange(-1.5, 2.0, 0.1, dtype=np.float64)


def objective_function(x, y):
    # print(x.shape, y.shape)
    return 100 * (x - y ** 2) ** 2 + (1 - y) ** 2


def derivative_objective_function(x, y):
    return np.array([200 * (x - y ** 2), 400 * y ** 3 - 400 * x * y + 2 * y - 2])


def Plot_MeshGrid():
    input_range_min, input_range_max = -1.50, 2.0

    x, y = np.meshgrid(Data.X, Data.Y)
    z = objective_function(x, y)
    # print(z.shape)

    ax.plot_surface(x, y, z, cmap='twilight_shifted', edgecolor='none', alpha=0.8)
    # line, = ax.plot(x, y, z)
    # ax.view_init(60, 60)
    # ax.xlabel("X axis")
    # plt.show()

    # return line,


def RMSProp(bounds, n_iter, step_size, lr):
    global z
    solution = bounds[:, 0] + np.random.rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    old_z = 1e8

    velocity = [0 for _ in range(bounds.shape[0])]

    # print(bounds, solution, velocity)

    for it in range(n_iter):
        gradient = derivative_objective_function(solution[0], solution[1])

        for i in range(gradient.shape[0]):
            gradient_square = gradient[i] ** 2.0
            velocity[i] = (velocity[i] * lr) + (gradient_square * (1.0 - lr))

        updated_solution = list()

        for i in range(solution.shape[0]):
            alpha = step_size / (1e-8 + np.sqrt(velocity[i]))

            value = solution[i] - alpha * gradient[i]

            updated_solution.append(value)

        X.append(solution[0])
        Y.append(solution[1])
        solution = np.asarray(updated_solution)
        z = objective_function(solution[0], solution[1])
        Z.append(z)

        # print(f"{it}. f({solution[0], solution[1]}) = {np.round(z, 5)}")

        # ax.scatter3D(solution[0], solution[1], z, color='b')
        # plt.pause(0.05)
        # plt.show()
        if abs(old_z - z) <= 1e-2:
            print(f"Convergence criteria satisfied at iteration number : {it}")
            return [solution, z]
        else:
            old_z = z
    return [solution, z]


bounds = np.asarray([[-1.0, 2.5], [-1.0, 2.5]])

iterations = 500
step_size = 0.01

lr = 0.9

start = time.perf_counter()
best, score = RMSProp(bounds, iterations, step_size, lr)
stop = time.perf_counter()

print(f"Final : f({best[0], best[1]}) = {score}\nExecution time : {stop - start}s")

# print(Z)

line, = ax.plot(X, Y, Z, markersize=5, color='lime')

Plot_MeshGrid()


def update(num):
    line.set_data(X[:num], Y[:num])
    line.set_3d_properties(Z[:num])
    return line,


ani = FuncAnimation(fig, update, frames=100, interval=10, blit=True)
# print(type(Data.X))

plt.show()

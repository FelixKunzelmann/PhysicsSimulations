import random as rd
import matplotlib.pyplot as plt


def randomwalk(n):
    x = [0]
    y = [0]

    for i in range(n):
        step = rd.choice(['N', 'S', 'E', 'W'])
        if step == 'N':
            x.append(x[-1])
            y.append(y[-1] + 1)
        elif step == 'S':
            x.append(x[-1])
            y.append(y[-1] - 1)
        elif step == 'E':
            x.append(x[-1] + 1)
            y.append(y[-1])
        else:
            x.append(x[-1] - 1)
            y.append(y[-1])

    return x, y


n = 1000000
x, y = randomwalk(n)

# Did the random walk ever return to the origin (after the start)?
returned_to_origin = any(xi == 0 and yi == 0 for xi, yi in zip(x[1:], y[1:]))
if returned_to_origin:
    print("The random walk returned to the origin at some point.")
else:
    print("The random walk did not return to the origin.")

plt.plot(x, y)
plt.title(f'Random Walk in 2D with {n} steps')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid()
plt.show()

import numpy as np
import time
import Gnuplot.Gnuplot as gplt

t = np.linspace(0, 10, 100)
y = 2 * np.sin(t + 0.6)
noise = 0.5 * np.random.randn(100)

# set title
gplt.title("Sin wave")

# set axis label
gplt.xlabel("X")
gplt.ylabel("Y")
# or this also leads same result
# gplt.labels('X', 'Y')

# line plot
gplt.plot(t, y, label="ground truth", color="black", dt=3, lw=3)

# scatter plot
gplt.scatter(t, y + noise, label="data", color="#0000ff", style=1)

# fitting
(a, b), _ = gplt.fit(
    t,
    y + noise,
    func="f(x)=a * sin(x + b)",
    via="a,b",
    plot=True,
    label="predicted",
    color="red",
)
print(f"a: {a}, b: {b}")

# finish plotting / add new figure
gplt.show()
gplt.figure()

# optional command
gplt.command("plot sin(x); replot cos(x);")
gplt.show()

# asking gnuplot
time.sleep(1)  # plotting process must be done to get answer
answer = gplt.ask("print pi; print gamma(2.4)")
print(answer)

# set range
gplt.lim([1, 1000], [0.01, 1])
# gplt.xlim(1, 1000)
# gplt.ylim(0.01, 1)

# set scale (linear / log)
gplt.scale("log")
# gplt.xscale('log')
# gplt.yscale('log')

# save figure as image file
gplt.command("plot sin(x)**2; replot cos(x)*cos(x);")
gplt.savefig("fig.png")

input()  # (pause)

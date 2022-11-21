# Gnuplot_python
 The Easy-to-use [PyGnuplot](https://github.com/benschneider/PyGnuplot) Wrapper.

# Installation

- Click `install.bat` (Windows only).
- `pip install -e .`

# How to Use?

## Import
```python
import Gnuplot.Gnuplot as gplt
```
## Set title / axis label
```python
gplt.title("Sin wave")

gplt.xlabel("X")
gplt.ylabel("Y")
# or
gplt.labels('X', 'Y')
```

## Basic Plot
```python
t = np.linspace(0, 10, 100)
y = 2 * np.sin(t + 0.6)
noise = 0.5 * np.random.randn(100)

# line plot
gplt.plot(t, y, label="ground truth", color="black", dt=3, lw=3)

# scatter plot
gplt.scatter(t, y + noise, label="raw data", color="#0000ff", style=1, size=2)
```
## Plot settings
```python
# set range
gplt.xlim(1, 1000)
gplt.ylim(0.01, 1)
# or
gplt.lim([1, 1000], [0.01, 1])

# set scale (linear / log)
gplt.xscale('log')
gplt.yscale('log')
# or
gplt.scale("log")
```

## Fitting

```python
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

gplt.show()
```

## Optional command
```python
gplt.command("plot sin(x); replot cos(x);")
gplt.show()
```

## Ask
```python
answer = gplt.ask("print pi; print gamma(2.4)")
print(answer)
```

## Save image
```python
gplt.command("plot sin(x)**2; replot cos(x)*cos(x);")
gplt.savefig("fig.png")
```
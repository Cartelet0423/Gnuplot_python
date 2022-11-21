"""
The Easy-to-use PyGnuplot Wrapper.

########################################
                EXAMPLE
########################################

import numpy as np
import time
import Gnuplot.Gnuplot as gplt

t = np.linspace(0, 10, 100)
y = 2 * np.sin(t + .6)
noise = 0.5 * np.random.randn(100)

# set title
gplt.title('Sin wave')

# set axis label
gplt.xlabel('X')
gplt.ylabel('Y')
# or this also leads same result
# gplt.labels('X', 'Y')

# line plot
gplt.plot(t, y, label='ground truth', color='black', dt=3, lw=3)

# scatter plot
gplt.scatter(t, y+noise, label='data', color='#0000ff', style=1)

# fitting
(a, b), _ = gplt.fit(t, y+noise, func='f(x)=a * sin(x + b)', via='a,b', plot=True, label='predicted', color='red')
print(f'a: {a}, b: {b}')

# finish plotting / add new figure
gplt.show()
gplt.figure()

# optional command
gplt.command('plot sin(x); replot cos(x);')

# asking gnuplot
time.sleep(1) # plotting process must be done to get answer
answer = gplt.ask(';print pi; print gamma(2.4)')
print(answer)

########################################
"""


import os
import numpy as np
from PyGnuplot import gp

# Preparation For Jupyter
try:

    def exit_register(fun, *args, **kwargs):
        """Decorator that registers at post_execute. After its execution it
        unregisters itself for subsequent runs."""

        def callback():
            fun(*args, **kwargs)
            ip.events.unregister("post_execute", callback)

        ip.events.register("post_execute", callback)

    ip = get_ipython()
except NameError:
    from atexit import register as exit_register


class Gnuplot:

    fig = None
    commands = [[]]
    cnt = [0]
    os.makedirs("tmp", exist_ok=True)

    @classmethod
    def show(cls):
        for fig in cls.commands:
            cls.fig = gp()
            settings = []
            plots = []
            for cmd in fig:
                (settings if cmd.startswith("set ") else plots).append(cmd)
            for cmd in settings + plots:
                cls.fig.c(cmd)
        cls.commands = [[]]
        cls.fig = None
        cls.cnt = [0]

    @classmethod
    def figure(cls):
        cls.fig = gp()
        cls.commands.append([])
        cls.cnt.append(0)

        @exit_register
        def _show():
            cls.show()

    def _figure(self):
        if self.fig is None:
            self.figure()

    @classmethod
    def title(cls, title):
        cls._figure(cls)
        cls.commands[-1].append(f'set title "{title}";')

    @classmethod
    def xlabel(cls, label):
        cls._figure(cls)
        cls.commands[-1].append(f'set xlabel "{label}";')

    @classmethod
    def ylabel(cls, label):
        cls._figure(cls)
        cls.commands[-1].append(f'set ylabel "{label}";')

    @classmethod
    def labels(cls, xlabel, ylabel):
        cls._figure(cls)
        cls.commands[-1].append(f'set xlabel "{xlabel}";')
        cls.commands[-1].append(f'set ylabel "{ylabel}";')

    @classmethod
    def plot(cls, x, y=None, label="", color=None, dt=None, lw=None, lt=None):
        cls._figure(cls)
        if y is None:
            x, y = np.arange(len(x)), x
        cls.fig.save([x, y], filename=f"tmp/tmp{sum(cls.cnt)+1}.dat")
        command = cls._make_style_command(cls, color, dt, lw, lt)
        if cls.cnt[-1] == 0:
            cls.commands[-1].append(
                f"plot 'tmp/tmp{sum(cls.cnt)+1}.dat' u 1:2 w l t '{label}'{command}"
            )
        else:
            cls.commands[-1].append(
                f"replot 'tmp/tmp{sum(cls.cnt)+1}.dat' u 1:2 w l t '{label}'{command}"
            )
        cls.cnt[-1] += 1

    @classmethod
    def scatter(cls, x, y, label="", color=None, style=None, size=None):
        cls._figure(cls)
        cls.fig.save([x, y], filename=f"tmp/tmp{sum(cls.cnt)+1}.dat")
        command = cls._make_style_command(cls, color, style, size, scatter=True)
        if cls.cnt[-1] == 0:
            cls.commands[-1].append(
                f"plot 'tmp/tmp{sum(cls.cnt)+1}.dat' u 1:2 w p t '{label}'{command}"
            )
        else:
            cls.commands[-1].append(
                f"replot 'tmp/tmp{sum(cls.cnt)+1}.dat' u 1:2 w p t '{label}'{command}"
            )
        cls.cnt[-1] += 1

    @classmethod
    def fit(
        cls,
        x,
        y,
        func,
        via,
        limit=1e-9,
        wait=1,
        plot=False,
        label=None,
        color=None,
        dt=None,
        lw=None,
        lt=None,
    ):
        cls._figure(cls)
        result = cls.fig.fit(
            [x, y],
            func=func,
            via=via,
            limit=limit,
            wait=wait,
            filename=f"tmp/tmp{sum(cls.cnt)+1}.dat",
        )
        if plot:
            command = cls._make_style_command(cls, color, dt, lw, lt)
            f, g = func[: func.index("=")].strip(), func[func.index("=") :].strip()
            if label is None:
                label = f
            v = via.split(",")
            if cls.cnt[-1] == 0:
                cls.commands[-1].append(
                    f"{func};{'; '.join([f'{i}={r}' for i, r in zip(v, result[0])])};plot {f} w l t '{label}'{command}"
                )
            else:
                cls.commands[-1].append(
                    f"{func};{'; '.join([f'{i}={r}' for i, r in zip(v, result[0])])};replot {f} w l t '{label}'{command}"
                )
        cls.cnt[-1] += 1

        return result

    @classmethod
    def command(cls, cmd):
        cls._figure(cls)
        cls.commands[-1].append(cmd)
        cls.cnt[-1] += "plot " in cmd

    @classmethod
    def ask(cls, cmd, timeout=0.05):
        cls._figure(cls)
        return cls.fig.a(cmd, timeout=timeout)

    @classmethod
    def xscale(cls, scale):
        cls._figure(cls)
        cls.commands[-1].append(cls._scale(cls, "x", scale))

    @classmethod
    def yscale(cls, scale):
        cls._figure(cls)
        cls.commands[-1].append(cls._scale(cls, "y", scale))

    @classmethod
    def scale(cls, scale):
        cls._figure(cls)
        cls.commands[-1].append(cls._scale(cls, "xy", scale))

    def _scale(self, axis, scale):
        return {
            "linear": f"set nologscale {axis}",
            "log": f"set logscale {axis}",
        }.get(scale, "set nologscale")

    @classmethod
    def xlim(cls, min_, max_):
        cls._figure(cls)
        min_ = "" if min_ is None else min_
        max_ = "" if max_ is None else max_
        cls.commands[-1].append(f"set xr[{min_}:{max_}];")

    @classmethod
    def ylim(cls, min_, max_):
        cls._figure(cls)
        min_ = "" if min_ is None else min_
        max_ = "" if max_ is None else max_
        cls.commands[-1].append(f"set yr[{min_}:{max_}];")

    @classmethod
    def lim(cls, xlim, ylim):
        cls._figure(cls)
        assert len(xlim) == 2 and len(ylim) == 2, "xlim and ylim must have length 2."
        lim = [
            ("" if min_ is None else min_, "" if max_ is None else max_)
            for min_, max_ in [xlim, ylim]
        ]
        cls.commands[-1].append(
            f"set xr[{lim[0][0]}:{lim[0][1]}];set yr[{lim[1][0]}:{lim[1][1]}];"
        )

    @classmethod
    def savefig(cls, filename):
        cls._figure(cls)
        assert filename[-4:] in (
            ".png",
            ".PNG",
            ".eps",
            ".EPS",
        ), "File expansion must be PNG or EPS."
        if filename[-4:] in (".png", ".PNG"):
            t = "pngcairo"
        else:
            t = "postscript eps"
        cls.commands[-1].append(
            f'set terminal {t};set out "{filename}";replot; set terminal wxt;'
        )
        cls.show()

    def _make_style_command(self, color, style, size, lt=None, scatter=False):
        cc = f" lc '{color}' " if color is not None else ""
        if scatter:
            dtc = ""
            tc = f" pt {style}" if style is not None else ""
            sc = f" ps {size}" if size is not None else ""
        else:
            dtc = f" dt {style}" if style is not None else ""
            tc = f" lt {lt}" if lt is not None else ""
            sc = f" lw {size}" if size is not None else ""
        return "".join([cc, dtc, sc, tc])

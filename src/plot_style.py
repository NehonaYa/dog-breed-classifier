"""Барлық графиктерге ортақ стиль: тексерілген түстер палитрасы (түс соқырлығына төзімді),
жұқа белгілер, әлсіз тор сызықтары. Бір серия — бір түс; ерекшелеу — сұр фон + бір түс."""
import matplotlib as mpl

C = {
    "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
    "grid": "#e1e0d9", "axis": "#c3c2b7", "gray": "#c3c2b7",
    # категориялық слоттар (тұрақты ретпен)
    "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
    "magenta": "#e87ba4", "green": "#008300", "violet": "#6250d6", "red": "#e34948",
    # күй түстері (тек дұрыс/қате мағынасында)
    "good": "#0ca30c", "critical": "#d03b3b",
}
SERIES = [C["blue"], C["orange"], C["aqua"], C["yellow"], C["magenta"], C["green"], C["violet"], C["red"]]
SEQ_BLUE = ["#fcfcfb", "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
            "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]


def apply() -> None:
    mpl.rcParams.update({
        "figure.facecolor": C["surface"], "axes.facecolor": C["surface"], "savefig.facecolor": C["surface"],
        "axes.edgecolor": C["axis"], "axes.linewidth": 0.8,
        "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": C["axis"], "ytick.color": C["axis"],
        "xtick.labelcolor": C["ink2"], "ytick.labelcolor": C["ink2"],
        "text.color": C["ink"], "axes.labelcolor": C["ink2"], "axes.titlecolor": C["ink"],
        "axes.titleweight": "semibold", "axes.titlesize": 12, "axes.titlelocation": "left",
        "axes.titlepad": 10, "font.size": 10,
        "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "legend.frameon": False, "lines.linewidth": 2,
        "savefig.dpi": 160, "savefig.bbox": "tight", "figure.dpi": 100,
    })


def seq_cmap():
    return mpl.colors.LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)

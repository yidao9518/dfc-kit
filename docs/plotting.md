# Plotting

Plotting is available from dfc-kit 1.1.0 as an optional extra. Analysis-only
installations remain NumPy-only:

```bash
python -m pip install "dfc-kit[plotting]>=1.1.0"
```

Seven small functions cover network, state and inference displays. They accept
arrays and return Matplotlib `Axes`, so they can share a figure without a
separate layout framework. They do not load participant data, run statistical
tests, select significant endpoints, or choose a state count.

## Shared visual style

The default displays use segmented ROI rings, saturated red/blue signs, pale
individual observations and open-diamond summaries. Effects with samples use
horizontal rainclouds, showing a density above the observations. Pairwise
conditions use amber/teal. Use the same exported palettes across panels:

```python
from dfckit.plotting import (
    publication_style, SYSTEM_PALETTE, CONDITION_PALETTE,
    POSITIVE_COLOR, NEGATIVE_COLOR,
)

with publication_style(font_size=7):
    # Create/compose axes and export figures here.
    ...
```

`publication_style` supplies a white background, thin dark axes, no grid and
no top/right frame. It keeps SVG text editable and embeds TrueType fonts in
PDF. The settings are temporary and restored on leaving the context. Palettes
are display choices, not predefined system identities; callers assign names,
colours and their biological meaning. All colours can be overridden.

| Function | Typical use | Input |
| --- | --- | --- |
| `plot_ring_network` | NBS components or state connections | Ordered ROIs, exact edges, optional signed weights |
| `plot_connectivity_matrix` | FC or cluster-centre matrices | Signed matrix, explicit common colour limit |
| `plot_paired` | OFF/ON FC, occupancy or dwell | Aligned participant-by-condition values |
| `plot_effects` | FC/PCA changes or regression coefficients | Estimates, optional errors and individual observations |
| `plot_association` | Imaging/clinical relationship | Paired observations, optional supplied fit |
| `plot_curve` | K comparison, window sensitivity or variance curves | Ordered x/y values, optional errors |
| `plot_null_distribution` | Permutation reference distribution | Actual null draws and observed statistic |

## Rings and matrices

`plot_ring_network` draws every supplied ROI-level edge inside a continuous
annulus. Nodes follow the supplied order counterclockwise, with the first node
centred at the top by default (`start_angle=90`). Node order, node colours, and
edges are explicit; the function does
not threshold, select, sort, or aggregate connections. Thin radial lines mark
individual ROI boundaries, including boundaries between adjacent ROIs with the
same system colour.
Unconnected ROIs remain in the supplied frame with pale segments. Set
`unconnected_alpha=1` to retain their full saturation, or
`label_unconnected=False` to label only ROIs incident to supplied edges.

```python
import matplotlib.pyplot as plt

from dfckit.plotting import plot_ring_network

nodes = ("V1-L", "V1-R", "PUT-L", "PUT-R")
edges = (("V1-L", "PUT-L"), ("V1-R", "PUT-R"))
values = (-0.18, 0.12)
colours = {
    "V1-L": SYSTEM_PALETTE[0],
    "V1-R": SYSTEM_PALETTE[0],
    "PUT-L": SYSTEM_PALETTE[3],
    "PUT-R": SYSTEM_PALETTE[3],
}

fig, ax = plt.subplots(figsize=(4, 4))
plot_ring_network(
    nodes,
    edges,
    edge_values=values,
    value_limit=0.25,
    node_colors=colours,
    ax=ax,
)
fig.savefig("network.svg", bbox_inches="tight")
```

`value_limit` is deliberately required for signed edge values. Reuse the same
limit for panels that share a visual scale. Values beyond it raise an error
instead of being silently clipped. The returned Matplotlib axes can be combined
with other subplots or exported by the caller as SVG, PDF, or PNG.

For matrices, `NaN` or masked entries are displayed as unselected/unavailable
cells, not zero FC. The diagonal and all other supplied values are preserved. Reuse
`value_limit` and `node_colors` across matrices and rings to keep the scale and
anatomical colours consistent. The default matrix colour map is diverging and
centred on zero, using the same `NEGATIVE_COLOR` and `POSITIVE_COLOR` endpoints
as the rings. Supply `cmap=` to override it. No absolute-value transform or
ROI reordering is performed.
For stronger negative-side colour contrast, explicitly pass a smaller
`negative_limit` (for example, `value_limit=1.2, negative_limit=0.5`). The
colour bar then displays the asymmetric limits, zero remains neutral and
values outside either bound raise an error. Reuse both limits across panels
being compared; the plotted FC values are unchanged.

```python
import numpy as np
from dfckit.plotting import plot_connectivity_matrix

matrix = np.array([[np.nan, -0.18], [-0.18, np.nan]])
ax = plot_connectivity_matrix(
    matrix, value_limit=0.25, labels=["Visual", "Putamen"],
    node_colors=["#0072B2", "#AA3377"],
    group_sizes=[1, 1], colorbar_label="FC (Fisher-z)",
)
```

For multiple matrices, pass `colorbar=False` and build a shared colour bar with
`fig.colorbar(ax.images[0], ax=your_axes)`. For a ring, magnitude controls width;
for a matrix, magnitude controls colour. Equal limits do not make these visual
channels identical.

## Paired observations and effect estimates

Pass one row per participant, with the conditions in columns. Align participant
IDs **before** calling the plotter: it connects rows and cannot infer identities.
Missing, infinite and masked observations are rejected, not silently dropped.
Points are jittered only along the category axis, using a local seeded generator.
The jitter half-width must be in `[0, 0.5)` so points stay within their condition.

```python
from dfckit.plotting import plot_paired, plot_effects

# Illustrative values only; columns have matching participant rows.
values = np.array([[0.10, 0.15], [0.25, 0.31], [0.19, 0.17]])
means = values.mean(axis=0)
se = values.std(axis=0, ddof=1) / np.sqrt(len(values))
ax = plot_paired(
    values, labels=["OFF", "ON"], colors=CONDITION_PALETTE[:2],
    estimates=means, errors=se, error_label="SE",
)
ax.set_ylabel("FC (Fisher-z)")
ax.legend()

delta = values[:, 1] - values[:, 0]
ax = plot_effects(
    [delta.mean()], ["Visual internal"], samples=[delta],
    errors=[delta.std(ddof=1) / np.sqrt(len(delta))], error_label="SE",
    colors=[NEGATIVE_COLOR], density=True,
)
ax.set_xlabel("ON minus OFF (Fisher-z)")
ax.legend()
```

Summary estimates and errors are always supplied explicitly; the plotting
functions do not compute means, SE, CI, coefficients or adjusted contrasts.
`errors` are **non-negative distances from the estimate**, shaped `(n,)` for
symmetric errors or `(2, n)` for lower/upper distances. For interval bounds
`lower` and `upper`, use `np.array([estimate - lower, upper - estimate])`.
`error_label` is required when errors are supplied; examples are `SE`, `SD` and
`95% CI`. It labels the errorbar legend handle; call `ax.legend()` to show it.
An interval not enclosing the displayed estimate cannot be represented as
error distances; plot its endpoints separately instead.

`plot_effects` optionally adds supplied `annotations` to the right of each row
(for example, precomputed q values), and individual `samples` behind the summary.
Samples should use the same endpoint and units as the estimate. It does not
assign significance stars or infer the meaning of a supplied error.
The default half-density uses a display-only Gaussian KDE. Every observation
is retained, including extremes; singleton and constant samples are shown as
points without an artificial density. Set `density=False` for points only.
No automatic zoom or removal of outliers is performed.

## Associations, comparison curves and permutation distributions

```python
from dfckit.plotting import plot_association, plot_curve, plot_null_distribution

ax = plot_association([1, 2, 3], [3, 4, 6], fit=(1.5, 1.3))
ax.set(xlabel="Imaging measure", ylabel="Clinical score")

ax = plot_curve([2, 3, 4, 5], [0.9, 0.7, 0.6, 0.55])
ax.set(xlabel="Number of states K", ylabel="Held-out error")

# Replace these illustrative draws with the actual permutation output.
ax = plot_null_distribution([-0.4, -0.1, 0.0, 0.2, 0.3], 0.25,
                            bins=5, tail="greater")
ax.legend()
```

`plot_association` only draws a fit if `(slope, intercept)` is supplied. It does
not fit a regression or make a raw scatter into an adjusted association. Use
model-appropriate residuals or predictions when showing covariate-adjusted
results, and label them accordingly.

`plot_association(..., marginals=True)` adds pale filled distributions above
and to the right of the scatter. Leave room for these marginal axes in your
layout; they are accessible through `ax.child_axes`. The regression line is
still supplied explicitly, and the original observations are unchanged.

`plot_curve` retains the supplied order. It has no automatic elbow detector or
"optimal K" marker. Optional `errors` follow the same explicit convention as
`plot_effects`.

`plot_null_distribution` displays histogram counts from the supplied draws.
`tail="greater"` highlights draws greater than or equal to the observation;
`"less"` highlights those less than or equal to it. `"two-sided"` uses
`abs(draw) >= abs(observed)`; choose this only for an absolute-statistic test.
There is no null simulation, smoothing, p-value calculation, or multiplicity
correction inside the plotter. Use the same null statistic as your inference,
including its maximum-statistic transform where applicable.
The highlighted bars count extreme draws within each histogram bin; a boundary
bin can straddle the observation line without all values in that bin being
extreme. The highlighted count includes only the draws satisfying the chosen
tail rule, including ties.

## Composition and export

All seven plot functions accept `ax=` and return that same axes. Use `plt.subplots` or
`GridSpec` for composition, and `fig.savefig` for PNG, PDF and SVG. Fonts, axis
labels, legends, page layout and manuscript-specific captions stay with the
caller. The package does not modify global Matplotlib settings or select a
backend. For headless use, choose `MPLBACKEND=Agg` before running Python.

The source checkout includes a complete **synthetic-only** gallery and a
separate enlarged association/marginal-density example:

```bash
python examples/plotting_gallery.py --output /tmp/dfckit-plotting-gallery
```

It writes PNG, editable-text SVG and PDF. The eighth gallery panel demonstrates
composing an R² decomposition directly with Matplotlib from supplied numbers.
None of the example numbers are study results.
Matplotlib is imported only when drawing, so importing `dfckit.plotting` does
not require the optional dependency.

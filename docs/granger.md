# Conditional Granger predictability

Install the inference dependency before running this analysis:

```bash
python -m pip install 'dfc-kit[inference]'
```

`conditional_granger()` fits pooled linear VAR models while constructing every
lagged row inside one contiguous retained-frame segment. Its Geweke statistic is
the log ratio of restricted to unrestricted target-residual variance (or
generalized variance for a multivariate target). The restricted model removes
source lags and retains target and conditioning lags.

```python
from dfckit.connectivity import conditional_granger

result = conditional_granger(
    run,
    source=[0],
    target=[1],
    conditioning=[2, 3],
    lag_order=2,
)
print(result.geweke, result.spectral_radius, result.whiteness.min_ljung_box_p)
```

The result reports residual degrees of freedom, VAR spectral radius, and a
Ljung--Box residual-autocorrelation diagnostic. Segment standardization is
enabled by default. These estimates describe lagged linear predictability in the
processed time series and do not by themselves establish neural causality.

References: Geweke (1982), *Journal of the American Statistical Association*,
doi:10.1080/01621459.1982.10477803; Barnett and Seth (2014), *Journal of
Neuroscience Methods*, doi:10.1016/j.jneumeth.2013.10.018.

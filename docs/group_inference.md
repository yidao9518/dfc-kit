# Independent group inference

`infer-independent-endpoints` compares the same named endpoints across two
independent participant cohorts. The caller declares both group labels, their
subtraction order, any numeric covariates, and the FDR family. The command does
not infer cohort membership, pairing, or covariate meaning from filenames.

## Input artifacts

Both JSON inputs use the named endpoint row format produced by commands such as
`summarize-store`, `lowrank-endpoints`, and `static-fc-endpoints`. Each row must
contain a subject, endpoint, finite value, and optional acquisition and endpoint
metadata. Endpoint names and metadata must agree between groups.

The same textual subject identifier may occur in both cohorts: independent
identity is keyed by `(group, subject)`. Within one cohort, repeated
acquisitions are averaged per participant by default. Use
`--within-group-aggregation error` to reject them instead. Temporal samples are
never concatenated across acquisitions.

## Covariate model

With optional covariates, the TSV must contain one row per group and subject:

```text
group  subject  age  sex
PD_OFF sub-001  64   1
HC     sub-001  61   1
```

Requested covariates must be finite numeric columns. For each endpoint,
complete-case covariates are centered and divided by their sample SD over both
groups together. The model is

```text
endpoint = intercept + group_A_indicator + covariates + error
```

The group coefficient therefore estimates `group A - group B` at the combined
sample mean of each covariate. HC3 sandwich standard errors and finite-sample
Student-t p values are calculated through the package's common OLS
implementation.

## Outputs

Every tested endpoint reports:

- sample size and unadjusted mean in each group;
- adjusted raw and outcome-standardized group coefficients;
- HC3 standard error, residual degrees of freedom, finite-sample Student-t
  confidence interval, t statistic, and p value;
- unadjusted Hedges' g;
- full-model in-sample R2;
- incremental in-sample R2 from adding the group indicator to the
  covariate-only model;
- Benjamini-Hochberg q value and status within the declared endpoint family.

Hedges' g and the group means are descriptive, whereas the group coefficient
is the covariate-adjusted effect. Incremental R2 is a fitted-sample quantity,
not held-out predictive performance.

## Command-line example

```bash
dfc-kit infer-independent-endpoints \
  pd-off-static-fc.json \
  hc-static-fc.json \
  results/pd-off-vs-hc.json \
  --group-a PD_OFF \
  --group-b HC \
  --fdr-family "static FC group comparison" \
  --covariates group-covariates.tsv \
  --covariate age \
  --covariate sex
```

Use `infer-paired-endpoints` instead when both conditions belong to the same
participants.

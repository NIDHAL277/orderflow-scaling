# ADR 0005 — Volatility: extraday horizon; principal estimator decided by simulation

**Status**: accepted (2026-11-08). The full simulation study then made Whittle the principal
estimator, a deviation from the rule recorded as D1 (2026-11-10, `config/deviations.md`).

**Context**: the paper states the volatility prediction `2H0 − 3/2` for the *extraday*
volatility (Remark 4.2) and quotes intraday estimates near 0.25 as a different object.
Preliminary simulations of rough Bergomi show that GJR moments on daily RV are biased
upward near zero (≈ +0.13 to +0.19 for H ∈ {0.05, 0.1}) because daily integration smooths
the path, while the Whittle estimator of Fukasawa, Takabatake & Westphal stays close.

**Decision**: daily horizon primary; intraday descriptive. The principal estimator follows
the rule of `config/preregistration.md` §4.1, applied to the full simulation study before
the tag `prereg-v1`; both estimators are always computed and bootstrapped. GJR lags extend
to 50 days (the bias falls with longer lags).

**Consequences**: H2 may be "not decided" if neither estimator is reliable near zero; the
report then states the sample size that would be needed.

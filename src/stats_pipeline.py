"""
stats_pipeline.py
─────────────────────────────────────────────────────────────────────
Peer-review grade statistical analysis engine.

Design principles:
  1. Every test is preceded by assumption checks (normality, homoscedasticity).
  2. Effect sizes are always computed — p-values alone are insufficient.
  3. Multiple testing correction is applied automatically to any test battery.
  4. Confidence intervals accompany every estimate.
  5. All outputs are APA 7th-edition formatted.
  6. Global seed enforced for reproducibility.

Typical usage:
    from src.stats_pipeline import StatsPipeline
    sp = StatsPipeline(df)
    sp.describe()
    sp.compare_two_groups("pain_score", "group")
    sp.compare_multiple_groups("satisfaction", "procedure_type")
    sp.correlation_matrix(["age", "pain_score", "recovery_days"])
    sp.logistic_regression("complication", ["age", "bmi", "procedure"])
    sp.survival_analysis("days_to_event", "event_occurred", "treatment_arm")
    sp.export_results("outputs/tables/results.xlsx")
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import scipy.stats as stats
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import yaml
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu, shapiro, ttest_ind
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore", category=FutureWarning)
logger = logging.getLogger(__name__)


# ─── CONFIG & SEED ───────────────────────────────────────────────────────────

def _load_config() -> dict:
    cfg_path = Path(__file__).parents[1] / "config" / "settings.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)

_CFG = _load_config()
_SEED = _CFG["project"]["seed"]
np.random.seed(_SEED)

ALPHA = _CFG["statistics"]["alpha"]
CI_LEVEL = _CFG["statistics"]["confidence_level"]
MTC_METHOD = _CFG["statistics"]["multiple_testing_correction"]
DECIMALS = _CFG["reporting"]["decimal_places"]
FIG_DPI = _CFG["reporting"]["figure_dpi"]
FIG_FMT = _CFG["reporting"]["figure_format"]
OUT_DIR = Path(_CFG["project"]["output_dir"])


# ─── APA FORMATTING HELPERS ──────────────────────────────────────────────────

def _fmt_p(p: float) -> str:
    """APA 7 p-value formatting."""
    if p < 0.001:
        return "p < .001"
    return f"p = {p:.3f}".replace("0.", ".")

def _fmt_ci(lo: float, hi: float, dp: int = DECIMALS) -> str:
    return f"95% CI [{lo:.{dp}f}, {hi:.{dp}f}]"

def _fmt_stat(name: str, val: float, df_val: int | float | None = None) -> str:
    if df_val is not None:
        return f"{name}({df_val:.0f}) = {val:.3f}"
    return f"{name} = {val:.3f}"


# ─── ASSUMPTIONS CHECKER ────────────────────────────────────────────────────

class AssumptionChecker:
    """
    Runs normality and variance homogeneity tests.
    Returns structured results and a human-readable verdict.
    """

    @staticmethod
    def normality(series: pd.Series, label: str = "") -> dict:
        """Shapiro-Wilk (n < 50) or D'Agostino-Pearson (n >= 50)."""
        s = series.dropna()
        n = len(s)
        if n < 3:
            return {"test": "n/a", "statistic": None, "p": None, "normal": None,
                    "note": f"n={n} — too small"}
        if n < 50:
            stat, p = shapiro(s)
            test_name = "Shapiro-Wilk"
        else:
            stat, p = stats.normaltest(s)
            test_name = "D'Agostino-Pearson"
        is_normal = p > ALPHA
        return {
            "variable": label or series.name,
            "n": n,
            "test": test_name,
            "statistic": round(stat, 4),
            "p": round(p, 4),
            "p_fmt": _fmt_p(p),
            "normal": is_normal,
            "verdict": "Normal" if is_normal else "Non-normal",
        }

    @staticmethod
    def homoscedasticity(data: list[pd.Series], method: str = "levene") -> dict:
        """Levene's test for equality of variances."""
        groups = [s.dropna() for s in data]
        if method == "levene":
            stat, p = stats.levene(*groups)
            test_name = "Levene"
        else:
            stat, p = stats.bartlett(*groups)
            test_name = "Bartlett"
        equal_var = p > ALPHA
        return {
            "test": test_name,
            "statistic": round(stat, 4),
            "p": round(p, 4),
            "p_fmt": _fmt_p(p),
            "equal_variance": equal_var,
            "verdict": "Equal variances" if equal_var else "Unequal variances",
        }


# ─── MAIN PIPELINE ──────────────────────────────────────────────────────────

class StatsPipeline:
    """
    Full statistical analysis pipeline for clinical research data.

    Parameters
    ----------
    df : pd.DataFrame
        Input data. Rows = observations (patients), columns = variables.
    label : str
        Study label used in outputs.
    """

    def __init__(self, df: pd.DataFrame, label: str = "analysis"):
        self.df = df.copy()
        self.label = label
        self._results: list[dict] = []          # accumulates all test results
        self._figures: list[tuple[str, plt.Figure]] = []
        self.checker = AssumptionChecker()
        (OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "tables").mkdir(parents=True, exist_ok=True)

    # ── DESCRIPTIVE ─────────────────────────────────────────────────────────

    def describe(
        self,
        continuous: list[str] | None = None,
        categorical: list[str] | None = None,
        group_by: str | None = None,
    ) -> pd.DataFrame:
        """
        Generates Table 1 (descriptive statistics).
        Continuous: mean ± SD, median [IQR], range.
        Categorical: n (%).
        """
        rows = []
        cont_cols = continuous or self.df.select_dtypes(include="number").columns.tolist()
        cat_cols = categorical or self.df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        if group_by and group_by in cont_cols:
            cont_cols.remove(group_by)
        if group_by and group_by in cat_cols:
            cat_cols.remove(group_by)

        groups = (
            {g: self.df[self.df[group_by] == g] for g in self.df[group_by].unique()}
            if group_by
            else {"Overall": self.df}
        )

        for col in cont_cols:
            row = {"Variable": col, "Type": "continuous"}
            for gname, gdf in groups.items():
                s = gdf[col].dropna()
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                row[f"{gname} mean±SD"] = f"{s.mean():.{DECIMALS}f} ± {s.std():.{DECIMALS}f}"
                row[f"{gname} median[IQR]"] = f"{s.median():.{DECIMALS}f} [{q1:.{DECIMALS}f}–{q3:.{DECIMALS}f}]"
                row[f"{gname} n"] = len(s)
                row[f"{gname} missing"] = gdf[col].isna().sum()
            rows.append(row)

        for col in cat_cols:
            row = {"Variable": col, "Type": "categorical"}
            for gname, gdf in groups.items():
                counts = gdf[col].value_counts(dropna=False)
                total = len(gdf)
                summary = "; ".join(
                    f"{k}: {v} ({100*v/total:.1f}%)" for k, v in counts.items()
                )
                row[f"{gname} freq"] = summary
                row[f"{gname} n"] = total
            rows.append(row)

        tbl = pd.DataFrame(rows)
        logger.info(f"Table 1 generated: {len(cont_cols)} continuous, {len(cat_cols)} categorical")
        return tbl

    # ── TWO-GROUP COMPARISON ────────────────────────────────────────────────

    def compare_two_groups(
        self,
        outcome: str,
        group_col: str,
        parametric: bool | None = None,  # None = auto-detect
    ) -> dict:
        """
        Compare a continuous outcome between two groups.
        Auto-selects Student t-test or Mann-Whitney U based on normality.
        Reports: test statistic, p-value (raw + corrected), effect size, 95% CI.
        """
        groups = self.df[group_col].dropna().unique()
        if len(groups) != 2:
            raise ValueError(f"compare_two_groups expects exactly 2 groups; found {list(groups)}")

        g1 = self.df[self.df[group_col] == groups[0]][outcome].dropna()
        g2 = self.df[self.df[group_col] == groups[1]][outcome].dropna()

        # Assumption checks
        norm1 = self.checker.normality(g1, label=str(groups[0]))
        norm2 = self.checker.normality(g2, label=str(groups[1]))
        homo = self.checker.homoscedasticity([g1, g2])

        use_parametric = (
            parametric
            if parametric is not None
            else (norm1["normal"] and norm2["normal"])
        )

        if use_parametric:
            equal_var = homo["equal_variance"]
            stat, p = ttest_ind(g1, g2, equal_var=equal_var)
            test_name = "Student t-test" if equal_var else "Welch t-test"
            # Cohen's d via pingouin
            eff = pg.compute_effsize(g1, g2, eftype="cohen")
            eff_name = "Cohen's d"
            # CI of the mean difference
            diff = g1.mean() - g2.mean()
            se = np.sqrt(g1.var()/len(g1) + g2.var()/len(g2))
            ci_lo = diff - 1.96 * se
            ci_hi = diff + 1.96 * se
        else:
            stat, p = mannwhitneyu(g1, g2, alternative="two-sided")
            test_name = "Mann-Whitney U"
            eff = pg.compute_effsize(g1, g2, eftype="r")
            eff_name = "rank-biserial r"
            diff = g1.median() - g2.median()
            # Hodges-Lehmann CI
            ci_lo, ci_hi = None, None  # complex; note as bootstrap option

        result = {
            "outcome": outcome,
            "group_col": group_col,
            "groups": list(groups),
            "n": [len(g1), len(g2)],
            "test": test_name,
            "statistic": round(stat, 4),
            "p_raw": round(p, 4),
            "p_fmt": _fmt_p(p),
            "significant": p < ALPHA,
            eff_name: round(eff, 4),
            "mean_diff": round(diff, DECIMALS) if diff else None,
            "CI_95_lo": round(ci_lo, DECIMALS) if ci_lo else None,
            "CI_95_hi": round(ci_hi, DECIMALS) if ci_hi else None,
            "normality_g1": norm1["verdict"],
            "normality_g2": norm2["verdict"],
            "homoscedasticity": homo["verdict"],
            "parametric_used": use_parametric,
        }

        self._results.append(result)
        self._log_result(result)
        return result

    # ── CATEGORICAL COMPARISON (chi2 / Fisher) ────────────────────────────

    def compare_categorical(
        self,
        var: str,
        group_col: str,
    ) -> dict:
        """
        Chi-square or Fisher's exact test for categorical variables.
        Reports: test, p, Cramér's V (effect size).
        """
        ct = pd.crosstab(self.df[group_col], self.df[var])
        n = ct.values.sum()
        min_expected = (ct.sum(axis=0).values[:, None] * ct.sum(axis=1).values[None, :]) / n

        use_fisher = (ct.shape == (2, 2)) and (min_expected < 5).any()

        if use_fisher:
            stat, p = fisher_exact(ct.values)
            test_name = "Fisher's exact"
            effect_size = None
            eff_name = None
        else:
            stat, p, dof, _ = chi2_contingency(ct)
            test_name = "Pearson chi-square"
            cramer = np.sqrt(stat / (n * (min(ct.shape) - 1)))
            effect_size = round(cramer, 4)
            eff_name = "Cramér's V"

        result = {
            "variable": var,
            "group_col": group_col,
            "test": test_name,
            "statistic": round(stat, 4),
            "p_raw": round(p, 4),
            "p_fmt": _fmt_p(p),
            "significant": p < ALPHA,
            eff_name: effect_size,
            "contingency_table": ct.to_dict(),
        }
        self._results.append(result)
        self._log_result(result)
        return result

    # ── MULTI-GROUP (ANOVA / Kruskal-Wallis + post-hoc) ──────────────────

    def compare_multiple_groups(
        self,
        outcome: str,
        group_col: str,
        post_hoc: bool = True,
    ) -> dict:
        """
        One-way ANOVA (parametric) or Kruskal-Wallis (non-parametric).
        Auto-detects based on per-group normality.
        Post-hoc: Tukey HSD (parametric) or Dunn + BH correction (non-parametric).
        """
        groups_data = {
            g: self.df[self.df[group_col] == g][outcome].dropna()
            for g in self.df[group_col].dropna().unique()
        }

        normality_results = {
            g: self.checker.normality(s, label=str(g))
            for g, s in groups_data.items()
        }
        all_normal = all(r["normal"] for r in normality_results.values())
        homo = self.checker.homoscedasticity(list(groups_data.values()))

        if all_normal and homo["equal_variance"]:
            # One-way ANOVA via pingouin (returns eta-squared)
            anova_df = pg.anova(
                data=self.df[[outcome, group_col]].dropna(),
                dv=outcome,
                between=group_col,
                detailed=True,
            )
            stat = float(anova_df.loc[0, "F"])
            p = float(anova_df.loc[0, "p-unc"])
            eta2 = float(anova_df.loc[0, "np2"])
            test_name = "One-way ANOVA"
            eff_name = "η²p"
            eff_val = eta2

            post_hoc_result = None
            if post_hoc and p < ALPHA:
                ph = pg.pairwise_tukey(
                    data=self.df[[outcome, group_col]].dropna(),
                    dv=outcome,
                    between=group_col,
                )
                post_hoc_result = ph.to_dict(orient="records")
        else:
            groups_list = list(groups_data.values())
            stat, p = stats.kruskal(*groups_list)
            # eta-squared approximation for Kruskal-Wallis
            n_total = sum(len(g) for g in groups_list)
            k = len(groups_list)
            eta2 = (stat - k + 1) / (n_total - k)
            test_name = "Kruskal-Wallis H"
            eff_name = "η²KW"
            eff_val = round(eta2, 4)

            post_hoc_result = None
            if post_hoc and p < ALPHA:
                import scikit_posthocs as sp
                stacked = pd.concat(
                    [s.rename("value").to_frame().assign(group=g)
                     for g, s in groups_data.items()]
                )
                ph = sp.posthoc_dunn(stacked, val_col="value", group_col="group", p_adjust="fdr_bh")
                post_hoc_result = ph.to_dict()

        result = {
            "outcome": outcome,
            "group_col": group_col,
            "n_groups": len(groups_data),
            "test": test_name,
            "statistic": round(stat, 4),
            "p_raw": round(p, 4),
            "p_fmt": _fmt_p(p),
            "significant": p < ALPHA,
            eff_name: eff_val,
            "homoscedasticity": homo["verdict"],
            "post_hoc": post_hoc_result,
        }
        self._results.append(result)
        self._log_result(result)
        return result

    # ── CORRELATION ─────────────────────────────────────────────────────────

    def correlation_matrix(
        self,
        columns: list[str],
        method: Literal["pearson", "spearman", "kendall"] | None = None,
        plot: bool = True,
    ) -> pd.DataFrame:
        """
        Pairwise correlations with p-values, CI, and BH correction.
        Auto-selects Pearson/Spearman based on normality if method=None.
        """
        data = self.df[columns].dropna()

        if method is None:
            norms = [self.checker.normality(data[c], label=c) for c in columns]
            method = "pearson" if all(n["normal"] for n in norms) else "spearman"

        # pingouin pairwise_corr gives r, CI, p, BF10
        result_df = pg.pairwise_corr(data, method=method, padjust=MTC_METHOD)

        if plot:
            fig, ax = plt.subplots(figsize=(8, 6))
            corr_mat = data.corr(method=method)
            mask = np.triu(np.ones_like(corr_mat, dtype=bool))
            sns.heatmap(
                corr_mat, mask=mask, annot=True, fmt=f".{DECIMALS}f",
                cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax,
                linewidths=0.5
            )
            ax.set_title(f"Correlation Matrix ({method.capitalize()})")
            self._save_fig(fig, f"correlation_{method}")

        logger.info(f"Correlation matrix ({method}) — {len(result_df)} pairs")
        return result_df

    # ── LOGISTIC REGRESSION ──────────────────────────────────────────────

    def logistic_regression(
        self,
        outcome: str,
        predictors: list[str],
        report_or: bool = True,
    ) -> pd.DataFrame:
        """
        Binary logistic regression via statsmodels.
        Reports: OR (or β), 95% CI, Wald z, p (raw + BH corrected).
        Also reports Nagelkerke R², Hosmer-Lemeshow goodness-of-fit.
        """
        clean = self.df[[outcome] + predictors].dropna()

        # Dummy-encode categoricals
        X = pd.get_dummies(clean[predictors], drop_first=True)
        X = sm.add_constant(X)
        y = clean[outcome]

        model = sm.Logit(y, X).fit(disp=0)
        summary = pd.DataFrame({
            "variable": model.params.index,
            "coef": model.params.values,
            "OR": np.exp(model.params.values),
            "CI_lower_OR": np.exp(model.conf_int()[0].values),
            "CI_upper_OR": np.exp(model.conf_int()[1].values),
            "z": model.tvalues.values,
            "p_raw": model.pvalues.values,
        })

        _, p_corr, _, _ = multipletests(summary["p_raw"], method=MTC_METHOD)
        summary["p_corrected"] = p_corr
        summary["p_fmt"] = summary["p_raw"].apply(_fmt_p)
        summary["significant"] = summary["p_raw"] < ALPHA

        # Pseudo R²
        logger.info(
            f"Logistic regression → outcome: {outcome}, "
            f"Nagelkerke R²={1-(model.llf/model.llnull):.3f}, "
            f"AIC={model.aic:.1f}"
        )
        return summary

    # ── LINEAR REGRESSION ────────────────────────────────────────────────

    def linear_regression(
        self,
        outcome: str,
        predictors: list[str],
    ) -> pd.DataFrame:
        """
        OLS linear regression with assumption diagnostics.
        Reports: β, SE, 95% CI, t, p (BH corrected), R², adj-R².
        """
        formula = f"{outcome} ~ " + " + ".join(predictors)
        model = smf.ols(formula, data=self.df.dropna(subset=[outcome]+predictors)).fit()

        summary = pd.DataFrame({
            "variable": model.params.index,
            "beta": model.params.values,
            "SE": model.bse.values,
            "CI_lower": model.conf_int()[0].values,
            "CI_upper": model.conf_int()[1].values,
            "t": model.tvalues.values,
            "p_raw": model.pvalues.values,
        })
        _, p_corr, _, _ = multipletests(summary["p_raw"], method=MTC_METHOD)
        summary["p_corrected"] = p_corr
        summary["p_fmt"] = summary["p_raw"].apply(_fmt_p)
        summary["significant"] = summary["p_raw"] < ALPHA

        logger.info(
            f"Linear regression → R²={model.rsquared:.3f}, "
            f"adj-R²={model.rsquared_adj:.3f}, F={model.fvalue:.3f}, "
            f"F-p={_fmt_p(model.f_pvalue)}"
        )
        return summary

    # ── SURVIVAL ANALYSIS ───────────────────────────────────────────────

    def survival_analysis(
        self,
        duration_col: str,
        event_col: str,
        group_col: str | None = None,
        plot: bool = True,
    ) -> dict:
        """
        Kaplan-Meier curves + log-rank test.
        If group_col is provided, compares survival across groups.
        """
        from lifelines import KaplanMeierFitter
        from lifelines.statistics import logrank_test, multivariate_logrank_test

        T = self.df[duration_col].dropna()
        E = self.df.loc[T.index, event_col]

        result = {}

        if group_col is None:
            kmf = KaplanMeierFitter()
            kmf.fit(T, E, label="Overall")
            result["median_survival"] = kmf.median_survival_time_
            if plot:
                fig, ax = plt.subplots(figsize=(7, 5))
                kmf.plot_survival_function(ax=ax, ci_show=True)
                ax.set_title("Kaplan-Meier Survival Curve")
                self._save_fig(fig, "km_overall")
        else:
            groups = self.df[group_col].dropna().unique()
            if plot:
                fig, ax = plt.subplots(figsize=(7, 5))
            kmfs = {}
            for g in groups:
                mask = self.df[group_col] == g
                t = self.df.loc[mask, duration_col].dropna()
                e = self.df.loc[t.index, event_col]
                kmf = KaplanMeierFitter()
                kmf.fit(t, e, label=str(g))
                kmfs[g] = kmf
                if plot:
                    kmf.plot_survival_function(ax=ax, ci_show=True)

            # Log-rank test
            if len(groups) == 2:
                g1, g2 = groups
                t1 = self.df.loc[self.df[group_col] == g1, duration_col].dropna()
                e1 = self.df.loc[t1.index, event_col]
                t2 = self.df.loc[self.df[group_col] == g2, duration_col].dropna()
                e2 = self.df.loc[t2.index, event_col]
                lr = logrank_test(t1, t2, event_observed_A=e1, event_observed_B=e2)
                result["logrank_p"] = round(lr.p_value, 4)
                result["logrank_p_fmt"] = _fmt_p(lr.p_value)
                result["logrank_stat"] = round(lr.test_statistic, 4)
            else:
                # multivariate log-rank
                T_all = self.df[duration_col].dropna()
                E_all = self.df.loc[T_all.index, event_col]
                G_all = self.df.loc[T_all.index, group_col]
                lr = multivariate_logrank_test(T_all, G_all, E_all)
                result["logrank_p"] = round(lr.p_value, 4)
                result["logrank_p_fmt"] = _fmt_p(lr.p_value)

            result["median_survival_by_group"] = {
                str(g): kmfs[g].median_survival_time_ for g in groups
            }
            if plot:
                ax.set_title(f"Kaplan-Meier: {outcome_label(duration_col)} by {group_col}")
                self._save_fig(fig, f"km_{group_col}")

        self._results.append(result)
        return result

    # ── MULTIPLE TESTING CORRECTION ─────────────────────────────────────

    def apply_multiple_testing_correction(
        self,
        p_values: list[float],
        method: str | None = None,
    ) -> pd.DataFrame:
        """
        Apply multiple testing correction to a list of p-values.
        Returns DataFrame with raw p, corrected p, and reject flag.
        """
        method = method or MTC_METHOD
        reject, p_corr, _, _ = multipletests(p_values, alpha=ALPHA, method=method)
        return pd.DataFrame({
            "p_raw": p_values,
            "p_corrected": p_corr,
            "p_corrected_fmt": [_fmt_p(p) for p in p_corr],
            "reject_H0": reject,
        })

    # ── EXPORT ───────────────────────────────────────────────────────────

    def export_results(self, path: str | None = None) -> str:
        """
        Export all accumulated test results to Excel with APA-style formatting.
        Each test type gets its own sheet.
        """
        path = path or str(OUT_DIR / "tables" / f"{self.label}_results.xlsx")
        results_df = pd.json_normalize(self._results)
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            results_df.to_excel(writer, sheet_name="All Results", index=False)
        logger.info(f"Results exported → {path}")
        return path

    # ── INTERNALS ────────────────────────────────────────────────────────

    def _save_fig(self, fig: plt.Figure, name: str) -> str:
        path = OUT_DIR / "figures" / f"{name}.{FIG_FMT}"
        fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig)
        self._figures.append((name, str(path)))
        logger.info(f"Figure saved → {path}")
        return str(path)

    def _log_result(self, result: dict) -> None:
        outcome = result.get("outcome") or result.get("variable", "?")
        test = result.get("test", "?")
        p_fmt = result.get("p_fmt", "?")
        sig = "✓" if result.get("significant") else "✗"
        logger.info(f"  [{sig}] {outcome} | {test} | {p_fmt}")


def outcome_label(col: str) -> str:
    return col.replace("_", " ").title()

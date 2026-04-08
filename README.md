# notion_stats

**Peer-review grade statistical analysis from Notion databases.**

Built for clinical research workflows where reproducibility, effect size reporting, and methods transparency are non-negotiable.

---

## Quick Start

```bash
./setup.sh                          # one-command install
# fill in .env with NOTION_API_KEY
python notebooks/template_analysis.py
```

---

## Architecture

```
notion_stats/
├── setup.sh                        ← one-command environment setup
├── requirements.txt
├── .env.template                   ← copy to .env, add Notion token
├── config/
│   └── settings.yaml               ← α, correction method, output format, etc.
├── src/
│   ├── notion_extractor.py         ← Notion API → DataFrame + Parquet snapshot
│   ├── data_cleaner.py             ← Fluent cleaning API + CONSORT audit trail
│   └── stats_pipeline.py           ← Full analysis engine
├── notebooks/
│   └── template_analysis.py        ← End-to-end analysis template
├── data/
│   ├── raw/                        ← Auto-saved Parquet snapshots (timestamped)
│   └── processed/                  ← Cleaned datasets
└── outputs/
    ├── figures/                    ← SVG/PNG at 300 DPI
    ├── tables/                     ← CSV + Excel (APA format)
    └── reports/
```

---

## Statistical Methods

All methods are peer-review compliant by design:

| Concern | Implementation |
|---|---|
| **Normality** | Shapiro-Wilk (n<50), D'Agostino-Pearson (n≥50) — automatic selection |
| **Two-group comparison** | Student/Welch t-test or Mann-Whitney U — auto-selected |
| **Multi-group** | One-way ANOVA or Kruskal-Wallis + post-hoc (Tukey / Dunn) |
| **Categorical** | Pearson χ² or Fisher's exact (auto-selected on expected cell counts) |
| **Correlation** | Pearson or Spearman — auto-selected; with CI and BF₁₀ |
| **Regression** | OLS (linear) and Logit (binary outcome) with OR + CI |
| **Survival** | Kaplan-Meier + log-rank test (2 groups) / multivariate log-rank |
| **Effect sizes** | Cohen's d, rank-biserial r, η²p, Cramér's V, OR — always reported |
| **Multiple testing** | Benjamini-Hochberg FDR by default (configurable) |
| **Reproducibility** | Global seed; timestamped raw snapshots; full audit trail |

---

## Notion Setup

1. Go to [notion.so/my-integrations](https://www.notion.so/my-integrations) → New integration
2. Copy the token to `.env` as `NOTION_API_KEY=secret_...`
3. Share each database with your integration (database → Share → invite)
4. Add the database UUID to `config/settings.yaml → notion.databases`

**Finding a database UUID:** open the database in Notion browser → the UUID is the 32-char hex in the URL before the `?`.

---

## Peer Review Compliance Checklist

Before submitting:

- [ ] Power calculation documented (§2 of template)
- [ ] All exclusion criteria listed in `DataCleaner.exclude()` calls
- [ ] CONSORT flow printed and included in paper
- [ ] Raw Parquet snapshot archived (data/raw/)
- [ ] Normality test results reported in methods
- [ ] Effect size reported for every significant result
- [ ] Multiple testing correction method stated
- [ ] α threshold declared a-priori
- [ ] Seed fixed and stated in statistical methods
- [ ] Analysis script version-controlled (git tag)

---

## Reporting Standards

| Study Design | Checklist |
|---|---|
| Observational cohort | STROBE |
| RCT | CONSORT |
| Diagnostic accuracy | STARD |
| Case series | CARE |
| Systematic review | PRISMA |

---

## Configuration

All parameters in `config/settings.yaml`:

```yaml
statistics:
  alpha: 0.05
  multiple_testing_correction: "fdr_bh"   # fdr_bh | bonferroni | holm
  normality_test: "shapiro"

reporting:
  figure_dpi: 300
  figure_format: "svg"     # svg preferred for journals
  apa_tables: true
  decimal_places: 3
```

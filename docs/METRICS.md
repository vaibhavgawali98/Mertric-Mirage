## Policies (confirmed against the Online Retail II run)

- **Sheets:** the two sheets overlapped by 22,523 identical rows; the second copy
  is removed. 1,044,848 rows remain (1,044,848 + 22,523 = 1,067,371).
- **Zero denominators** return NaN, shown as "undefined", never 0 or infinity.
- **Missing customer IDs:** 235,287 rows (22.5%). They stay in revenue totals and are
  excluded from every customer metric. Both figures are always reported.
- **Non-product stock codes** (POST, DOT, M, D, S, C2, BANK CHARGES, ADJUST, AMAZONFEE,
  B, CRUK, TEST001, gift vouchers, and others) are set aside by a code-pattern
  heuristic and reported, never silently dropped. Postage charges are excluded from
  product revenue by definition. Known edge: the heuristic also sets aside a few
  genuine-looking items (DCGS... products, PADS), roughly 2.5k in total (about
  0.013% of gross sales) among the codes listed in the run output.
- **Zero-price rows:** 5,938 rows with Price <= 0 are excluded from revenue metrics.
- **Exact duplicates:** 11,812 rows repeat an earlier row on all eight raw columns
  (11,001 distinct repeated groups). Adding `Country` to the key changes nothing.
  They are kept and flagged: an anomaly, not a confirmed error.
- **Order-value metrics** use the order-level frame (one row per sale invoice).
- **Currency** is assumed to be GBP (verify on the UCI page).
- **Timestamps** are timezone-naive as provided; periods are calendar months or quarters.
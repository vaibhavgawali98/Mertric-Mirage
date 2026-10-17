# Uploading your own data

Open **Upload Data** in the sidebar. Pick a file (or press *Try the sample file*), confirm what each
column means, check the preview, and press **Use this dataset for analysis**. Every page then works
on your data.

## Files that work

| | |
|---|---|
| Formats | `.csv`, `.tsv`, `.txt` (comma, semicolon, tab or pipe separated), `.xlsx`, `.xlsm` |
| Not supported | old `.xls` (save as `.xlsx` or `.csv`) |
| Limits | 2,000,000 rows, 300 columns, 200 MB per file |
| Encodings | UTF-8 (with or without BOM), Windows-1252, Latin-1, UTF-16 are detected automatically |

A good file has one row per order, session or event: an ID, a date, a customer or user, a value, and
any columns you want to slice by (region, channel, plan, ...). Line-item exports (several rows per
order) work too: tick **combine lines into one row per order ID**.

## What you tell it (all optional)

| Role | Enables |
|---|---|
| Date or timestamp | Period labels (month, quarter or year), period comparisons, matched windows, change points, cohorts |
| Customer or user ID | Cohort retention, repeat purchasing, revenue concentration, value tiers |
| Order or transaction ID | Duplicate-order warnings, combining line items into orders |
| Value (revenue, amount) | Concentration, order-value analyses; renamed `order_value` |

Anything you skip simply keeps the analyses that need it switched off, with a reason. The preview
shows which pages your dataset will support before you commit.

Columns are also typed automatically: numbers with currency symbols, thousands separators,
`(negative)` brackets and decimal commas are understood; identifier-like columns (`customer_id`,
SKU, zip codes with leading zeros) are kept as text so they are never treated as metrics; yes/no
columns can be converted to 0/1 so they can be used as outcomes.

## Dates

* ISO dates (`2021-04-03`), times, time zones, AM/PM, month names and Excel/Unix numeric dates are understood.
* Dates like `03/04/2021` are **never guessed**. If some value in the column has a first part above 12
  the order is inferred; otherwise you are asked to choose *Day first* or *Month first*.
* Dates that cannot be read are counted and reported, and those rows get no period. A column where
  more than half the values are unreadable is rejected as "not a date".

## What changes in your data

The prepared table keeps your columns, with these standard names for mapped roles: `order_date`,
`customer_id`, `order_id`, `order_value`, plus a derived `period` label. If your file already has a
column called `period` or `order_value` that is not the mapped one, it is kept as `period_original`
or `order_value_original`. The page lists every cleaning step (renamed headers, removed empty rows
and columns, values set to missing because they were not numeric). Exact duplicate rows are counted,
not removed.

## Privacy and reproducibility

* The file is read on the machine running the app and kept only in your browser session
  (`st.session_state`). It is never written to disk and never shared between sessions.
* Findings you save are stored in the Evidence Room, but the data is not. A saved finding records a
  dataset id (a hash of the file and the mapping). Upload the same file with the same mapping and the
  id matches, so *Re-run from stored parameters* works again.
* Findings from uploads are labelled with source `upload`. A file with an `is_simulated` column that
  is true in every row is labelled SIMULATED instead.

## Limits to keep in mind

* The SQL panel and the retail-specific metrics (returns, postage, invoices) apply to the built-in
  retail dataset only. Uploads use the Pandas engines.
* Cohorts and concentration assume one row per order; use the line-item option if that is not true.
* Standard errors assume independent rows. Orders from the same customer are not independent, so
  real uncertainty is larger than shown.
* A pattern in your data is a lead, not proof of cause.

## Troubleshooting

| Message | What to do |
|---|---|
| "The file has a header but no data rows" | The export is empty or the wrong sheet was chosen |
| "Could not decode the file" | Open *Reading options* and pick the encoding (try `cp1252`) |
| "Some malformed lines were skipped" | Some rows have a different number of fields; check quoting in the source |
| "Date order needed" | Choose Day first or Month first and press *Check and prepare data* again |
| "does not look like dates" | The mapped date column holds something else; pick another column |
| "is not numeric" | The mapped value column is mostly text; pick a numeric column or none |
| A cohort page says the analysis is off | Map customer ID, date and value (all three are required) |

## For the README

> **Upload your own data.** Metric Mirage accepts CSV and Excel files. A reading step detects
> encoding and delimiter, a cleaning step reports every change, dates are parsed without silent
> guessing (ambiguous formats require a user decision), and a column-mapping step turns the file into a
> dataset the same analysis engines use for the built-in data. The sample file and 58 ingestion tests
> run on pandas 2.2 and 3.x.

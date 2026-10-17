import io

import numpy as np
import pandas as pd
import pytest

from core import ingest as ig
from core.ingest import IngestError, Mapping
from data.sample_upload import SAMPLE_NAME, make_sample_orders, sample_csv_bytes


def clean(df):
    out, info = ig.clean_table(df.astype(object))
    return out, info


def prepare(df, mapping, name="t.csv", source_id="src"):
    cleaned, info = clean(df)
    return ig.prepare_dataset(cleaned, mapping, source_id, name, clean_info=info)


def dates(values, **kw):
    return ig.parse_dates(pd.Series(values, dtype=object), **kw)


def as_days(parsed):
    return [None if pd.isna(v) else str(v)[:10] for v in parsed.values]


# ------------------------------------------------------------------ reading
@pytest.mark.parametrize("delim", [",", ";", "\t", "|"])
def test_delimiter_is_detected(delim):
    text = delim.join(["a", "b", "c"]) + "\n" + "\n".join(delim.join(["1", "2", "3"]) for _ in range(5))
    assert ig.detect_delimiter(text) == delim
    df, info = ig.read_table(text.encode(), "x.csv")
    assert info.delimiter == delim and df.shape == (5, 3)


def test_utf8_bom_and_quoted_commas():
    data = "\ufeffname,note\n\"Smith, J\",ok\nLee,fine\n".encode("utf-8")
    df, info = ig.read_table(data, "x.csv")
    assert list(df.columns) == ["name", "note"] and df.loc[0, "name"] == "Smith, J"
    assert info.encoding == "utf-8-sig"


def test_non_utf8_files_are_decoded_and_flagged():
    df, info = ig.read_table("name,city\nJosé,Zürich\n".encode("cp1252"), "x.csv")
    assert df.loc[0, "name"] == "José" and info.encoding == "cp1252"
    assert any("not UTF-8" in n for n in info.notes)
    df16, info16 = ig.read_table("a,b\n1,2\n3,4\n".encode("utf-16"), "x.csv")
    assert info16.encoding == "utf-16" and df16.shape == (2, 2)
    with pytest.raises(IngestError):
        ig.read_table(b"a,b\n1,2\n", "x.csv", encoding="not-a-codec")


def test_tsv_suffix_and_forced_delimiter():
    df, info = ig.read_table(b"a\tb\n1\t2\n", "x.tsv")
    assert info.delimiter == "\t"
    df, info = ig.read_table(b"a;b\n1;2\n", "x.csv", delimiter=";")
    assert df.shape == (1, 2)


@pytest.mark.parametrize("data,name", [(b"", "a.csv"), (b"   \n ", "a.csv"), (b"a,b\n", "a.csv"),
                                       (b"x", "a.xls"), (b"x", "a.pdf"), (b"x", "a.json")])
def test_bad_files_raise_ingest_error(data, name):
    with pytest.raises(IngestError):
        ig.read_table(data, name)


def test_malformed_lines_are_skipped_and_reported():
    data = b"a,b,c\n1,2,3\n4,5,6,7,8\n9,10,11\n"
    df, info = ig.read_table(data, "x.csv")
    assert len(df) == 2 and any("malformed" in n for n in info.notes)


def test_size_limits(monkeypatch):
    monkeypatch.setattr(ig, "MAX_ROWS", 3)
    with pytest.raises(IngestError, match="rows"):
        ig.read_table(b"a\n1\n2\n3\n4\n", "x.csv")
    monkeypatch.setattr(ig, "MAX_ROWS", 100)
    monkeypatch.setattr(ig, "MAX_COLUMNS", 2)
    with pytest.raises(IngestError, match="columns"):
        ig.read_table(b"a,b,c\n1,2,3\n", "x.csv")


def _workbook():
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        pd.DataFrame({"when": pd.to_datetime(["2021-01-05", "2021-02-07", "2021-03-09"]),
                      "amount": [10.5, 20.0, 30.25], "code": ["007", "008", "009"],
                      "flag": [True, False, True]}).to_excel(w, sheet_name="Data", index=False)
        pd.DataFrame({"z": [1, 2]}).to_excel(w, sheet_name="Other", index=False)
    return buf.getvalue()


def test_excel_sheets_and_cell_fidelity():
    data = _workbook()
    assert ig.list_sheets(data) == ["Data", "Other"]
    df, info = ig.read_table(data, "b.xlsx", sheet="Data")
    assert info.sheet == "Data" and df.shape == (3, 4)
    assert df["code"].tolist() == ["007", "008", "009"]            # leading zeros survive
    other, _ = ig.read_table(data, "b.xlsx", sheet="Other")
    assert list(other.columns) == ["z"]
    with pytest.raises(IngestError, match="not found"):
        ig.read_table(data, "b.xlsx", sheet="Nope")
    with pytest.raises(IngestError):
        ig.list_sheets(b"this is not a workbook")


def test_excel_end_to_end():
    df, _ = ig.read_table(_workbook(), "b.xlsx", sheet="Data")
    p = prepare(df, Mapping(date="when", value="amount"))
    assert p.frame["order_value"].tolist() == [10.5, 20.0, 30.25]
    assert as_days(ig.parse_dates(p.frame["order_date"])) == ["2021-01-05", "2021-02-07", "2021-03-09"]
    assert p.frame["code"].tolist() == ["007", "008", "009"]
    assert "code" in " ".join(p.report.notes)
    assert ig.binary_text_candidates(clean(df)[0]) == ["flag"]


# ----------------------------------------------------------------- cleaning
def test_headers_are_made_unique_and_named():
    raw = pd.DataFrame([[1, 2, 3, 4]], columns=["a", "a", "Unnamed: 2", " spaced   name "]).astype(object)
    out, info = ig.clean_table(raw)
    assert list(out.columns) == ["a", "a_2", "column_3", "spaced name"]
    assert info["renamed"]


def test_text_is_trimmed_and_na_tokens_become_missing():
    raw = pd.DataFrame({"x": ["  a ", "N/A", "-", "", "null", "b"], "y": [1, 2, 3, 4, 5, 6]}).astype(object)
    out, info = ig.clean_table(raw)
    assert out["x"].tolist()[0] == "a" and out["x"].isna().sum() == 4


def test_empty_rows_columns_and_duplicates_are_counted():
    raw = pd.DataFrame({"a": ["1", "1", None, "2"], "b": ["x", "x", None, "y"], "c": [None] * 4}).astype(object)
    out, info = ig.clean_table(raw)
    assert info["dropped_empty_columns"] == ["c"] and info["dropped_empty_rows"] == 1
    assert info["duplicate_rows"] == 1 and len(out) == 3


def test_mixed_type_columns_become_text():
    raw = pd.DataFrame({"m": ["a", 5, 2.5, None]}, dtype=object)
    out, _ = ig.clean_table(raw)
    assert out["m"].tolist()[:3] == ["a", "5", "2.5"]


def test_all_empty_table_raises():
    with pytest.raises(IngestError):
        ig.clean_table(pd.DataFrame({"a": [None, None]}, dtype=object))


# -------------------------------------------------------------------- dates
def test_iso_and_year_first_formats():
    r = dates(["2021-04-03", "2021/4/5", "2021-04-07 14:30:00", "2021-04-09T10:00:00Z", None, "garbage"])
    assert as_days(r) == ["2021-04-03", "2021-04-05", "2021-04-07", "2021-04-09", None, None]
    assert r.n_failed == 1 and r.n_missing == 1 and r.ambiguity == "none"
    assert r.failed_examples == ["garbage"]
    assert str(r.values.iloc[2]) == "2021-04-07 14:30:00"


def test_day_first_is_inferred_from_a_value_above_twelve():
    r = dates(["03/04/2021", "13/04/2021"])
    assert as_days(r) == ["2021-04-03", "2021-04-13"] and r.dayfirst is True and r.ambiguity == "none"


def test_month_first_is_inferred_from_a_second_part_above_twelve():
    r = dates(["03/04/2021", "04/13/2021"])
    assert as_days(r) == ["2021-03-04", "2021-04-13"] and r.dayfirst is False and r.ambiguity == "none"


def test_truly_ambiguous_dates_are_flagged_not_guessed():
    r = dates(["03/04/2021", "05/06/2021"])
    assert r.ambiguity == "ambiguous"
    chosen = dates(["03/04/2021", "05/06/2021"], dayfirst=True)
    assert chosen.ambiguity == "none" and as_days(chosen) == ["2021-04-03", "2021-06-05"]
    other = dates(["03/04/2021", "05/06/2021"], dayfirst=False)
    assert as_days(other) == ["2021-03-04", "2021-05-06"]


def test_equal_day_and_month_is_not_ambiguous():
    assert dates(["05/05/2021", "07/07/2021"]).ambiguity == "none"


def test_mixed_day_and_month_first_is_reported_as_inconsistent():
    assert dates(["13/04/2021", "04/13/2021", "03/04/2021"]).ambiguity == "inconsistent"


def test_times_am_pm_and_month_names():
    r = dates(["3/4/2021 2:30 PM", "5/6/2021 11:05 AM"], dayfirst=False)
    assert [str(v) for v in r.values] == ["2021-03-04 14:30:00", "2021-05-06 11:05:00"]
    names = dates(["3 Apr 2021", "April 5, 2021", "2021-04"])
    assert as_days(names) == ["2021-04-03", "2021-04-05", "2021-04-01"] and names.n_failed == 0


def test_numeric_date_encodings():
    assert as_days(ig.parse_dates(pd.Series([20210403, 20210404, None]))) == ["2021-04-03", "2021-04-04", None]
    assert as_days(ig.parse_dates(pd.Series([44197, 44198]))) == ["2021-01-01", "2021-01-02"]
    assert as_days(ig.parse_dates(pd.Series([1609459200, 1609545600]))) == ["2021-01-01", "2021-01-02"]
    assert as_days(dates(["20210403", "20210404"])) == ["2021-04-03", "2021-04-04"]
    weird = ig.parse_dates(pd.Series([5, 7, 9]))
    assert weird.n_failed == 3


def test_datetime_inputs_pass_through_and_time_zones_are_dropped():
    r = ig.parse_dates(pd.Series(pd.to_datetime(["2021-01-01", None])))
    assert r.n_failed == 0 and r.n_missing == 1
    tz = ig.parse_dates(pd.Series(pd.to_datetime(["2021-01-01 10:00"]).tz_localize("UTC")))
    assert getattr(tz.values.dtype, "tz", None) is None
    mixed = dates([pd.Timestamp("2021-05-01"), "2021-05-02"])
    assert as_days(mixed) == ["2021-05-01", "2021-05-02"]


def test_choose_freq_and_labels():
    d = pd.Series(pd.date_range("2021-01-01", periods=100, freq="D"))
    assert ig.choose_freq(d) == "M"
    assert ig.choose_freq(pd.Series(pd.date_range("2010-01-01", periods=100, freq="MS"))) == "Q"
    assert ig.choose_freq(pd.Series(pd.date_range("1900-01-01", periods=300, freq="QS"))) == "Y"
    assert ig.choose_freq(pd.Series(pd.to_datetime(["2021-03-01", "2021-03-30"]))) is None
    labels = ig.period_labels(pd.Series(pd.to_datetime(["2021-03-05", None])), "Q")
    assert labels.iloc[0] == "2021Q1" and pd.isna(labels.iloc[1])
    assert ig.period_labels(pd.Series(pd.to_datetime(["2021-03-05"])), "M").iloc[0] == "2021-03"


# ------------------------------------------------------------- type inference
def test_numbers_with_currency_brackets_and_thousands():
    s = pd.Series(["$1,234.50", "(12.5)", "1 000", "abc", None, "7", "1,2"], dtype=object)
    got = ig._to_number(s).tolist()
    assert got[:3] == [1234.5, -12.5, 1000.0] and np.isnan(got[3]) and np.isnan(got[4]) and got[5] == 7.0
    assert np.isnan(got[6])                       # '1,2' is ambiguous with a dot decimal, so it is not guessed
    assert ig._to_number(pd.Series(["1.234,56", "12,5", "x"], dtype=object), ",").tolist()[:2] == [1234.56, 12.5]


def test_column_decimal_is_per_column():
    comma = pd.Series(["12,5", "1.234,56", "3,00", "4,25"], dtype=object)
    dot = pd.Series(["1,234.50", "12.5", "3.00"], dtype=object)
    assert ig.column_decimal(comma) == "," and ig.column_decimal(dot) == "."
    assert ig.column_decimal(dot, forced=True) == "," and ig.column_decimal(pd.Series([1.5, 2.5])) == "."


def test_infer_types_reports_what_it_changed():
    raw = pd.DataFrame({
        "sales": [str(i) for i in range(1, 40)] + ["x"],      # 97.5% numeric -> converted, 'x' counted
        "mostly": ["1", "2", "a", "b", "c"] * 8,               # 40% numeric -> stays text
        "zip": ["00123", "04567", "01234", "09999"] * 10,      # leading zeros -> stays text
        "customer_id": [str(i) for i in range(40)],            # id-like name, all unique -> text
        "label": ["a", "b"] * 20,
    }).astype(object)
    out, rep = ig.infer_types(raw)
    assert pd.api.types.is_numeric_dtype(out["sales"]) and rep.coerced_to_missing == {"sales": 1}
    assert out["mostly"].dtype == object and "mostly" not in rep.numeric
    assert out["zip"].tolist()[0] == "00123" and rep.leading_zero_text == ["zip"]
    assert rep.id_as_text == ["customer_id"] and out["customer_id"].dtype == object
    assert out["label"].dtype == object


def test_binary_text_candidates_and_conversion():
    raw = pd.DataFrame({"paid": ["Yes", "no", "YES", "No"], "city": ["a", "b", "c", "d"]}).astype(object)
    assert ig.binary_text_candidates(raw) == ["paid"]
    out, rep = ig.infer_types(raw, binary_text=("paid",))
    assert out["paid"].tolist() == [1, 0, 1, 0] and rep.binary_converted == ["paid"]
    with pytest.raises(IngestError, match="yes/no"):
        ig.infer_types(raw, binary_text=("city",))


def test_bool_columns_become_integers():
    out, rep = ig.infer_types(pd.DataFrame({"b": [True, False, True]}))
    assert out["b"].tolist() == [1, 0, 1] and rep.bool_converted == ["b"]


# ------------------------------------------------------------ prepare_dataset
def _orders(n=60):
    return pd.DataFrame({
        "Invoice": [f"A{i}" for i in range(n)],
        "Order Date": [f"2021-{(i % 6) + 1:02d}-{(i % 27) + 1:02d}" for i in range(n)],
        "Cust": [f"{1000 + i % 9}.0" for i in range(n)],
        "Amount": [f"{10 + i}.50" for i in range(n)],
        "Region": ["N", "S"] * (n // 2),
    })


def test_standard_names_period_and_column_order():
    p = prepare(_orders(), Mapping(date="Order Date", customer="Cust", order_id="Invoice", value="Amount"))
    f = p.frame
    assert list(f.columns)[:5] == ["order_id", "order_date", "period", "customer_id", "order_value"]
    assert pd.api.types.is_datetime64_any_dtype(f["order_date"]) and pd.api.types.is_numeric_dtype(f["order_value"])
    assert f["customer_id"].iloc[0] == "1000"                    # '1000.0' -> '1000'
    assert p.freq == "M" and f["period"].nunique() == 6 and p.report.date["n_periods"] == 6
    assert p.source == "upload" and p.key.startswith(ig.UPLOAD_PREFIX) and p.report.rows_out == 60


def test_no_roles_still_works_and_creates_no_period():
    p = prepare(pd.DataFrame({"a": ["x", "y"] * 10, "b": list(range(20))}), Mapping())
    assert "period" not in p.frame.columns and p.freq is None and p.frame["b"].dtype != object


def test_period_grain_choices():
    base = _orders()
    assert prepare(base, Mapping(date="Order Date", period="Q")).frame["period"].iloc[0].startswith("2021Q")
    assert prepare(base, Mapping(date="Order Date", period="Y")).frame["period"].unique().tolist() == ["2021"]
    assert "period" not in prepare(base, Mapping(date="Order Date", period="none")).frame.columns
    one_month = pd.DataFrame({"d": ["2021-03-01", "2021-03-09"] * 5, "v": ["1"] * 10})
    p = prepare(one_month, Mapping(date="d"))
    assert "period" not in p.frame.columns and any("one month" in w for w in p.report.warnings)


def test_reserved_name_clash_keeps_the_original_column():
    df = pd.DataFrame({"when": ["2021-01-05", "2021-02-06"] * 5, "period": ["a", "b"] * 5,
                       "order_value": ["1", "2"] * 5, "money": ["5", "6"] * 5})
    p = prepare(df, Mapping(date="when", value="money"))
    assert "period_original" in p.frame.columns and "order_value_original" in p.frame.columns
    assert p.frame["period"].str.match(r"^\d{4}-\d{2}$").all()
    assert p.frame["order_value"].tolist()[:2] == [5.0, 6.0]


def test_ambiguous_dates_require_a_decision():
    df = pd.DataFrame({"d": ["03/04/2021", "05/06/2021"] * 5, "v": ["1"] * 10})
    undecided = prepare(df, Mapping(date="d"))
    assert undecided.report.date_ambiguous
    decided = prepare(df, Mapping(date="d", dayfirst=True))
    assert not decided.report.date_ambiguous and decided.frame["order_date"].min() == pd.Timestamp("2021-04-03")
    assert undecided.key != decided.key


def test_failed_dates_are_counted_and_mostly_failed_columns_are_rejected():
    ok = pd.DataFrame({"d": ["2021-01-05"] * 9 + ["garbage"], "v": ["1"] * 10})
    p = prepare(ok, Mapping(date="d"))
    assert p.report.date["n_failed"] == 1 and any("could not be read" in w for w in p.report.warnings)
    bad = pd.DataFrame({"d": ["apple", "pear", "plum"] * 4, "v": ["1"] * 12})
    with pytest.raises(IngestError, match="does not look like dates"):
        prepare(bad, Mapping(date="d"))


def test_value_column_must_be_numeric_and_failures_are_counted():
    df = _orders()
    df.loc[0, "Amount"] = "oops"
    p = prepare(df, Mapping(value="Amount"))
    assert p.report.coerced_to_missing["Amount"] == 1 and p.frame["order_value"].isna().sum() == 1
    with pytest.raises(IngestError, match="not numeric"):
        prepare(_orders(), Mapping(value="Region"))


def test_negative_values_are_noted_not_dropped():
    df = pd.DataFrame({"v": ["10", "-5", "20", "(3)"]})
    p = prepare(df, Mapping(value="v"))
    assert p.report.value["n_negative"] == 2 and p.frame["order_value"].tolist() == [10, -5, 20, -3]


def test_mapping_validation():
    cleaned, _ = clean(_orders())
    for bad in [Mapping(date="nope"), Mapping(date="Invoice", order_id="Invoice"),
                Mapping(value="Amount", drop=("Amount",)), Mapping(aggregate_lines=True),
                Mapping(period="weekly")]:
        with pytest.raises(IngestError):
            ig.prepare_dataset(cleaned, bad, "s", "t")
    with pytest.raises(IngestError):
        ig.prepare_dataset(pd.DataFrame(), Mapping(), "s", "t")


def test_drop_columns():
    p = prepare(_orders(), Mapping(drop=("Region",)))
    assert "Region" not in p.frame.columns


def test_duplicate_order_ids_warn_when_not_aggregated():
    df = _orders()
    df.loc[1, "Invoice"] = df.loc[0, "Invoice"]
    p = prepare(df, Mapping(order_id="Invoice"))
    assert any("share an order ID" in w for w in p.report.warnings)


def test_line_items_are_combined_into_orders():
    rows = []
    for i in range(1, 41):
        for j in range(3):
            rows.append({"inv": f"INV{i}", "dt": f"2021-{(i % 6) + 1:02d}-{(i % 27) + 1:02d}", "cust": f"{1000 + i % 9}.0",
                         "qty": j + 1, "amount": 10.0 * (j + 1), "country": "UK" if i % 3 else "FR", "sku": f"S{j}"})
    p = prepare(pd.DataFrame(rows).astype(str),
                Mapping(date="dt", customer="cust", order_id="inv", value="amount", aggregate_lines=True))
    f = p.frame
    assert len(f) == 40 and f["order_id"].is_unique and (f["n_lines"] == 3).all()
    assert (f["order_value"] == 60.0).all()                        # 10 + 20 + 30 summed
    assert "country" in f.columns and "qty" not in f.columns and "sku" not in f.columns
    assert p.report.aggregation["lines"] == 120 and p.report.aggregation["orders"] == 40
    assert p.report.aggregation["dropped_line_level_columns"] == ["qty", "sku"]


def test_aggregation_reports_customer_conflicts_and_missing_ids():
    df = pd.DataFrame({"inv": ["A", "A", "B", None], "cust": ["1", "2", "3", "4"], "v": ["1", "2", "3", "4"]})
    p = prepare(df, Mapping(customer="cust", order_id="inv", value="v", aggregate_lines=True))
    assert len(p.frame) == 2 and p.frame.loc[p.frame["order_id"] == "A", "order_value"].iloc[0] == 3
    assert any("more than one customer" in w for w in p.report.warnings)
    assert any("no order ID" in w for w in p.report.warnings)


def test_simulated_files_are_labelled_synthetic():
    df = pd.DataFrame({"v": ["1", "2"] * 5, "is_simulated": ["True"] * 10})
    assert prepare(df, Mapping(value="v")).source == "synthetic"
    df2 = pd.DataFrame({"v": ["1", "2"] * 5, "is_simulated": ["True"] * 9 + ["False"]})
    assert prepare(df2, Mapping(value="v")).source == "upload"


def test_dataset_key_is_stable_and_depends_on_content_and_mapping():
    cleaned, info = clean(_orders())
    a = ig.prepare_dataset(cleaned, Mapping(value="Amount"), "digest1", "f.csv", clean_info=info).key
    assert a == ig.prepare_dataset(cleaned, Mapping(value="Amount"), "digest1", "f.csv", clean_info=info).key
    assert a != ig.prepare_dataset(cleaned, Mapping(value="Amount"), "digest2", "f.csv", clean_info=info).key
    assert a != ig.prepare_dataset(cleaned, Mapping(value="Amount", date="Order Date"), "digest1", "f.csv", clean_info=info).key
    assert ig.file_digest(b"abc") == ig.file_digest(b"abc") != ig.file_digest(b"abd")


def test_profile_columns():
    p = prepare(_orders(), Mapping(date="Order Date", value="Amount"))
    prof = ig.profile_columns(p.frame).set_index("Column")
    assert prof.loc["order_date", "Type"] == "date" and prof.loc["order_value", "Type"] == "numeric"
    assert prof.loc["Region", "Type"] == "text" and prof.loc["Region", "Distinct"] == 2
    assert ig.profile_columns(pd.DataFrame({"f": [0, 1, 1]})).loc[0, "Type"] == "numeric (0/1)"


# ----------------------------------------------------------- suggest_mapping
def test_suggestions_for_a_typical_export():
    df = pd.DataFrame({
        "Invoice No": [f"I{i}" for i in range(40)],
        "Purchase Date": [f"2021-{(i % 6) + 1:02d}-{(i % 27) + 1:02d}" for i in range(40)],
        "Client": [f"c{i % 7}" for i in range(40)],
        "Total Sales": [str(10 + i) for i in range(40)],
        "Region": ["a", "b"] * 20,
    })
    cleaned, _ = clean(df)
    m, why = ig.suggest_mapping(cleaned)
    assert (m.date, m.customer, m.order_id, m.value) == ("Purchase Date", "Client", "Invoice No", "Total Sales")
    assert set(why) == {"date", "customer", "order_id", "value"}


def test_suggestions_do_not_pick_a_numeric_id_as_a_date_or_value():
    df = pd.DataFrame({"user id": [str(1000 + i) for i in range(40)], "score": [str(i) for i in range(40)]})
    cleaned, _ = clean(df)
    m, _ = ig.suggest_mapping(cleaned)
    assert m.date is None and m.value is None and m.customer == "user id"


# ------------------------------------------------------------------- sample file
def test_sample_file_is_reproducible_and_simulated():
    a, b = make_sample_orders(), make_sample_orders()
    pd.testing.assert_frame_equal(a, b)
    assert sample_csv_bytes() == sample_csv_bytes() and SAMPLE_NAME.endswith("SIMULATED.csv")
    assert a["is_simulated"].all() and a["order_id"].is_unique and len(a) > 1500
    assert a["order_date"].min() >= "2023-01-01" and a["order_date"].max() <= "2024-06-30"
    assert make_sample_orders(seed=1)["revenue"].sum() != a["revenue"].sum()


def test_sample_file_goes_through_the_whole_pipeline():
    df, info = ig.read_table(sample_csv_bytes(), SAMPLE_NAME)
    cleaned, cinfo = ig.clean_table(df)
    mapping, _ = ig.suggest_mapping(cleaned)
    assert (mapping.date, mapping.customer, mapping.order_id, mapping.value) == (
        "order_date", "customer_id", "order_id", "revenue")
    p = ig.prepare_dataset(cleaned, mapping, ig.file_digest(sample_csv_bytes()), SAMPLE_NAME, info.notes, cinfo)
    f = p.frame
    assert p.source == "synthetic" and p.freq == "M" and f["period"].nunique() == 18
    assert f["order_id"].is_unique and not p.report.warnings
    assert {"region", "channel"} <= set(f.columns)
    assert f["order_value"].notna().all() and (f["order_value"] > 0).all()


def test_suggestions_understand_decimal_comma_value_columns():
    df = pd.DataFrame({"Order No": ["A1", "A2", "A3", "A4"], "Revenue": ["1.234,50", "250,00", "99,90", "310,25"],
                       "Region": ["N", "S", "N", "S"]})
    cleaned, _ = clean(df)
    assert ig.suggest_mapping(cleaned)[0].value == "Revenue"

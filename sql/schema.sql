-- Grain: transactions has one row per INVOICE LINE (a product on an invoice).
-- The orders view has one row per SALE INVOICE. Returns, postage and zero-price
-- rows stay in transactions, classified by row_class, and never enter orders.
DROP VIEW IF EXISTS orders;
DROP TABLE IF EXISTS transactions;
DROP TABLE IF EXISTS meta;

CREATE TABLE meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE transactions (
    line_id      INTEGER PRIMARY KEY,
    invoice      TEXT NOT NULL,
    stock_code   TEXT,
    description  TEXT,
    quantity     REAL,
    invoice_date TEXT NOT NULL,   -- ISO text 'YYYY-MM-DD HH:MM:SS' (sorts chronologically)
    unit_price   REAL,
    customer_id  INTEGER,         -- NULL when the source has no customer ID
    country      TEXT,
    row_class    TEXT NOT NULL CHECK (row_class IN
                 ('sale', 'return', 'non_product', 'invalid_price', 'inconsistent')),
    line_revenue REAL
);

CREATE VIEW orders AS
SELECT invoice,
       MIN(invoice_date) AS order_date,   -- earliest line timestamp
       MIN(customer_id)  AS customer_id,
       MIN(country)      AS country,
       SUM(line_revenue) AS order_value,
       COUNT(*)          AS n_lines
FROM transactions
WHERE row_class = 'sale'
GROUP BY invoice;
CREATE INDEX IF NOT EXISTS idx_tx_invoice  ON transactions (invoice);
CREATE INDEX IF NOT EXISTS idx_tx_customer ON transactions (customer_id);
CREATE INDEX IF NOT EXISTS idx_tx_class    ON transactions (row_class, invoice_date);
ANALYZE;
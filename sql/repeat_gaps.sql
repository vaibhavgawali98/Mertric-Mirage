-- Days between consecutive purchase DAYS per customer (several invoices on one day
-- count as one purchase occasion). LAG looks at the customer's previous purchase day.
WITH purchase_days AS (
    SELECT DISTINCT customer_id, date(order_date) AS day
    FROM orders
    WHERE customer_id IS NOT NULL
),
gaps AS (
    SELECT customer_id,
           julianday(day) - julianday(LAG(day) OVER (PARTITION BY customer_id ORDER BY day)) AS gap_days
    FROM purchase_days
),
per_customer AS (
    SELECT customer_id, COUNT(*) AS n_days
    FROM purchase_days
    GROUP BY customer_id
)
SELECT (SELECT COUNT(*) FROM per_customer)                    AS n_customers,
       (SELECT COUNT(*) FROM per_customer WHERE n_days >= 2)  AS n_repeat_customers,
       COUNT(gap_days)                                        AS n_gaps,
       AVG(gap_days)                                          AS mean_days
FROM gaps;
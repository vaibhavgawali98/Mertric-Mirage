-- Customers active k months after their first-order month.
-- Mirrors core.cohorts.cohort_retention: orders without a customer ID are excluded,
-- and the final month is dropped when the data stops before that month ends.
WITH bounds AS (
    SELECT strftime('%Y-%m', MAX(order_date)) AS last_month,
           CASE WHEN date(MAX(order_date)) < date(MAX(order_date), 'start of month', '+1 month', '-1 day')
                THEN 1 ELSE 0 END AS last_is_incomplete
    FROM orders
    WHERE customer_id IS NOT NULL
),
customer_months AS (
    SELECT DISTINCT o.customer_id, strftime('%Y-%m', o.order_date) AS ym
    FROM orders AS o
    CROSS JOIN bounds AS b
    WHERE o.customer_id IS NOT NULL
      AND NOT (b.last_is_incomplete = 1 AND strftime('%Y-%m', o.order_date) = b.last_month)
),
first_month AS (
    SELECT customer_id, MIN(ym) AS cohort
    FROM customer_months
    GROUP BY customer_id
)
SELECT f.cohort AS cohort,
       (CAST(substr(c.ym, 1, 4) AS INTEGER) * 12 + CAST(substr(c.ym, 6, 2) AS INTEGER))
     - (CAST(substr(f.cohort, 1, 4) AS INTEGER) * 12 + CAST(substr(f.cohort, 6, 2) AS INTEGER))
           AS months_since_first,
       COUNT(DISTINCT c.customer_id) AS active_customers
FROM customer_months AS c
JOIN first_month AS f ON f.customer_id = c.customer_id
GROUP BY 1, 2
ORDER BY 1, 2;
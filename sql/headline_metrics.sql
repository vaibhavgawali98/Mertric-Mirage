-- Headline metrics, defined exactly as in docs/METRICS.md.
-- A zero denominator yields NULL (shown as undefined), never 0 or infinity.
WITH sale AS (
    SELECT invoice, customer_id, line_revenue
    FROM transactions
    WHERE row_class = 'sale'
),
cust AS (
    SELECT customer_id, COUNT(DISTINCT invoice) AS n_orders
    FROM sale
    WHERE customer_id IS NOT NULL
    GROUP BY customer_id
),
tot AS (
    SELECT SUM(line_revenue) AS gross_sales,
           COUNT(DISTINCT invoice) AS orders,
           SUM(CASE WHEN customer_id IS NOT NULL THEN line_revenue ELSE 0 END) AS revenue_with_id
    FROM sale
),
ret AS (
    SELECT -SUM(line_revenue) AS returns_value
    FROM transactions
    WHERE row_class = 'return'
),
cs AS (
    SELECT COUNT(*) AS active_customers,
           SUM(CASE WHEN n_orders >= 2 THEN 1 ELSE 0 END) AS repeat_customers
    FROM cust
)
SELECT COALESCE(tot.gross_sales, 0)                                   AS gross_sales,
       COALESCE(ret.returns_value, 0)                                 AS returns_value,
       COALESCE(tot.gross_sales, 0) - COALESCE(ret.returns_value, 0)  AS net_revenue,
       tot.orders                                                     AS orders,
       cs.active_customers                                            AS active_customers,
       tot.gross_sales * 1.0 / NULLIF(tot.orders, 0)                  AS average_order_value,
       tot.revenue_with_id * 1.0 / NULLIF(cs.active_customers, 0)     AS revenue_per_customer,
       ret.returns_value * 1.0 / NULLIF(tot.gross_sales, 0)           AS return_rate_value,
       cs.repeat_customers * 1.0 / NULLIF(cs.active_customers, 0)     AS repeat_rate
FROM tot, ret, cs;
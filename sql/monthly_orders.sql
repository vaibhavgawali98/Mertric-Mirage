-- Orders, sales and average order value per month, with a month-over-month change
-- (LAG) and a running total (SUM OVER). Grain: one row per month.
WITH monthly AS (
    SELECT strftime('%Y-%m', order_date)       AS period,
           COUNT(*)                            AS orders,
           SUM(order_value)                    AS gross_sales,
           SUM(order_value) * 1.0 / COUNT(*)   AS average_order_value
    FROM orders
    GROUP BY 1
)
SELECT period,
       orders,
       gross_sales,
       average_order_value,
       average_order_value - LAG(average_order_value) OVER (ORDER BY period) AS aov_change_vs_prior_month,
       SUM(gross_sales) OVER (ORDER BY period ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cumulative_sales
FROM monthly
ORDER BY period;
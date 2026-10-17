-- Revenue concentration across identified customers (no customer ID means excluded).
-- :frac is the share of customers counted as "top", for example 0.10.
-- The Gini coefficient uses the running total of revenue in ascending order.
WITH cust AS (
    SELECT customer_id, SUM(order_value) AS rev
    FROM orders
    WHERE customer_id IS NOT NULL
    GROUP BY customer_id
),
sized AS (
    SELECT customer_id,
           rev,
           COUNT(*) OVER ()                                               AS n,
           SUM(rev) OVER ()                                               AS total,
           ROW_NUMBER() OVER (ORDER BY rev DESC, customer_id)             AS rank_desc,
           SUM(rev) OVER (ORDER BY rev ASC, customer_id
                          ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cum_asc
    FROM cust
),
raw_cut AS (
    SELECT ROUND(n * :frac, 9) AS x FROM sized LIMIT 1
),
cut AS (
    SELECT MAX(1, CAST(x AS INTEGER) + (x > CAST(x AS INTEGER))) AS k FROM raw_cut
)
SELECT MAX(n)     AS n_customers,
       MAX(total) AS total_revenue,
       SUM(CASE WHEN rank_desc <= (SELECT k FROM cut) THEN rev ELSE 0 END) * 1.0 / MAX(total) AS top_share,
       (MAX(n) + 1 - 2.0 * SUM(cum_asc) / MAX(total)) / MAX(n) AS gini
FROM sized;
-- Average order value per segment (country) and period, for the contradiction lab.
-- :freq is M, Q or Y. :baseline and :comparison are period labels such as 2010Q4.
WITH labelled AS (
    SELECT COALESCE(country, '(missing)') AS segment,
           CASE :freq
               WHEN 'M' THEN strftime('%Y-%m', order_date)
               WHEN 'Q' THEN strftime('%Y', order_date) || 'Q'
                             || ((CAST(strftime('%m', order_date) AS INTEGER) - 1) / 3 + 1)
               ELSE strftime('%Y', order_date)
           END AS period,
           order_value
    FROM orders
)
SELECT segment,
       period,
       COUNT(*)                          AS n,
       SUM(order_value)                  AS total_value,
       SUM(order_value) * 1.0 / COUNT(*) AS average_order_value
FROM labelled
WHERE period IN (:baseline, :comparison)
GROUP BY segment, period
ORDER BY segment, period;
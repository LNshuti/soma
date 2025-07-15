

WITH daily_demand AS (
    SELECT
        book_id,
        sale_date,
        SUM(quantity) AS daily_quantity,
        SUM(total_amount) AS daily_revenue,
        COUNT(*) AS daily_transactions
    FROM "soma"."main"."fact_sales"
    GROUP BY book_id, sale_date
),

time_series_features AS (
    SELECT
        book_id,
        sale_date,
        daily_quantity,
        daily_revenue,
        daily_transactions,
        
        -- Date features
        EXTRACT(YEAR FROM sale_date) AS year,
        EXTRACT(MONTH FROM sale_date) AS month,
        EXTRACT(DAY FROM sale_date) AS day,
        EXTRACT(DOW FROM sale_date) AS day_of_week,
        EXTRACT(WEEK FROM sale_date) AS week_of_year,
        EXTRACT(QUARTER FROM sale_date) AS quarter,
        
        -- Lag features
        LAG(daily_quantity, 1) OVER (PARTITION BY book_id ORDER BY sale_date) AS quantity_lag_1,
        LAG(daily_quantity, 7) OVER (PARTITION BY book_id ORDER BY sale_date) AS quantity_lag_7,
        LAG(daily_quantity, 30) OVER (PARTITION BY book_id ORDER BY sale_date) AS quantity_lag_30,
        
        -- Rolling averages
        AVG(daily_quantity) OVER (
            PARTITION BY book_id 
            ORDER BY sale_date 
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) AS quantity_ma_7,
        
        AVG(daily_quantity) OVER (
            PARTITION BY book_id 
            ORDER BY sale_date 
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ) AS quantity_ma_30,
        
        -- Rolling standard deviation
        STDDEV(daily_quantity) OVER (
            PARTITION BY book_id 
            ORDER BY sale_date 
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) AS quantity_std_7,
        
        -- Growth rates
        CASE WHEN LAG(daily_quantity, 1) OVER (PARTITION BY book_id ORDER BY sale_date) > 0
             THEN (daily_quantity - LAG(daily_quantity, 1) OVER (PARTITION BY book_id ORDER BY sale_date)) 
                  / LAG(daily_quantity, 1) OVER (PARTITION BY book_id ORDER BY sale_date) * 100
             ELSE NULL END AS quantity_growth_rate_1d,
             
        CASE WHEN LAG(daily_quantity, 7) OVER (PARTITION BY book_id ORDER BY sale_date) > 0
             THEN (daily_quantity - LAG(daily_quantity, 7) OVER (PARTITION BY book_id ORDER BY sale_date)) 
                  / LAG(daily_quantity, 7) OVER (PARTITION BY book_id ORDER BY sale_date) * 100
             ELSE NULL END AS quantity_growth_rate_7d
             
    FROM daily_demand
),

book_context AS (
    SELECT * FROM "soma"."main"."dim_books"
)

SELECT 
    tsf.*,
    bc.genre,
    bc.format,
    bc.price_category,
    bc.publication_year,
    CURRENT_TIMESTAMP AS feature_created_at
FROM time_series_features tsf
LEFT JOIN book_context bc ON tsf.book_id = bc.book_id

  
    
    

    create  table
      "soma"."main"."sales_performance_daily__dbt_tmp"
  
    as (
      

WITH daily_sales AS (
    SELECT
        sale_date,
        genre,
        channel,
        region,
        COUNT(*) AS transaction_count,
        SUM(quantity) AS total_quantity,
        SUM(total_amount) AS total_revenue,
        AVG(total_amount) AS avg_transaction_value,
        AVG(unit_price) AS avg_unit_price,
        AVG(discount_percent) AS avg_discount_percent
    FROM "soma"."main"."fact_sales"
    GROUP BY 1, 2, 3, 4
),

with_trends AS (
    SELECT
        *,
        LAG(total_revenue) OVER (
            PARTITION BY genre, channel, region 
            ORDER BY sale_date
        ) AS prev_day_revenue,
        AVG(total_revenue) OVER (
            PARTITION BY genre, channel, region 
            ORDER BY sale_date 
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) AS rolling_7day_avg_revenue
    FROM daily_sales
)

SELECT
    *,
    CASE 
        WHEN prev_day_revenue IS NOT NULL AND prev_day_revenue > 0
        THEN (total_revenue - prev_day_revenue) / prev_day_revenue * 100 
        ELSE NULL 
    END AS day_over_day_growth_pct,
    CASE 
        WHEN rolling_7day_avg_revenue IS NOT NULL AND rolling_7day_avg_revenue > 0
        THEN (total_revenue - rolling_7day_avg_revenue) / rolling_7day_avg_revenue * 100 
        ELSE NULL 
    END AS vs_7day_avg_pct
FROM with_trends
    );
  
  
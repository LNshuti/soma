
  
    
    

    create  table
      "soma"."main"."book_features__dbt_tmp"
  
    as (
      

WITH book_sales_metrics AS (
    SELECT
        book_id,
        COUNT(*) AS total_transactions,
        SUM(quantity) AS total_quantity_sold,
        SUM(total_amount) AS total_revenue,
        AVG(total_amount) AS avg_transaction_value,
        AVG(unit_price) AS avg_selling_price,
        AVG(discount_percent) AS avg_discount,
        COUNT(DISTINCT channel) AS num_channels,
        COUNT(DISTINCT region) AS num_regions,
        MIN(sale_date) AS first_sale_date,
        MAX(sale_date) AS last_sale_date
    FROM "soma"."main"."fact_sales"
    GROUP BY book_id
),

recent_performance AS (
    SELECT
        book_id,
        COUNT(*) AS transactions_last_30d,
        SUM(quantity) AS quantity_sold_last_30d,
        SUM(total_amount) AS revenue_last_30d
    FROM "soma"."main"."fact_sales"
    WHERE sale_date >= CURRENT_DATE - INTERVAL '30 days'
    GROUP BY book_id
),

books AS (
    SELECT * FROM "soma"."main"."dim_books"
),

publishers AS (
    SELECT * FROM "soma"."main"."dim_publishers"
),

ml_features AS (
    SELECT
        b.book_id,
        b.isbn,
        b.title,
        
        -- Book attributes
        b.genre,
        b.publication_year,
        b.page_count,
        b.price,
        b.format,
        b.language,
        b.length_category,
        b.price_category,
        b.recency_category,
        
        -- Publisher attributes
        b.publisher_id,
        p.publisher_type,
        p.publisher_era,
        p.publisher_size,
        
        -- Sales performance features
        COALESCE(sm.total_transactions, 0) AS total_transactions,
        COALESCE(sm.total_quantity_sold, 0) AS total_quantity_sold,
        COALESCE(sm.total_revenue, 0) AS total_revenue,
        COALESCE(sm.avg_transaction_value, 0) AS avg_transaction_value,
        COALESCE(sm.avg_selling_price, b.price) AS avg_selling_price,
        COALESCE(sm.avg_discount, 0) AS avg_discount,
        COALESCE(sm.num_channels, 0) AS num_channels,
        COALESCE(sm.num_regions, 0) AS num_regions,
        
        -- Recent performance
        COALESCE(rp.transactions_last_30d, 0) AS transactions_last_30d,
        COALESCE(rp.quantity_sold_last_30d, 0) AS quantity_sold_last_30d,
        COALESCE(rp.revenue_last_30d, 0) AS revenue_last_30d,
        
        -- Derived features
        CASE WHEN sm.total_transactions > 0 
             THEN sm.total_quantity_sold::FLOAT / sm.total_transactions 
             ELSE 0 END AS avg_quantity_per_transaction,
             
        CASE WHEN sm.first_sale_date IS NOT NULL 
             THEN (CURRENT_DATE - sm.first_sale_date)
             ELSE NULL END AS days_since_first_sale,
             
        CASE WHEN sm.last_sale_date IS NOT NULL 
             THEN (CURRENT_DATE - sm.last_sale_date)
             ELSE NULL END AS days_since_last_sale,
             
        -- Velocity score
        CASE WHEN COALESCE(sm.total_transactions, 0) > 0
             THEN (COALESCE(rp.transactions_last_30d, 0) * 12.0) / sm.total_transactions
             ELSE 0 END AS velocity_score,
             
        -- Price efficiency
        CASE WHEN b.page_count > 0 
             THEN b.price / b.page_count 
             ELSE 0 END AS price_per_page,
             
        -- Book age in months
        (EXTRACT(YEAR FROM CURRENT_DATE) - b.publication_year) * 12 
        + EXTRACT(MONTH FROM CURRENT_DATE) AS book_age_months,
        
        CURRENT_TIMESTAMP AS feature_created_at
        
    FROM books b
    LEFT JOIN publishers p ON b.publisher_id = p.publisher_id
    LEFT JOIN book_sales_metrics sm ON b.book_id = sm.book_id
    LEFT JOIN recent_performance rp ON b.book_id = rp.book_id
)

SELECT * FROM ml_features
    );
  
  


WITH inventory AS (
    SELECT * FROM "soma"."main"."stg_inventory"
),

sales_velocity AS (
    SELECT
        book_id,
        COUNT(*) AS transaction_count_30d,
        SUM(quantity) AS units_sold_30d,
        AVG(quantity) AS avg_quantity_per_transaction
    FROM "soma"."main"."fact_sales"
    WHERE sale_date >= CURRENT_DATE - INTERVAL '30 days'
    GROUP BY book_id
),

books AS (
    SELECT * FROM "soma"."main"."dim_books"
),

inventory_analysis AS (
    SELECT
        i.inventory_id,
        i.book_id,
        b.title,
        b.genre,
        b.publisher_name,
        i.warehouse_location,
        i.stock_quantity,
        i.reorder_point,
        i.needs_restock,
        i.storage_cost_per_unit,
        COALESCE(sv.units_sold_30d, 0) AS units_sold_30d,
        COALESCE(sv.transaction_count_30d, 0) AS transaction_count_30d,
        CASE 
            WHEN sv.units_sold_30d > 0 
            THEN i.stock_quantity::FLOAT / (sv.units_sold_30d / 30.0)
            ELSE NULL 
        END AS days_of_inventory,
        i.stock_quantity * i.storage_cost_per_unit AS total_storage_cost,
        CASE 
            WHEN sv.units_sold_30d = 0 THEN 'Dead Stock'
            WHEN i.stock_quantity::FLOAT / NULLIF(sv.units_sold_30d / 30.0, 0) < 7 THEN 'Fast Moving'
            WHEN i.stock_quantity::FLOAT / NULLIF(sv.units_sold_30d / 30.0, 0) < 30 THEN 'Normal'
            WHEN i.stock_quantity::FLOAT / NULLIF(sv.units_sold_30d / 30.0, 0) < 90 THEN 'Slow Moving'
            ELSE 'Very Slow'
        END AS velocity_category
    FROM inventory i
    LEFT JOIN sales_velocity sv ON i.book_id = sv.book_id
    LEFT JOIN books b ON i.book_id = b.book_id
)

SELECT * FROM inventory_analysis
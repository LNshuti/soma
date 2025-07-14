{{ config(
    materialized='incremental',
    unique_key='transaction_id',
    on_schema_change='fail'
) }}

WITH sales AS (
    SELECT * FROM {{ ref('stg_sales') }}
),

books AS (
    SELECT * FROM {{ ref('dim_books') }}
),

final AS (
    SELECT
        s.transaction_id,
        s.book_id,
        s.sale_date,
        s.quantity,
        s.unit_price,
        s.discount_percent,
        s.total_amount,
        s.channel,
        s.customer_type,
        s.region,
        s.sale_year,
        s.sale_month,
        s.day_of_week,
        b.publisher_id,
        b.genre,
        b.format,
        b.price_category,
        b.length_category,
        s.created_at,
        s.dbt_updated_at
    FROM sales s
    LEFT JOIN books b ON s.book_id = b.book_id
    
    {% if is_incremental() %}
        WHERE s.dbt_updated_at > (SELECT MAX(dbt_updated_at) FROM {{ this }})
    {% endif %}
)

SELECT * FROM final
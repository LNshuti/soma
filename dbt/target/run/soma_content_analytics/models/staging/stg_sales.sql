
  
  create view "soma"."main"."stg_sales__dbt_tmp" as (
    

WITH source AS (
    SELECT * FROM "soma"."raw"."sales"
),

cleaned AS (
    SELECT
        transaction_id,
        book_id,
        sale_date,
        quantity,
        unit_price,
        discount_percent,
        total_amount,
        channel,
        customer_type,
        region,
        created_at,
        EXTRACT(YEAR FROM sale_date) AS sale_year,
        EXTRACT(MONTH FROM sale_date) AS sale_month,
        EXTRACT(DOW FROM sale_date) AS day_of_week,
        CURRENT_TIMESTAMP AS dbt_updated_at
    FROM source
    WHERE quantity > 0 
      AND unit_price > 0
      AND total_amount > 0
)

SELECT * FROM cleaned
  );

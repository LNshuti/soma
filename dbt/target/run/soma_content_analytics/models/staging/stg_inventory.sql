
  
  create view "soma"."main"."stg_inventory__dbt_tmp" as (
    

WITH source AS (
    SELECT * FROM "soma"."raw"."inventory"
),

cleaned AS (
    SELECT
        inventory_id,
        book_id,
        warehouse_location,
        stock_quantity,
        reorder_point,
        last_restock_date,
        storage_cost_per_unit,
        shelf_life_days,
        created_at,
        CASE 
            WHEN stock_quantity <= reorder_point THEN true 
            ELSE false 
        END AS needs_restock,
        CURRENT_TIMESTAMP AS dbt_updated_at
    FROM source
)

SELECT * FROM cleaned
  );

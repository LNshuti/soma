

WITH source AS (
    SELECT * FROM "soma"."raw"."publishers"
),

cleaned AS (
    SELECT
        publisher_id,
        publisher_name,
        publisher_type,
        country,
        established_year,
        total_titles,
        CASE 
            WHEN active_status = 'Active' THEN true 
            ELSE false 
        END AS is_active,
        created_at,
        CURRENT_TIMESTAMP AS dbt_updated_at
    FROM source
)

SELECT * FROM cleaned

  
    
    

    create  table
      "soma"."main"."dim_publishers__dbt_tmp"
  
    as (
      

WITH publishers AS (
    SELECT * FROM "soma"."main"."stg_publishers"
),

enriched AS (
    SELECT
        publisher_id,
        publisher_name,
        publisher_type,
        country,
        established_year,
        total_titles,
        is_active,
        CASE 
            WHEN established_year < 1980 THEN 'Legacy'
            WHEN established_year < 2000 THEN 'Established'
            ELSE 'Modern'
        END AS publisher_era,
        CASE 
            WHEN total_titles > 100 THEN 'Large'
            WHEN total_titles > 20 THEN 'Medium'
            ELSE 'Small'
        END AS publisher_size,
        created_at,
        dbt_updated_at
    FROM publishers
)

SELECT * FROM enriched
    );
  
  
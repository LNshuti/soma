{{ config(materialized='view') }}

WITH source AS (
    SELECT * FROM {{ source('raw', 'books') }}
),

cleaned AS (
    SELECT
        book_id,
        isbn,
        title,
        author,
        publisher_id,
        genre,
        publication_year,
        page_count,
        price,
        format,
        language,
        created_at,
        CURRENT_TIMESTAMP AS dbt_updated_at
    FROM source
    WHERE page_count > 0 
      AND price > 0
)

SELECT * FROM cleaned
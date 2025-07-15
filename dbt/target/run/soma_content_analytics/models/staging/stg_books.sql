
  
  create view "soma"."main"."stg_books__dbt_tmp" as (
    

WITH source AS (
    SELECT * FROM "soma"."raw"."books"
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
  );

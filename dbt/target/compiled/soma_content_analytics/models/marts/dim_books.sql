

WITH books AS (
    SELECT * FROM "soma"."main"."stg_books"
),

publishers AS (
    SELECT * FROM "soma"."main"."dim_publishers"
),

enriched AS (
    SELECT
        b.book_id,
        b.isbn,
        b.title,
        b.author,
        b.publisher_id,
        p.publisher_name,
        p.publisher_type,
        b.genre,
        b.publication_year,
        b.page_count,
        b.price,
        b.format,
        b.language,
        CASE 
            WHEN b.page_count < 150 THEN 'Short'
            WHEN b.page_count < 300 THEN 'Medium'
            WHEN b.page_count < 500 THEN 'Long'
            ELSE 'Very Long'
        END AS length_category,
        CASE 
            WHEN b.price < 15 THEN 'Budget'
            WHEN b.price < 30 THEN 'Standard'
            WHEN b.price < 50 THEN 'Premium'
            ELSE 'Luxury'
        END AS price_category,
        CASE 
            WHEN b.publication_year >= 2020 THEN 'Recent'
            WHEN b.publication_year >= 2015 THEN 'Current'
            WHEN b.publication_year >= 2010 THEN 'Older'
            ELSE 'Legacy'
        END AS recency_category,
        b.created_at,
        b.dbt_updated_at
    FROM books b
    LEFT JOIN publishers p ON b.publisher_id = p.publisher_id
)

SELECT * FROM enriched
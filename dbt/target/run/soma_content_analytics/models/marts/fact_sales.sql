
        
            delete from "soma"."main"."fact_sales"
            where (
                transaction_id) in (
                select (transaction_id)
                from "fact_sales__dbt_tmp20250715124143780326"
            );

        
    

    insert into "soma"."main"."fact_sales" ("transaction_id", "book_id", "sale_date", "quantity", "unit_price", "discount_percent", "total_amount", "channel", "customer_type", "region", "sale_year", "sale_month", "day_of_week", "publisher_id", "genre", "format", "price_category", "length_category", "created_at", "dbt_updated_at")
    (
        select "transaction_id", "book_id", "sale_date", "quantity", "unit_price", "discount_percent", "total_amount", "channel", "customer_type", "region", "sale_year", "sale_month", "day_of_week", "publisher_id", "genre", "format", "price_category", "length_category", "created_at", "dbt_updated_at"
        from "fact_sales__dbt_tmp20250715124143780326"
    )
  
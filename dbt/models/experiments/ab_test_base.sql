{{ config(materialized='view') }}

WITH base AS (
  SELECT
    campaign_id,
    experiment_id,
    treatment_group,
    impressions,
    clicks,
    spend,
    CAST(clicks AS FLOAT) / NULLIF(impressions, 0) AS ctr,
    CAST(spend AS FLOAT) / NULLIF(clicks, 0) AS cpc
  FROM {{ source('raw', 'campaign_events') }}
  WHERE experiment_id IS NOT NULL
)

SELECT * FROM base
{{ config(materialized='view') }}

WITH base AS (
  SELECT * FROM {{ ref('ab_test_base') }}
)

SELECT
  experiment_id,
  treatment_group,
  COUNT(*) AS num_campaigns,
  AVG(ctr) AS avg_ctr,
  STDDEV(ctr) AS std_ctr,
  AVG(cpc) AS avg_cpc,
  STDDEV(cpc) AS std_cpc
FROM base
GROUP BY 1, 2
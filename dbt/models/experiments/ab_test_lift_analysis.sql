{{ config(materialized='view') }}

WITH summary AS (
  SELECT * FROM {{ ref('ab_test_summary') }}
),

pivoted AS (
  SELECT
    s1.experiment_id,
    s1.avg_ctr AS control_ctr,
    s2.avg_ctr AS treatment_ctr,
    s1.avg_cpc AS control_cpc,
    s2.avg_cpc AS treatment_cpc
  FROM summary s1
  JOIN summary s2
    ON s1.experiment_id = s2.experiment_id
   AND s1.treatment_group = 'control'
   AND s2.treatment_group = 'treatment'
)

SELECT
  experiment_id,
  control_ctr,
  treatment_ctr,
  COALESCE(treatment_ctr - control_ctr, 0) AS ctr_lift,
  CASE 
    WHEN control_ctr > 0 
    THEN 100.0 * (treatment_ctr - control_ctr) / control_ctr 
    ELSE 0 
  END AS ctr_lift_pct,
  control_cpc,
  treatment_cpc,
  COALESCE(control_cpc - treatment_cpc, 0) AS cpc_reduction,
  CASE 
    WHEN control_cpc > 0 
    THEN 100.0 * (control_cpc - treatment_cpc) / control_cpc 
    ELSE 0 
  END AS cpc_reduction_pct
FROM pivoted
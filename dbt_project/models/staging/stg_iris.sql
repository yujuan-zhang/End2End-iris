-- Staging 层：清洗原始数据，统一格式
-- 只有 dbt 内部用，下游不直接读这个表

{% set measurements = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'] %}

SELECT
    ROW_NUMBER() OVER () AS measurement_id,

    {% for col in measurements %}
    ROUND({{ col }}::numeric, 1) AS {{ col }},
    {% endfor %}

    species,

    CASE
        WHEN sepal_length < 5.0 THEN 'small'
        WHEN sepal_length < 6.5 THEN 'medium'
        ELSE 'large'
    END AS size_category

FROM iris

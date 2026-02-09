-- 事实表：每条测量记录，增加计算字段
-- 这是分析师和 BI 工具查询的主表

SELECT
    measurement_id,
    species,
    size_category,
    sepal_length,
    sepal_width,
    petal_length,
    petal_width,
    ROUND((sepal_length * sepal_width)::numeric, 1) AS sepal_area,
    ROUND((petal_length * petal_width)::numeric, 1) AS petal_area
FROM {{ ref('stg_iris') }}

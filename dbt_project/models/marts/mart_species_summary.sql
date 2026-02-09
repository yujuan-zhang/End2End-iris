-- 指标层：给 BI / Streamlit 用的汇总表
-- 这是最终的稳定接口，下游只读这个表

SELECT
    species,
    size_category,
    COUNT(*) AS total,
    ROUND(AVG(sepal_length), 1) AS avg_sepal_length,
    ROUND(AVG(sepal_width), 1) AS avg_sepal_width,
    ROUND(AVG(petal_length), 1) AS avg_petal_length,
    ROUND(AVG(petal_width), 1) AS avg_petal_width,
    ROUND(AVG(sepal_area), 1) AS avg_sepal_area,
    ROUND(AVG(petal_area), 1) AS avg_petal_area
FROM {{ ref('fct_measurements') }}
GROUP BY species, size_category
ORDER BY species, size_category

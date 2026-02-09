-- 维度表：每个品种的基本信息
-- BI 工具用这个表做筛选器和标签

SELECT
    species,
    COUNT(*) AS sample_count,
    ROUND(AVG(sepal_length), 1) AS avg_sepal_length,
    ROUND(AVG(petal_length), 1) AS avg_petal_length
FROM {{ ref('stg_iris') }}
GROUP BY species
ORDER BY species

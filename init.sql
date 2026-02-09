-- 1. 创建表
CREATE TABLE iris (
    sepal_length REAL,
    sepal_width REAL,
    petal_length REAL,
    petal_width REAL,
    species VARCHAR(50)
);

-- 2. 从CSV导入数据
COPY iris FROM '/data/iris.csv' DELIMITER ',' CSV HEADER;


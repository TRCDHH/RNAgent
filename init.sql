-- RNAgent 本地 MySQL 初始化脚本
-- 用法：以 root 身份执行
--   mysql -u root -p < init.sql

CREATE DATABASE IF NOT EXISTS rnagent CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- 创建业务用户（rnagent / 123456）
CREATE USER IF NOT EXISTS 'rnagent'@'localhost' IDENTIFIED BY '123456';
CREATE USER IF NOT EXISTS 'rnagent'@'127.0.0.1' IDENTIFIED BY '123456';
CREATE USER IF NOT EXISTS 'rnagent'@'%' IDENTIFIED BY '123456';
GRANT ALL PRIVILEGES ON rnagent.* TO 'rnagent'@'localhost';
GRANT ALL PRIVILEGES ON rnagent.* TO 'rnagent'@'127.0.0.1';
GRANT ALL PRIVILEGES ON rnagent.* TO 'rnagent'@'%';
FLUSH PRIVILEGES;

USE rnagent;

-- 数据集表：id, 名称, 路径, 创建时间
CREATE TABLE IF NOT EXISTS dataset (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(255) NOT NULL,
    path VARCHAR(512) NOT NULL,
    create_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 任务表：id, 名称(数据集名_任务id), 路径(输出结果目录), 过程(json)
CREATE TABLE IF NOT EXISTS task (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(255) NOT NULL,
    path VARCHAR(512),
    process TEXT,
    create_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

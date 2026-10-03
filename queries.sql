-- Total Sales

SELECT SUM(Sales)
FROM sales;


-- Total Profit

SELECT SUM(Profit)
FROM sales;


-- Sales by Category

SELECT
    Category,
    SUM(Sales) AS Total_Sales
FROM sales
GROUP BY Category
ORDER BY Total_Sales DESC;


-- Profit by Region

SELECT
    Region,
    SUM(Profit) AS Total_Profit
FROM sales
GROUP BY Region
ORDER BY Total_Profit DESC;


-- Top Products

SELECT
    "Product Name",
    SUM(Sales) AS Total_Sales
FROM sales
GROUP BY "Product Name"
ORDER BY Total_Sales DESC
LIMIT 10;
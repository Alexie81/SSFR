# Data

`products.csv` is the 20-row UTF-8 product catalog supplied with the task.
`search_queries.csv` contains the reproducible Romanian evaluation queries.

Required product columns are `product_id`, `title`, and `description`. Optional
fields are `category`, `brand`, `price_ron`, `color`, `audience`, and `in_stock`.
Price and stock are structured fields and are not included in the embedding text.

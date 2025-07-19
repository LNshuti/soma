# Data Pipeline Documentation

The SOMA data pipeline handles data generation, transformation, and preparation for analytics and machine learning workloads.

## Overview

The data pipeline consists of:
1. **Data Generation**: Synthetic data creation for development and testing
2. **Data Validation**: Quality checks and validation rules
3. **Data Transformation**: dbt-based ETL processes
4. **Feature Engineering**: ML-ready feature preparation

## Pipeline Architecture

```
Raw Data Sources → Data Generators → DuckDB (raw) → dbt → Analytics Tables → ML Features
```

## Data Generation (`src/data/generators.py`)

### Purpose
Creates realistic synthetic data for development, testing, and demonstration purposes.

### Generated Tables

#### Publishers (`raw.publishers`)
- **Records**: 10 publishers
- **Schema**:
  ```sql
  publisher_id VARCHAR      -- Unique identifier (PUB_XXXXXX)
  publisher_name VARCHAR    -- Company name
  publisher_type VARCHAR    -- Traditional, Independent, Academic, Digital
  country VARCHAR           -- Publisher location
  established_year INTEGER  -- Year founded
  total_titles INTEGER      -- Number of published titles
  active_status VARCHAR     -- Active, Inactive
  created_at TIMESTAMP      -- Record creation time
  ```

#### Books (`raw.books`)
- **Records**: 100 books
- **Schema**:
  ```sql
  book_id VARCHAR           -- Unique identifier (BOOK_XXXXXX)
  isbn VARCHAR              -- ISBN-13 number
  title VARCHAR             -- Book title
  author VARCHAR            -- Author name
  publisher_id VARCHAR      -- Foreign key to publishers
  genre VARCHAR             -- Fiction, Non-Fiction, Science, Biography, Fantasy
  publication_year INTEGER  -- Year published
  page_count INTEGER        -- Number of pages
  price DECIMAL             -- List price
  format VARCHAR            -- Hardcover, Paperback, eBook, Audiobook
  language VARCHAR          -- English, Spanish, French, German
  created_at TIMESTAMP      -- Record creation time
  ```

#### Sales Transactions (`raw.sales`)
- **Records**: 100 transactions
- **Schema**:
  ```sql
  transaction_id VARCHAR    -- Unique identifier (TXN_XXXXXXXX)
  book_id VARCHAR           -- Foreign key to books
  sale_date DATE            -- Transaction date
  quantity INTEGER          -- Number of units sold
  unit_price DECIMAL        -- Price per unit
  discount_percent DECIMAL  -- Discount applied
  total_amount DECIMAL      -- Final transaction amount
  channel VARCHAR           -- Online, Retail, Wholesale, Direct
  customer_type VARCHAR     -- Individual, Business, Educational
  region VARCHAR            -- North, South, East, West, Central
  created_at TIMESTAMP      -- Record creation time
  ```

#### Inventory (`raw.inventory`)
- **Records**: 50 inventory records
- **Schema**:
  ```sql
  inventory_id VARCHAR      -- Unique identifier (INV_XXXXXXXX)
  book_id VARCHAR           -- Foreign key to books
  warehouse_location VARCHAR -- Warehouse_A, Warehouse_B, Warehouse_C
  stock_quantity INTEGER    -- Current stock level
  reorder_point INTEGER     -- Minimum stock threshold
  last_restock_date DATE    -- Last replenishment date
  storage_cost_per_unit DECIMAL -- Storage cost
  shelf_life_days INTEGER   -- Product shelf life
  created_at TIMESTAMP      -- Record creation time
  ```

#### Campaign Events (`raw.campaign_events`)
- **Records**: 50 campaign records
- **Schema**:
  ```sql
  campaign_id VARCHAR       -- Unique identifier (CAMP_XXXXXX)
  experiment_id VARCHAR     -- A/B test identifier (optional)
  treatment_group VARCHAR   -- control, treatment (optional)
  impressions INTEGER       -- Ad impressions
  clicks INTEGER            -- Click-through count
  spend DECIMAL             -- Campaign spend
  campaign_date DATE        -- Campaign date
  created_at TIMESTAMP      -- Record creation time
  ```

### Configuration

```python
@dataclass
class DataGenerationConfig:
    n_publishers: int = 10
    n_books: int = 100
    n_sales: int = 100
    n_inventory: int = 50
    n_campaigns: int = 50
```

### Usage

```bash
# Generate all synthetic data
python -m src.data.generators

# With custom database path
DB_PATH="/custom/path/soma.duckdb" python -m src.data.generators
```

## Data Transformation (dbt)

### Model Structure

```
dbt/models/
├── sources.yml           # Source table definitions
├── staging/              # Cleaned and validated data
│   ├── stg_books.sql
│   ├── stg_publishers.sql
│   ├── stg_sales.sql
│   └── stg_inventory.sql
├── marts/                # Business logic tables
│   ├── dim_books.sql     # Book dimension
│   ├── dim_publishers.sql # Publisher dimension
│   └── fact_sales.sql    # Sales fact table
├── ml_features/          # ML-ready features
│   ├── book_features.sql
│   └── demand_forecasting_features.sql
└── experiments/          # A/B testing tables
    ├── ab_test_base.sql
    ├── ab_test_summary.sql
    └── ab_test_lift_analysis.sql
```

### Key Transformations

#### Staging Layer
- Data type casting and validation
- Null handling and default values
- Basic data quality checks
- Column renaming for consistency

#### Dimension Tables
- **dim_books**: Enhanced book data with categories
  ```sql
  -- Price categorization
  CASE 
    WHEN price < 15 THEN 'Budget'
    WHEN price < 30 THEN 'Standard'
    WHEN price < 50 THEN 'Premium'
    ELSE 'Luxury'
  END AS price_category
  
  -- Length categorization
  CASE 
    WHEN page_count < 150 THEN 'Short'
    WHEN page_count < 300 THEN 'Medium'
    WHEN page_count < 500 THEN 'Long'
    ELSE 'Very Long'
  END AS length_category
  ```

- **dim_publishers**: Publisher dimension with metadata
- **fact_sales**: Denormalized sales facts with dimensions

#### Feature Engineering
- **book_features**: ML features for recommendations
- **demand_forecasting_features**: Time series features
- Sales velocity and trend calculations
- Customer behavior patterns

### Running dbt

```bash
# Run all models
cd dbt
dbt run --profiles-dir . --target dev

# Run specific models
dbt run --models staging
dbt run --models marts.fact_sales

# Test data quality
dbt test

# Generate documentation
dbt docs generate
dbt docs serve
```

## Data Quality

### Validation Rules

#### Source Data Validation
- Non-null checks for required fields
- Data type validation
- Range checks for numeric fields
- Format validation for identifiers

#### Business Logic Validation
- Referential integrity checks
- Date consistency validation
- Price and quantity validations
- Duplicate detection

### dbt Tests

```yaml
# Example test configuration
models:
  - name: fact_sales
    tests:
      - unique:
          column_name: transaction_id
      - not_null:
          column_name: book_id
    columns:
      - name: total_amount
        tests:
          - not_null
          - dbt_utils.accepted_range:
              min_value: 0
```

## Database Schema Management

### Schema Organization
- **raw**: Source system data (read-only)
- **staging**: Cleaned and validated data
- **analytics**: Business logic and aggregations
- **ml_features**: Machine learning feature store

### Naming Conventions
- Tables: `snake_case`
- Columns: `snake_case`
- IDs: `{entity}_id` format
- Timestamps: `{action}_at` format

## Performance Optimization

### Indexing Strategy
```sql
-- Primary key indexes
CREATE UNIQUE INDEX idx_books_pk ON raw.books(book_id);
CREATE UNIQUE INDEX idx_sales_pk ON raw.sales(transaction_id);

-- Query optimization indexes
CREATE INDEX idx_sales_date ON raw.sales(sale_date);
CREATE INDEX idx_sales_book ON raw.sales(book_id);
```

### Query Optimization
- Efficient JOIN strategies
- Proper use of WHERE clauses
- Column pruning in transformations
- Incremental model updates

## Monitoring and Alerting

### Data Quality Monitoring
- Row count validations
- Schema drift detection
- Data freshness checks
- Anomaly detection

### Pipeline Monitoring
- ETL job success/failure rates
- Processing time metrics
- Data volume trends
- Error rate tracking

## Configuration Management

### Environment Variables
```bash
# Database configuration
DB_PATH=/app/data/soma.duckdb
DBT_PROFILES_DIR=/app/dbt

# Data generation settings
DATA_GENERATION_SEED=42
SYNTHETIC_DATA_SCALE=1.0
```

### dbt Profiles
```yaml
# profiles.yml
soma_content_analytics:
  target: dev
  outputs:
    dev:
      type: duckdb
      path: ../data/soma.duckdb
      schema: main
```

## Error Handling

### Common Issues
1. **Database Lock Errors**: Handle concurrent access
2. **Memory Issues**: Optimize query complexity
3. **Schema Changes**: Version control and migration
4. **Data Type Mismatches**: Explicit casting

### Recovery Procedures
- Database backup and restore
- Failed job retry mechanisms
- Data consistency checks
- Manual intervention protocols

## Development Workflow

### Local Development
1. Generate synthetic data
2. Run dbt transformations
3. Validate data quality
4. Test ML pipelines

### Testing Strategy
- Unit tests for data generators
- Integration tests for dbt models
- End-to-end pipeline validation
- Performance benchmarking

### Deployment Process
1. Schema validation
2. Data migration planning
3. Incremental deployment
4. Rollback procedures

## Future Enhancements

### Planned Features
- Real-time data streaming
- Advanced data lineage tracking
- Automated data profiling
- Enhanced quality monitoring

### Scalability Improvements
- Distributed processing
- Columnar storage optimization
- Parallel transformation execution
- Cloud data platform integration

---

*For detailed model documentation, see the [dbt docs](../dbt/README.md)*
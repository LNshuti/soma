# Architecture Overview

SOMA follows a microservices architecture designed for scalability, maintainability, and efficient content analytics processing.

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Interface │    │   API Gateway   │    │  ML Pipeline    │
│   (Gradio)      │    │   (Flask)       │    │  (Training)     │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────┴───────────┐
                    │    Database Layer       │
                    │    (DuckDB + dbt)       │
                    └─────────────────────────┘
```

## Core Components

### 1. Web Interface Layer
- **Technology**: Gradio framework
- **Purpose**: User-facing web application
- **Features**:
  - Interactive dashboards
  - Real-time analytics
  - ML model interaction
  - Report generation

### 2. API Layer
- **Technology**: Flask REST API
- **Purpose**: Backend services and data access
- **Components**:
  - Health monitoring
  - Prediction endpoints
  - Recommendation services
  - Data query interface

### 3. Data Layer
- **Database**: DuckDB (analytical workloads)
- **Transformation**: dbt (data build tool)
- **Schemas**:
  - `raw`: Source data tables
  - `staging`: Cleaned and validated data
  - `analytics`: Business logic and metrics

### 4. Machine Learning Pipeline
- **Demand Forecasting**: Time series prediction
- **Recommendation Engine**: Content-based and collaborative filtering
- **RAG System**: Retrieval-Augmented Generation for insights

## Data Flow

### 1. Data Ingestion
```
Raw Data Sources → Data Generators → DuckDB (raw schema)
```

### 2. Data Transformation
```
Raw Data → dbt Models → Staging → Analytics Tables
```

### 3. ML Processing
```
Analytics Data → Feature Engineering → Model Training → Model Artifacts
```

### 4. Serving
```
User Request → API → ML Models → Real-time Predictions
```

## Component Details

### API Services (`src/api/`)

#### Core Modules
- `app.py`: Flask application factory
- `routes/health.py`: System health monitoring
- `routes/predictions.py`: ML prediction endpoints
- `routes/recommendations.py`: Recommendation services

#### Features
- RESTful API design
- JSON response format
- Error handling and logging
- Health check endpoints

### Web Interface (`src/web/`)

#### Components
- `gradio_app.py`: Main web application
- Interactive dashboards
- Real-time data visualization
- Model interaction interfaces

#### Features
- Responsive design
- Real-time updates
- Export capabilities
- User-friendly interface

### Data Pipeline (`src/data/`)

#### Components
- `generators.py`: Synthetic data generation
- Data validation and cleaning
- Schema management

#### Synthetic Data
- Publishers (10 records)
- Books (100 records)
- Sales transactions (100 records)
- Inventory data (50 records)
- Campaign events (50 records)

### ML Models (`src/models/`)

#### Base Framework
- `base.py`: Common model interface
- Standardized training/prediction pipeline
- Model versioning and artifacts

#### Specific Models
1. **Demand Forecasting** (`forecasting/demand_model.py`)
   - Time series analysis
   - Feature engineering
   - Trend and seasonality detection

2. **Recommendation Engine** (`recommendation/engine.py`)
   - Content-based filtering
   - Collaborative filtering
   - Hybrid approach

3. **RAG System** (`rag/`)
   - Document retrieval
   - Context-aware generation
   - Knowledge base integration

### Database Schema

#### Raw Tables
- `raw.publishers`: Publisher information
- `raw.books`: Book catalog
- `raw.sales`: Transaction data
- `raw.inventory`: Stock levels
- `raw.campaign_events`: Marketing campaigns

#### Transformed Tables
- `dim_books`: Book dimension with categories
- `dim_publishers`: Publisher dimension
- `fact_sales`: Sales fact table
- `book_features`: ML feature store

## Infrastructure

### Docker Architecture
```
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  Web Container  │  │  API Container  │  │ Data Container  │
│  (soma-web)     │  │  (soma-api)     │  │ (soma-data)     │
└─────────────────┘  └─────────────────┘  └─────────────────┘
         │                     │                     │
         └─────────────────────┼─────────────────────┘
                               │
                   ┌─────────────────┐
                   │ Shared Volume   │
                   │ (Database)      │
                   └─────────────────┘
```

### Kubernetes Deployment
- **Namespace**: `soma-local`
- **Services**: API, Web, Data Setup Job
- **Storage**: Persistent Volume for database
- **Configuration**: ConfigMaps and Secrets

## Security Considerations

### Data Security
- Database file permissions
- Container security contexts
- Network policies

### API Security
- Input validation
- Error handling
- Rate limiting (planned)
- Authentication (planned)

## Scalability Design

### Horizontal Scaling
- Stateless API services
- Database read replicas
- Load balancing

### Performance Optimization
- Database indexing
- Query optimization
- Caching layers
- Async processing

## Monitoring and Observability

### Logging
- Structured JSON logging
- Centralized log aggregation
- Error tracking

### Metrics
- Application performance
- Database query performance
- ML model metrics

### Health Checks
- API endpoint health
- Database connectivity
- Model availability

## Development Workflow

### Local Development
1. Docker Compose for quick setup
2. Hot reloading for development
3. Test database seeding

### Testing Strategy
- Unit tests for individual components
- Integration tests for API endpoints
- End-to-end workflow tests

### Deployment Pipeline
1. Build Docker images
2. Run test suite
3. Deploy to Kubernetes
4. Health verification

## Future Enhancements

### Planned Features
- Real-time data streaming
- Advanced ML models
- Multi-tenant architecture
- Cloud deployment

### Technology Roadmap
- Kubernetes operators
- Service mesh integration
- Advanced monitoring
- Auto-scaling capabilities

---

*For implementation details, see the [Development Guide](development.md)*
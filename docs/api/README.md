# API Documentation

The SOMA API provides RESTful endpoints for content analytics, machine learning predictions, and system monitoring.

## Base URL

- **Local Development**: `http://localhost:5001`
- **Kubernetes**: `http://localhost:5001` (with port-forward)

## Authentication

Currently, the API operates without authentication for development purposes. Authentication will be added in future versions.

## API Endpoints

### Health Monitoring

#### GET /health
Returns the health status of the API and its dependencies.

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-19T10:30:00Z",
  "version": "1.0.0",
  "services": {
    "database": "healthy",
    "ml_models": "healthy"
  }
}
```

**Response Codes**:
- `200`: Service is healthy
- `503`: Service is unhealthy

---

### Predictions

#### POST /api/predictions/demand
Generates demand forecasting predictions for specified books.

**Request Body**:
```json
{
  "book_ids": ["BOOK_000001", "BOOK_000002"],
  "forecast_horizon": 30,
  "confidence_level": 0.95
}
```

**Response**:
```json
{
  "predictions": [
    {
      "book_id": "BOOK_000001",
      "forecasted_demand": 150.5,
      "confidence_interval": {
        "lower": 120.2,
        "upper": 180.8
      },
      "forecast_date": "2024-02-19"
    }
  ],
  "model_version": "v1.0.0",
  "generated_at": "2024-01-19T10:30:00Z"
}
```

#### POST /api/predictions/batch
Batch prediction endpoint for multiple books and time periods.

**Request Body**:
```json
{
  "requests": [
    {
      "book_id": "BOOK_000001",
      "start_date": "2024-01-01",
      "end_date": "2024-01-31"
    }
  ],
  "model_type": "demand_forecasting"
}
```

---

### Recommendations

#### GET /api/recommendations/similar/{book_id}
Get books similar to the specified book.

**Parameters**:
- `book_id` (path): The book ID to find similar items for
- `limit` (query, optional): Number of recommendations (default: 10)

**Response**:
```json
{
  "type": "content_based",
  "source_book_id": "BOOK_000001",
  "recommendations": [
    {
      "book_id": "BOOK_000015",
      "title": "Advanced Analytics",
      "author": "Jane Smith",
      "genre": "Non-Fiction",
      "price": 29.99,
      "similarity_score": 0.85,
      "reason": "Similar content and features"
    }
  ],
  "total_found": 10,
  "method": "similarity_matrix",
  "generated_at": "2024-01-19T10:30:00Z"
}
```

#### GET /api/recommendations/popular/{user_type}
Get popular books for a specific user type.

**Parameters**:
- `user_type` (path): Type of user (`Individual`, `Business`, `Educational`)
- `limit` (query, optional): Number of recommendations (default: 10)

**Response**:
```json
{
  "type": "popular_by_user_type",
  "user_type": "Individual",
  "recommendations": [
    {
      "book_id": "BOOK_000023",
      "title": "Data Science Basics",
      "author": "John Doe",
      "genre": "Science",
      "price": 24.99,
      "popularity_score": 150.0,
      "reason": "Popular among Individual customers"
    }
  ],
  "total_found": 10,
  "method": "sales_popularity",
  "generated_at": "2024-01-19T10:30:00Z"
}
```

#### POST /api/recommendations/personalized
Get personalized recommendations for a user.

**Request Body**:
```json
{
  "user_id": "user_123",
  "user_type": "Individual",
  "previous_purchases": ["BOOK_000001", "BOOK_000005"],
  "preferences": {
    "genres": ["Science", "Fiction"],
    "price_range": {
      "min": 10.0,
      "max": 50.0
    }
  },
  "limit": 10
}
```

---

### Data Query

#### GET /api/data/books
Get book catalog data with filtering and pagination.

**Query Parameters**:
- `genre` (optional): Filter by genre
- `author` (optional): Filter by author
- `price_min` (optional): Minimum price
- `price_max` (optional): Maximum price
- `page` (optional): Page number (default: 1)
- `limit` (optional): Items per page (default: 20)

**Response**:
```json
{
  "books": [
    {
      "book_id": "BOOK_000001",
      "isbn": "978-1234567890",
      "title": "Machine Learning Guide",
      "author": "Alice Johnson",
      "publisher_id": "PUB_000001",
      "genre": "Science",
      "publication_year": 2023,
      "page_count": 350,
      "price": 39.99,
      "format": "Paperback",
      "language": "English"
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 20,
    "total": 100,
    "pages": 5
  }
}
```

#### GET /api/data/sales
Get sales transaction data with filtering.

**Query Parameters**:
- `start_date` (optional): Start date (YYYY-MM-DD)
- `end_date` (optional): End date (YYYY-MM-DD)
- `book_id` (optional): Filter by book
- `channel` (optional): Sales channel
- `page` (optional): Page number
- `limit` (optional): Items per page

---

## Error Handling

### Error Response Format
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input parameters",
    "details": {
      "field": "book_id",
      "reason": "Book ID not found"
    }
  },
  "timestamp": "2024-01-19T10:30:00Z"
}
```

### Common Error Codes

| Code | Status | Description |
|------|--------|-------------|
| `VALIDATION_ERROR` | 400 | Invalid request parameters |
| `NOT_FOUND` | 404 | Resource not found |
| `MODEL_ERROR` | 422 | ML model processing error |
| `INTERNAL_ERROR` | 500 | Internal server error |
| `SERVICE_UNAVAILABLE` | 503 | Service temporarily unavailable |

## Rate Limiting

Currently not implemented. Will be added in future versions.

## SDKs and Libraries

### Python Client Example
```python
import requests

# Health check
response = requests.get('http://localhost:5001/health')
print(response.json())

# Get recommendations
response = requests.get(
    'http://localhost:5001/api/recommendations/similar/BOOK_000001',
    params={'limit': 5}
)
recommendations = response.json()
```

### JavaScript Client Example
```javascript
// Health check
fetch('http://localhost:5001/health')
  .then(response => response.json())
  .then(data => console.log(data));

// Get recommendations
fetch('http://localhost:5001/api/recommendations/similar/BOOK_000001?limit=5')
  .then(response => response.json())
  .then(data => console.log(data.recommendations));
```

## Testing the API

### Using curl
```bash
# Health check
curl http://localhost:5001/health

# Get similar books
curl "http://localhost:5001/api/recommendations/similar/BOOK_000001?limit=5"

# Get popular books for user type
curl http://localhost:5001/api/recommendations/popular/Individual
```

### Using httpie
```bash
# Health check
http GET localhost:5001/health

# Post prediction request
http POST localhost:5001/api/predictions/demand \
  book_ids:='["BOOK_000001"]' \
  forecast_horizon:=30
```

## OpenAPI Specification

The API follows OpenAPI 3.0 specification. The full specification will be available at `/api/docs` in future versions.

## Versioning

The API uses semantic versioning:
- Current version: `v1.0.0`
- Version information available in `/health` endpoint
- Breaking changes will increment major version

## Support

For API support and questions:
- Check the [troubleshooting guide](../troubleshooting.md)
- Review API logs for detailed error information
- Submit issues on GitHub

---

*Next: [Web Interface Documentation](../web-interface.md)*
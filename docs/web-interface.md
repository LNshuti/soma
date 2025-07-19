# Web Interface Guide

The SOMA web interface provides an intuitive dashboard for content analytics, built with Gradio for interactive data exploration and machine learning model interaction.

## Overview

The web interface serves as the primary user interaction point for:
- Data exploration and visualization
- ML model predictions and insights
- Real-time analytics dashboards
- Report generation and export

## Accessing the Interface

### Local Development
- **URL**: http://localhost:7860
- **Startup**: `python -m src.web.gradio_app`

### Docker Deployment
- **URL**: http://localhost:7860
- **Command**: `docker compose up web`

### Kubernetes Deployment
- **Port Forward**: `kubectl port-forward -n soma-local svc/soma-web-service 7860:7860`
- **URL**: http://localhost:7860

## Interface Components

### 1. Dashboard Overview

The main dashboard provides:
- **System Status**: Real-time health indicators
- **Key Metrics**: Sales, inventory, and performance KPIs
- **Quick Actions**: Common tasks and shortcuts
- **Recent Activity**: Latest transactions and updates

### 2. Data Explorer

#### Books Catalog
- Browse and search book inventory
- Filter by genre, author, price range
- View detailed book information
- Export book lists

#### Sales Analytics
- Transaction history and trends
- Sales performance by channel
- Customer type analysis
- Revenue metrics and forecasts

#### Inventory Management
- Stock levels and alerts
- Reorder point monitoring
- Warehouse distribution
- Inventory optimization insights

### 3. Machine Learning Tools

#### Demand Forecasting
- **Input**: Select books and forecast horizon
- **Output**: Predicted demand with confidence intervals
- **Visualization**: Time series charts and trends
- **Export**: CSV and PDF reports

**How to Use**:
1. Select one or more books from the catalog
2. Choose forecast horizon (7, 14, 30, or 90 days)
3. Set confidence level (90%, 95%, 99%)
4. Click "Generate Forecast"
5. Review results and download reports

#### Recommendation Engine
- **Content-Based**: Similar books based on features
- **Collaborative**: User behavior patterns
- **Popular Items**: Trending books by category
- **Personalized**: Custom recommendations

**How to Use**:
1. **Similar Books**: Enter a book ID to find similar titles
2. **Popular Books**: Select user type to see trending items
3. **Personalized**: Input user preferences and history
4. Review recommendations with explanations

### 4. Reports and Analytics

#### Performance Dashboard
- Sales trends and seasonality
- Top-performing books and authors
- Channel effectiveness analysis
- Customer segmentation insights

#### Executive Summary
- High-level KPIs and metrics
- Period-over-period comparisons
- Goal tracking and performance indicators
- Strategic recommendations

## Features and Functionality

### Interactive Visualizations

#### Charts and Graphs
- **Time Series**: Sales trends, demand patterns
- **Bar Charts**: Top performers, category comparisons
- **Heatmaps**: Correlation matrices, performance grids
- **Scatter Plots**: Price vs. performance analysis

#### Filters and Controls
- Date range selectors
- Category and genre filters
- Price range sliders
- Custom query builders

### Data Export Options

#### Supported Formats
- **CSV**: Raw data tables
- **PDF**: Formatted reports
- **PNG/SVG**: Chart images
- **Excel**: Workbook with multiple sheets

#### Export Locations
- Direct download to browser
- Email delivery (planned)
- Cloud storage integration (planned)

### Real-Time Updates

#### Live Data Refresh
- Automatic data synchronization
- Real-time metric updates
- Live chart animations
- Status indicator updates

#### Notifications
- System alerts and warnings
- Model completion notifications
- Data update confirmations
- Error and status messages

## User Interface Elements

### Navigation
- **Sidebar**: Main navigation menu
- **Breadcrumbs**: Current location tracking
- **Quick Links**: Frequently used features
- **Search**: Global search functionality

### Input Controls
- **Dropdowns**: Category and option selection
- **Sliders**: Numeric range inputs
- **Date Pickers**: Time period selection
- **Text Fields**: Search and filter inputs

### Output Displays
- **Data Tables**: Sortable and filterable grids
- **Cards**: Summary information blocks
- **Progress Bars**: Task completion status
- **Alert Boxes**: Important messages and warnings

## Customization Options

### Dashboard Layout
- Widget arrangement and sizing
- Theme selection (light/dark)
- Font size and accessibility options
- Chart color schemes

### User Preferences
- Default date ranges
- Preferred chart types
- Export format defaults
- Notification settings

## Mobile Responsiveness

The interface is optimized for:
- **Desktop**: Full feature set
- **Tablet**: Responsive layout adaptation
- **Mobile**: Core functionality with simplified UI

## Performance Considerations

### Loading Times
- Progressive data loading
- Cached query results
- Optimized chart rendering
- Lazy loading for large datasets

### Browser Compatibility
- Chrome 90+ (recommended)
- Firefox 88+
- Safari 14+
- Edge 90+

## Troubleshooting

### Common Issues

#### Interface Not Loading
1. Check if API service is running
2. Verify database connectivity
3. Review browser console for errors
4. Clear browser cache and cookies

#### Slow Performance
1. Check network connectivity
2. Reduce data query size
3. Use date range filters
4. Close unused browser tabs

#### Charts Not Displaying
1. Enable JavaScript in browser
2. Disable ad blockers temporarily
3. Update browser to latest version
4. Check for console errors

### Getting Help

#### Error Messages
- Review error details in UI
- Check browser developer tools
- Examine application logs
- Reference troubleshooting guide

#### Support Resources
- [Troubleshooting Guide](troubleshooting.md)
- [API Documentation](api/README.md)
- Application logs and monitoring
- GitHub issue tracker

## Advanced Features

### Custom Queries
- SQL query builder interface
- Saved query templates
- Query history and favorites
- Result set management

### Integration Options
- REST API integration
- Webhook notifications
- External data sources
- Third-party analytics tools

### Automation
- Scheduled report generation
- Automated model retraining
- Alert and notification rules
- Workflow automation

## Security and Privacy

### Data Protection
- Local data processing
- Secure API communications
- User session management
- Data anonymization options

### Access Control
- Role-based permissions (planned)
- Feature access restrictions
- Audit logging
- Session timeout management

## Future Enhancements

### Planned Features
- Advanced visualization types
- Collaborative workspaces
- Real-time collaboration
- Enhanced mobile experience

### Integration Roadmap
- Business intelligence tools
- Cloud data platforms
- Machine learning platforms
- Enterprise analytics suites

---

*For technical implementation details, see the [Development Guide](development.md)*
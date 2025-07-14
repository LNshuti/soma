#!/usr/bin/env python3
"""
Entry point for synthetic data generation
"""

import sys
import os
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Main entry point for data generation"""
    try:
        # Import after path setup
        from src.data.generators import ContentDataGenerator, DataGenerationConfig
        from src.data.validators import DataValidator
        from src.utils.database import DatabaseManager
        from src.utils.helpers import setup_logging
        
        logger = setup_logging(__name__)
        logger.info("Starting Soma Data Generation")
        
        # Ensure data directory exists
        data_dir = project_root / "data"
        data_dir.mkdir(exist_ok=True)
        
        # Initialize database connection
        db_path = data_dir / "soma.duckdb"  # Changed from soma_id.duckdb
        logger.info(f" Database path: {db_path}")
        
        # Initialize database manager
        db = DatabaseManager(str(db_path))
        
        # Initialize configuration
        config = DataGenerationConfig(
            n_publishers=100000,    # Reduce as needed for testing
            n_books=100000,
            n_sales=100000,
            n_inventory=50000,
            n_campaigns=50000
        )
        
        logger.info(f"Generation config: {config}")
        
        # Generate data
        logger.info("Generating synthetic data...")
        generator = ContentDataGenerator(config=config)
        results = generator.generate_all()
        
        logger.info("Data generation completed!")
        logger.info(f"Generated data summary:")
        for table, count in results.items():
            if isinstance(count, int):
                logger.info(f"  {table}: {count:,} records")
        
        # Validate data
        logger.info("Validating generated data...")
        validator = DataValidator(db)
        validation_results = validator.validate_all_tables()
        
        # Report validation results
        all_passed = True
        for table, result in validation_results.items():
            status = result.get('overall_status', 'ERROR')
            record_count = result.get('record_count', 0)
            logger.info(f"  {table}: {status} ({record_count:,} records)")
            
            if status == 'FAIL':
                all_passed = False
                issues = result.get('issues', [])
                for issue in issues:
                    logger.warning(f"    Issue: {issue}")
            elif status == 'ERROR':
                all_passed = False
                error = result.get('error', 'Unknown error')
                logger.error(f"    Error: {error}")
        
        if all_passed:
            logger.info("All validation checks passed!")
        else:
            logger.warning("Some validation checks failed - review above")
        
        logger.info("Next steps:")
        logger.info("   1. Check the generated data in the database")
        logger.info("   2. Run any additional processing needed")
        logger.info("   3. Start the application services")
        
        return 0
        
    except ImportError as e:
        print(f"Import error: {e}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Python path: {sys.path}")
        return 1
    except Exception as e:
        print(f"Data generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Cleanup
        try:
            if 'db' in locals():
                db.close()
        except Exception:
            pass

if __name__ == "__main__":
    sys.exit(main())
from schema_enricher import SchemaEnricher
from vector_store import VectorStore
import logging
import os
from dotenv import load_dotenv
import sys
import time

def setup_logging():

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('digest.log')
        ]
    )
    return logging.getLogger("SchemaDigest")

def create_default_ignore_file():

    if not os.path.exists('.nlsqlignore'):
        with open('.nlsqlignore', 'w') as f:
            f.write("""# Tables to exclude from natural language SQL processing
# Each line is a pattern (glob-style) matching table names to ignore
# Examples:
# temp_*          # Ignore all tables starting with temp_
# *_backup        # Ignore all tables ending with _backup
# test_table      # Ignore specific table
# *_log*          # Ignore any table containing _log
#
# Lines starting with # are comments
""")
        return True
    return False

def main():

    logger = setup_logging()
    
    try:
        if create_default_ignore_file():
            logger.info("Created default .nlsqlignore file")
            print("\nA default .nlsqlignore file has been created.")
            print("Please review and modify it to specify tables you want to exclude.")
            print("Then run this script again.")
            sys.exit(0)
        
        load_dotenv()
        
        database_url = os.getenv('DATABASE_URL')
        openai_api_key = os.getenv('OPENAI_API_KEY')
        qdrant_url = os.getenv('QDRANT_URL', 'localhost')
        qdrant_port = int(os.getenv('QDRANT_PORT', '6333'))
        
        if not database_url or not openai_api_key:
            raise ValueError("DATABASE_URL and OPENAI_API_KEY must be set in .env file")
        
        logger.info("Starting schema digestion process...")
        start_time = time.time()
        
        logger.info("Step 1: Enriching schema with OpenAI descriptions...")
        enricher = SchemaEnricher(database_url, openai_api_key)
        
        existing_schema = enricher.load_enriched_schema()
        if existing_schema:
            logger.info("Found existing enriched schema. To regenerate, delete the cache directory first.")
            should_continue = input("Do you want to continue with the existing schema? (y/n): ").lower()
            if should_continue != 'y':
                logger.info("Exiting as per user request.")
                sys.exit(0)
            enriched_schema = existing_schema
        else:
            logger.info("Generating new enriched schema...")
            enriched_schema = enricher.process_schema()
        
        logger.info("Step 2: Generating and storing embeddings in Qdrant...")
        vector_store = VectorStore(
            openai_api_key,
            qdrant_url,
            qdrant_port
        )
        
        vector_store.store_schema_embeddings(enriched_schema)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Schema digestion completed successfully in {elapsed_time:.2f} seconds")
        logger.info(f"Processed {len(enriched_schema['tables'])} tables")
        
        print("\nDigestion Summary:")
        print("=" * 50)
        print(f"Total tables processed: {len(enriched_schema['tables'])}")
        
        ignored_patterns = enriched_schema['metadata'].get('ignored_patterns', [])
        if ignored_patterns:
            print("\nIgnored table patterns:")
            for pattern in ignored_patterns:
                print(f"  - {pattern}")
        
        print(f"\nCache location: {enricher.cache_dir}")
        print(f"Enriched schema file: {enricher.enriched_schema_file}")
        print(f"Vector store: {qdrant_url}:{qdrant_port}")
        print("\nYou can now use the improved chat interface to query your database.")
        
    except KeyboardInterrupt:
        logger.error("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during schema digestion: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
from schema_enricher import SchemaEnricher
from vector_store import VectorStore
from query_processor import QueryProcessor
import logging
import os
from dotenv import load_dotenv
import sys
from typing import Optional, Tuple
import pandas as pd

class ImprovedChatInterface:

    def __init__(self):

        load_dotenv()
        
        self.database_url = os.getenv('DATABASE_URL')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.qdrant_url = os.getenv('QDRANT_URL', 'localhost')
        self.qdrant_port = int(os.getenv('QDRANT_PORT', '6333'))
        
        if not self.database_url or not self.openai_api_key:
            raise ValueError("DATABASE_URL and OPENAI_API_KEY must be set in .env file")
        
        if not os.path.exists(os.path.join(".cache", "enriched_schema.json")):
            raise ValueError("Schema not digested. Please run improved_digest_schema.py first")
        
        self.logger = logging.getLogger("ImprovedChatInterface")
        self.logger.setLevel(logging.INFO)
        
        self.schema_enricher = SchemaEnricher(self.database_url, self.openai_api_key)
        self.vector_store = VectorStore(
            self.openai_api_key,
            self.qdrant_url,
            self.qdrant_port
        )
        self.query_processor = QueryProcessor(self.openai_api_key, self.database_url)

    def process_query(self, query: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:

        try:
            relevant_tables = self.vector_store.find_relevant_tables(query)
            
            if not relevant_tables:
                raise ValueError("No relevant tables found for the query")
            
            sql_query = self.query_processor.generate_sql(query, relevant_tables)
            
            if not sql_query:
                raise ValueError("Failed to generate SQL query")
            
            return self.query_processor.execute_query(sql_query)
            
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            return None, None

    def show_schema(self):

        enriched_schema = self.schema_enricher.load_enriched_schema()
        if not enriched_schema:
            print("Error: Schema information not available")
            return
        
        print("\nAvailable Tables:")
        print("=" * 50)
        
        for table_name, table_info in enriched_schema['tables'].items():
            print(f"\nTable: {table_name}")
            print("-" * len(f"Table: {table_name}"))
            
            print("\nDescription:")
            print(table_info['description'])
            print("\nSchema:")
            
            print("Columns:")
            for col_name, col_info in table_info['schema']['columns'].items():
                attrs = []
                if not col_info['nullable']:
                    attrs.append("NOT NULL")
                if col_info.get('primary_key'):
                    attrs.append("PRIMARY KEY")
                attr_str = " ".join(attrs)
                print(f"  - {col_name} ({col_info['type']}) {attr_str}".rstrip())
            
            if table_info['schema']['foreign_keys']:
                print("\nRelationships:")
                for fk in table_info['schema']['foreign_keys']:
                    print(f"  - References {fk['referred_table']} "
                          f"({', '.join(fk['referred_columns'])})")
            print()

def main():
    try:

        chat = ImprovedChatInterface()
        
        print("\nWelcome to Improved Database Chat!")
        print("Type 'exit' to quit, 'schema' to see database structure.")
        print("=" * 70)
        
        while True:

            user_input = input("\nEnter your question: ").strip()
            
            if user_input.lower() == 'exit':
                print("\nGoodbye!")
                break
            elif user_input.lower() == 'schema':
                chat.show_schema()
                continue
            elif not user_input:
                continue
            
            print("\nProcessing query...")
            result, sql_query = chat.process_query(user_input)
            
            if isinstance(result, pd.DataFrame):
                if sql_query:
                    print("\nGenerated SQL Query:")
                    print("-" * 50)
                    print(sql_query)
                    print("-" * 50)
                
                if len(result) > 0:
                    print("\nResults:")
                    print("-" * 50)

                    pd.set_option('display.max_columns', None)
                    pd.set_option('display.expand_frame_repr', False)
                    pd.set_option('display.max_rows', 20)
                    print(result)
                    print(f"\nTotal rows: {len(result)}")
                else:
                    print("\nNo results found.")
            else:
                print("\nError processing query")
                if sql_query:
                    print("\nAttempted SQL Query:")
                    print("-" * 50)
                    print(sql_query)

    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main() 
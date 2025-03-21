from sqlalchemy import create_engine, inspect
from openai import OpenAI
import json
import os
from typing import Dict, List, Any, Set
import logging
from datetime import datetime
import fnmatch

class SchemaEnricher:

    def __init__(self, database_url: str, openai_api_key: str, cache_dir: str = ".cache"):

        self.database_url = database_url
        self.engine = create_engine(database_url)
        self.client = OpenAI(api_key=openai_api_key)
        self.cache_dir = cache_dir
        self.enriched_schema_file = os.path.join(cache_dir, "enriched_schema.json")
        
        self.logger = logging.getLogger("SchemaEnricher")
        self.logger.setLevel(logging.INFO)
        
        os.makedirs(cache_dir, exist_ok=True)
        
        self.ignored_tables = self._load_ignored_tables()

    def _load_ignored_tables(self) -> Set[str]:

        ignored_tables = set()
        if os.path.exists('.nlsqlignore'):
            with open('.nlsqlignore', 'r') as f:
                for line in f:
                    line = line.split('#')[0].strip()
                    if line:  # Skip empty lines
                        ignored_tables.add(line)
        return ignored_tables

    def _should_process_table(self, table_name: str) -> bool:

        for pattern in self.ignored_tables:
            if fnmatch.fnmatch(table_name.lower(), pattern.lower()):
                self.logger.info(f"Skipping table {table_name} (matched pattern {pattern})")
                return False
        return True

    def extract_schema(self) -> Dict[str, Any]:

        inspector = inspect(self.engine)
        schema_info = {}
        
        for table_name in inspector.get_table_names():

            if not self._should_process_table(table_name):
                continue
                
            columns = inspector.get_columns(table_name)
            foreign_keys = inspector.get_foreign_keys(table_name)
            primary_key = inspector.get_pk_constraint(table_name)
            indexes = inspector.get_indexes(table_name)
            
            schema_info[table_name] = {
                'columns': {
                    col['name']: {
                        'type': str(col['type']),
                        'nullable': col.get('nullable', True),
                        'default': str(col.get('default')),
                        'primary_key': col['name'] in primary_key['constrained_columns'] if primary_key else False
                    } for col in columns
                },
                'foreign_keys': foreign_keys,
                'primary_key': primary_key.get('constrained_columns', []),
                'indexes': indexes
            }
        
        return schema_info

    def enrich_table_schema(self, table_name: str, schema_info: Dict) -> str:

        table_info = schema_info[table_name]
        
        context = f"Table: {table_name}\n"
        context += "Columns:\n"
        for col_name, col_info in table_info['columns'].items():
            attrs = []
            if not col_info['nullable']:
                attrs.append("NOT NULL")
            if col_info['primary_key']:
                attrs.append("PRIMARY KEY")
            if col_info['default'] != "None":
                attrs.append(f"DEFAULT {col_info['default']}")
            context += f"- {col_name} ({col_info['type']}) {' '.join(attrs)}\n"
        
        if table_info['foreign_keys']:
            context += "\nRelationships:\n"
            for fk in table_info['foreign_keys']:
                context += f"- References {fk['referred_table']} ({', '.join(fk['referred_columns'])})\n"
        
        messages = [
            {"role": "system", "content": """You are a database expert. Analyze the provided table schema and generate a detailed description including:
1. The purpose of the table
2. Explanation of key columns
3. Relationships with other tables
4. Common business use cases
Be concise but comprehensive."""},
            {"role": "user", "content": context}
        ]
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        
        return response.choices[0].message.content.strip()

    def process_schema(self) -> Dict[str, Any]:
        
        self.logger.info("Starting schema enrichment process...")
        schema_info = self.extract_schema()
        enriched_schema = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'database_url': self.database_url,
                'ignored_patterns': list(self.ignored_tables)
            },
            'tables': {}
        }
        
        for table_name in schema_info.keys():
            self.logger.info(f"Enriching schema for table: {table_name}")
            description = self.enrich_table_schema(table_name, schema_info)
            enriched_schema['tables'][table_name] = {
                'schema': schema_info[table_name],
                'description': description
            }
        
        with open(self.enriched_schema_file, 'w') as f:
            json.dump(enriched_schema, f, indent=2)
        
        self.logger.info("Schema enrichment completed successfully")
        return enriched_schema

    def load_enriched_schema(self) -> Dict[str, Any]:
        
        if os.path.exists(self.enriched_schema_file):
            with open(self.enriched_schema_file, 'r') as f:
                return json.load(f)
        return None 
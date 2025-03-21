from openai import OpenAI
from typing import Dict, List, Any, Tuple
import logging
from sqlalchemy import create_engine, text
import pandas as pd
import re

class QueryProcessor:

    def __init__(self, openai_api_key: str, database_url: str):

        self.client = OpenAI(api_key=openai_api_key)
        self.database_url = database_url
        self.engine = create_engine(database_url)
        
        self.logger = logging.getLogger("QueryProcessor")
        self.logger.setLevel(logging.INFO)
        
        self.MAX_TOKENS = 4000
        self.TOKENS_PER_TABLE = 800

    def _needs_quoting(self, identifier: str) -> bool:

        pg_keywords = {
            'all', 'analyse', 'analyze', 'and', 'any', 'array', 'as', 'asc', 
            'asymmetric', 'authorization', 'binary', 'both', 'case', 'cast', 
            'check', 'collate', 'column', 'constraint', 'create', 'cross', 
            'current_date', 'current_role', 'current_time', 'current_timestamp', 
            'current_user', 'default', 'deferrable', 'desc', 'distinct', 'do', 
            'else', 'end', 'except', 'false', 'for', 'foreign', 'freeze', 'from', 
            'full', 'grant', 'group', 'having', 'ilike', 'in', 'initially', 'inner', 
            'intersect', 'into', 'is', 'isnull', 'join', 'leading', 'left', 'like', 
            'limit', 'localtime', 'localtimestamp', 'natural', 'not', 'notnull', 
            'null', 'offset', 'on', 'only', 'or', 'order', 'outer', 'overlaps', 
            'placing', 'primary', 'references', 'right', 'select', 'session_user', 
            'similar', 'some', 'symmetric', 'table', 'then', 'to', 'trailing', 
            'true', 'union', 'unique', 'user', 'using', 'verbose', 'when', 'where'
        }
        
        return (
            identifier.upper() == identifier or
            identifier.lower() != identifier or
            ' ' in identifier or
            any(c in identifier for c in '.-') or
            identifier.lower() in pg_keywords
        )

    def _quote_identifier(self, identifier: str) -> str:

        if self._needs_quoting(identifier):
            return f'"{identifier}"'
        return identifier

    def _process_sql_query(self, sql_query: str, table_names: List[str]) -> str:

        table_case_map = {name.lower(): name for name in table_names}
        
        table_pattern = r'\b(FROM|JOIN|UPDATE|INTO|TABLE)\s+([a-zA-Z_][a-zA-Z0-9_]*)\b'
        
        def replace_table_name(match):
            keyword = match.group(1)
            table_name = match.group(2)
            original_case = table_case_map.get(table_name.lower(), table_name)
            quoted_name = self._quote_identifier(original_case)
            return f"{keyword} {quoted_name}"
        
        processed_query = re.sub(table_pattern, replace_table_name, sql_query, flags=re.IGNORECASE)
        return processed_query

    def validate_tables(self, relevant_tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

        context = "Given the following tables and their relevance scores, analyze which ones are most appropriate for the query:\n\n"
        for table in relevant_tables:
            table_name = self._quote_identifier(table['table_name'])
            context += f"Table: {table_name}\n"
            context += f"Relevance Score: {table['similarity_score']}\n"
            context += f"Description: {table['description']}\n\n"
        
        messages = [
            {"role": "system", "content": """You are a database expert. Analyze the provided tables and their relevance scores.
Return only the table names that are truly relevant, ordered by importance.
Format: table1,table2,table3
Note: Preserve the exact case of table names."""},
            {"role": "user", "content": context}
        ]
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=messages
        )
        
        validated_tables = response.choices[0].message.content.strip().split(',')
        
        return [
            table for table in relevant_tables 
            if table['table_name'] in validated_tables
        ]

    def process_schema_chunk(self, tables: List[Dict[str, Any]], query: str) -> str:

        schema_context = ""
        table_names = []
        for table in tables:
            table_name = self._quote_identifier(table['table_name'])
            table_names.append(table['table_name'])
            schema_context += f"\nTable: {table_name}\n"
            schema_context += f"Description: {table['description']}\n"
            schema_context += "Columns:\n"
            for col_name, col_info in table['schema']['columns'].items():
                attrs = []
                if not col_info['nullable']:
                    attrs.append("NOT NULL")
                if col_info.get('primary_key'):
                    attrs.append("PRIMARY KEY")
                schema_context += f"- {col_name} ({col_info['type']}) {' '.join(attrs)}\n"
            
            if table['schema']['foreign_keys']:
                schema_context += "\nForeign Keys:\n"
                for fk in table['schema']['foreign_keys']:
                    referred_table = self._quote_identifier(fk['referred_table'])
                    schema_context += f"- {', '.join(fk['constrained_columns'])} -> "
                    schema_context += f"{referred_table}({', '.join(fk['referred_columns'])})\n"
        
        messages = [
            {"role": "system", "content": f"""You are a SQL expert. Using the following schema, generate a SQL query for the user's request.
The query should be efficient and use proper joins when necessary.

{schema_context}

Important notes:
1. Some table names are case-sensitive. Use the exact table names as shown above.
2. Generate ONLY the SQL query without any markdown formatting or explanation.
3. Do not include ```sql or ``` markers.
4. For PostgreSQL, use NOW() - INTERVAL '1 month' for date arithmetic.
5. Do not include trailing commas in column lists.
6. Use table aliases for better readability (e.g., emp for employee).
7. Ensure proper SQL syntax, especially in SELECT clause.
8. Use LEFT JOINs when joining optional tables to preserve main records."""},
            {"role": "user", "content": query}
        ]
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=messages
        )
        
        sql_query = response.choices[0].message.content.strip()

        sql_query = sql_query.replace('```sql', '').replace('```', '').strip()
        
        sql_query = re.sub(r',(\s*FROM)', r'\1', sql_query)
        sql_query = re.sub(r',(\s*WHERE)', r'\1', sql_query)
        sql_query = re.sub(r',(\s*GROUP\s+BY)', r'\1', sql_query)
        sql_query = re.sub(r',(\s*ORDER\s+BY)', r'\1', sql_query)
        
        return self._process_sql_query(sql_query, table_names)

    def generate_sql(self, query: str, relevant_tables: List[Dict[str, Any]]) -> str:

        validated_tables = self.validate_tables(relevant_tables)
        
        tables_per_chunk = max(1, self.MAX_TOKENS // self.TOKENS_PER_TABLE)
        
        if len(validated_tables) <= tables_per_chunk:
            return self.process_schema_chunk(validated_tables, query)
        
        best_query = None
        highest_confidence = 0
        
        for i in range(0, len(validated_tables), tables_per_chunk):
            chunk = validated_tables[i:i + tables_per_chunk]
            proposed_query = self.process_schema_chunk(chunk, query)
            
            confidence = self.evaluate_query_confidence(proposed_query, query, chunk)
            
            if confidence > highest_confidence:
                best_query = proposed_query
                highest_confidence = confidence
        
        return best_query

    def evaluate_query_confidence(self, sql_query: str, original_query: str, 
                                tables: List[Dict[str, Any]]) -> float:

        context = f"""Original question: {original_query}
Generated SQL: {sql_query}

Available tables:
"""
        for table in tables:
            table_name = self._quote_identifier(table['table_name'])
            context += f"- {table_name}\n"
        
        messages = [
            {"role": "system", "content": """You are a SQL expert. Evaluate the confidence score
(0.0 to 1.0) of the generated SQL query based on:
1. Query completeness
2. Proper table usage and case sensitivity
3. Correct joins
4. Appropriate filtering
Return only the numeric score."""},
            {"role": "user", "content": context}
        ]
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=messages
        )
        
        try:
            return float(response.choices[0].message.content.strip())
        except ValueError:
            return 0.0

    def execute_query(self, sql_query: str) -> Tuple[pd.DataFrame, str]:

        try:
            with self.engine.connect() as connection:
                result = connection.execute(text(sql_query))
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
                return df, sql_query
        except Exception as e:
            self.logger.error(f"Error executing query: {str(e)}")
            raise 
from qdrant_client import QdrantClient
from qdrant_client.http import models
from openai import OpenAI
import numpy as np
from typing import Dict, List, Any, Optional
import logging
import json
import os

class VectorStore:

    def __init__(self, openai_api_key: str, qdrant_url: str = "localhost", 
                 qdrant_port: int = 6333, collection_name: str = "schema_embeddings_new"):
        
        self.client = OpenAI(api_key=openai_api_key)
        self.qdrant = QdrantClient(host=qdrant_url, port=qdrant_port)
        self.collection_name = collection_name
        
        self.logger = logging.getLogger("VectorStore")
        self.logger.setLevel(logging.INFO)
        
        self._init_collection()

    def _init_collection(self):

        try:
            self.qdrant.get_collection(self.collection_name)
        except Exception:
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=1536,
                    distance=models.Distance.COSINE
                )
            )

    def generate_embedding(self, text: str) -> np.ndarray:

        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return np.array(response.data[0].embedding)

    def store_schema_embeddings(self, enriched_schema: Dict[str, Any]):

        self.logger.info("Storing schema embeddings...")
        
        self.qdrant.delete_collection(self.collection_name)
        self._init_collection()
        
        points = []
        for table_name, table_info in enriched_schema['tables'].items():
            text_to_embed = f"Table: {table_name}\n{table_info['description']}"
            
            embedding = self.generate_embedding(text_to_embed)
            
            point = models.PointStruct(
                id=len(points),
                vector=embedding.tolist(),
                payload={
                    'table_name': table_name,
                    'description': table_info['description'],
                    'schema': table_info['schema']
                }
            )
            points.append(point)
        
        self.qdrant.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        self.logger.info(f"Stored embeddings for {len(points)} tables")

    def find_relevant_tables(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        
        query_embedding = self.generate_embedding(query)
        
        search_result = self.qdrant.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=top_k
        )
        
        relevant_tables = []
        for scored_point in search_result:
            relevant_tables.append({
                'table_name': scored_point.payload['table_name'],
                'description': scored_point.payload['description'],
                'schema': scored_point.payload['schema'],
                'similarity_score': scored_point.score
            })
        
        return relevant_tables 
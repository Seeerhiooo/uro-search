from http.server import BaseHTTPRequestHandler
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import json
import os

model = SentenceTransformer('intfloat/multilingual-e5-large')

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        query = self.path.split('?q=')[-1]
        query = query.replace('+', ' ')

        pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
        index = pc.Index('uro-archive')

        embedding = model.encode(
            f'query: {query}',
            normalize_embeddings=True
        ).tolist()

        results = index.query(
            vector=embedding,
            top_k=10,
            include_metadata=True
        )

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(results.matches, default=str).encode())
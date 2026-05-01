from http.server import BaseHTTPRequestHandler
import json
import os

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query).get('q', [''])[0]
        
        hf_token = os.environ.get('HF_TOKEN', 'NOT SET')
        pinecone_key = os.environ.get('PINECONE_API_KEY', 'NOT SET')
        
        result = {
            'query': query,
            'hf_token_set': hf_token != 'NOT SET',
            'hf_token_length': len(hf_token),
            'pinecone_set': pinecone_key != 'NOT SET'
        }
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(result).encode())
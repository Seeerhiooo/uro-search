from http.server import BaseHTTPRequestHandler
from pinecone import Pinecone
import json
import os
import urllib.request
import urllib.error

def get_embedding(text):
    hf_token = os.environ.get('HF_TOKEN', '')
    url = "https://router.huggingface.co/hf-inference/models/intfloat/multilingual-e5-large/v1/feature-extraction"
    data = json.dumps({"inputs": f"query: {text}"}).encode()
    req = urllib.request.Request(url, data=data, headers={
        'Authorization': f'Bearer {hf_token}',
        'Content-Type': 'application/json'
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            result = response.read()
            parsed = json.loads(result)
            return parsed[0] if isinstance(parsed[0], list) else parsed
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        raise Exception(f"HF Error {e.code}: {body}")
    except Exception as e:
        raise Exception(f"General error: {str(e)}")

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query).get('q', [''])[0]

        if not query:
            self._respond(200, [])
            return

        try:
            embedding = get_embedding(query)
            pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
            index = pc.Index('uro-archive')
            results = index.query(
                vector=embedding,
                top_k=10,
                include_metadata=True
            )
            self._respond(200, [{
                'score': m.score,
                'metadata': m.metadata
            } for m in results.matches])
        except Exception as e:
            self._respond(500, {'error': str(e)})

    def _respond(self, code, data):
        self.send_response(code)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
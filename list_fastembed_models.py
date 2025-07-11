from fastembed import TextEmbedding
import json 

# List all supported models
print(json.dumps(TextEmbedding.list_supported_models(), indent=4))
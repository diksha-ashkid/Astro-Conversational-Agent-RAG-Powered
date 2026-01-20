import json
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import os

# Initialize embedding model
print("Loading embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def create_faiss_index_for_file(file_path, category, file_type='json'):
    """Create a separate FAISS index for each data file"""
    
    documents = []
    metadata = []
    
    if file_type == 'json':
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for key, value in data.items():
            if isinstance(value, dict):
                text = f"{key}: "
                for sub_key, sub_value in value.items():
                    text += f"{sub_key}: {sub_value}. "
                documents.append(text.strip())
                metadata.append({
                    'source': file_path,
                    'category': category,
                    'key': key,
                    'type': 'json'
                })
            else:
                text = f"{key}: {value}"
                documents.append(text)
                metadata.append({
                    'source': file_path,
                    'category': category,
                    'key': key,
                    'type': 'json'
                })
    
    elif file_type == 'txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        

        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        for i, para in enumerate(paragraphs):
            if len(para) > 20:  # Skip very short paragraphs
                documents.append(para)
                metadata.append({
                    'source': file_path,
                    'category': category,
                    'chunk_id': i,
                    'type': 'text'
                })
    
    if not documents:
        print(f"⚠️ No documents found in {file_path}")
        return None, None, None
    
    # Create embeddings
    print(f"  Creating {len(documents)} embeddings...")
    embeddings = embedding_model.encode(documents, show_progress_bar=False)
    embeddings = np.array(embeddings).astype('float32')
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    return index, documents, metadata

def build_separate_vector_stores():

    # Define all data sources
    data_sources = {
        'zodiac_traits': {
            'path': '/data/zodiac_traits.json',
            'type': 'json'
        },
        'planetary_impacts': {
            'path': '/data/planetary_impacts.json',
            'type': 'json'
        },
        'nakshatra_mapping': {
            'path': '/data/nakshatra_mapping.json',
            'type': 'json'
        },
        'career_guidance': {
            'path': '/data/career_guidance.txt',
            'type': 'txt'
        },
        'love_guidance': {
            'path': '/data/love_guidance.txt',
            'type': 'txt'
        },
        'spiritual_guidance': {
            'path': '/data/spiritual_guidance.txt',
            'type': 'txt'
        }
    }
    
    vector_stores = {}
    
    for category, info in data_sources.items():
        file_path = info['path']
        file_type = info['type']
        
        if not os.path.exists(file_path):
            print(f"⚠️ Warning: {file_path} not found, skipping...")
            continue
        
        print(f"\n📁 Processing {category}...")
        index, documents, metadata = create_faiss_index_for_file(file_path, category, file_type)
        
        if index is not None:
            # Saved index
            index_filename = f"faiss_{category}.bin"
            metadata_filename = f"faiss_{category}_meta.pkl"
            
            faiss.write_index(index, index_filename)
            
            with open(metadata_filename, "wb") as f:
                pickle.dump({
                    'documents': documents,
                    'metadata': metadata,
                    'category': category
                }, f)
            
            vector_stores[category] = {
                'index_file': index_filename,
                'metadata_file': metadata_filename,
                'document_count': len(documents)
            }
            
            print(f"✅ Created {category}: {len(documents)} documents")
    
    # Saved index registry
    with open("vector_store_registry.pkl", "wb") as f:
        pickle.dump(vector_stores, f)
    
    print(f"\n{'='*60}")
    print("✅ ALL VECTOR STORES CREATED SUCCESSFULLY!")
    print(f"{'='*60}")
    print(f"\nCreated {len(vector_stores)} separate FAISS indexes:")
    for category, info in vector_stores.items():
        print(f"  📊 {category}: {info['document_count']} documents")
    print(f"\n📝 Registry saved to: vector_store_registry.pkl")

if __name__ == "__main__":
    # Create data folder if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    build_separate_vector_stores()
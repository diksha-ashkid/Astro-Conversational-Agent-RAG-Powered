from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from typing import Optional, Dict, List
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime
import requests
import json
from functools import lru_cache
from googletrans import Translator
import re
import os

# initialising
app = FastAPI(title="Astro Insight Agent", version="2.0.0")


embedding_model = None
vector_stores = {}  # faiss indexes
translator = Translator()
session_memory = {}  #session storage
prompt_cache = {}  #prompt caching

# Toxic words filter do that harmfulcontent and conversations dont go forward
TOXIC_PATTERNS = [
    r'\b(kill|murder|suicide|harm|abuse|hate)\b',
    r'\b(terrorist|weapon|bomb|drug)\b'
]


class UserProfile(BaseModel):
    name: str
    birth_date: str  # Format: YYYY-MM-DD
    birth_time: str  # Format: HH:MM
    birth_place: str
    preferred_language: Optional[str] = "en"
    
    @validator('birth_date')
    def validate_birth_date(cls, v):
        try:
            datetime.strptime(v, '%Y-%m-%d')
            return v
        except ValueError:
            raise ValueError('birth_date must be in YYYY-MM-DD format')

class ChatRequest(BaseModel):
    session_id: str
    message: str
    user_profile: UserProfile

class RetrievalScore(BaseModel):
    """Model for individual retrieval scores"""
    rank: int
    confidence: float
    category: str
    distance: float

class ChatResponse(BaseModel):
    response: str
    context_used: List[str]
    zodiac: str
    confidence_score: float
    retrieval_scores: List[RetrievalScore]


@app.on_event("startup")
async def load_models():
    """Load models and all FAISS indexes"""
    global embedding_model, vector_stores
    

    BASE_DIR = r"\directory\Astrology Conversational RAG Agent"
    
    print("Loading embedding model...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Loading FAISS indexes...")
    try:
        registry_path = os.path.join(BASE_DIR, "vector_store_registry.pkl")
        with open(registry_path, "rb") as f:
            registry = pickle.load(f)
        
        for category, info in registry.items():
            # Build absolute paths
            index_path = os.path.join(BASE_DIR, info['index_file'])
            metadata_path = os.path.join(BASE_DIR, info['metadata_file'])
            
            index = faiss.read_index(index_path)
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)
            
            vector_stores[category] = {
                'index': index,
                'documents': metadata['documents'],
                'metadata': metadata['metadata']
            }
        
        print(f"✅ Loaded {len(vector_stores)} FAISS indexes:")
        for cat in vector_stores.keys():
            print(f"   - {cat}: {len(vector_stores[cat]['documents'])} docs")
            
    except Exception as e:
        print(f" Error loading indexes: {e}")
        print("Run vector_store_setup.py first!")



def calculate_zodiac_sign(birth_date: str) -> str:
    """Calculate zodiac sign from birth date"""
    date_obj = datetime.strptime(birth_date, '%Y-%m-%d')
    month = date_obj.month
    day = date_obj.day
    
    zodiac_dates = [
        (3, 21, 4, 19, "Aries"), (4, 20, 5, 20, "Taurus"),
        (5, 21, 6, 20, "Gemini"), (6, 21, 7, 22, "Cancer"),
        (7, 23, 8, 22, "Leo"), (8, 23, 9, 22, "Virgo"),
        (9, 23, 10, 22, "Libra"), (10, 23, 11, 21, "Scorpio"),
        (11, 22, 12, 21, "Sagittarius"), (12, 22, 1, 19, "Capricorn"),
        (1, 20, 2, 18, "Aquarius"), (2, 19, 3, 20, "Pisces")
    ]
    
    for start_month, start_day, end_month, end_day, sign in zodiac_dates:
        if (month == start_month and day >= start_day) or \
           (month == end_month and day <= end_day):
            return sign
    return "Unknown"

def detect_language(text: str) -> str:
    """Detect if text is in Hindi"""
   #language setection for now hindi and english only
    hindi_chars = sum(1 for char in text if '\u0900' <= char <= '\u097F')
    return "hi" if hindi_chars > len(text) * 0.3 else "en"

def translate_text(text: str, src_lang: str, dest_lang: str) -> str:
   #Translate text using googletrans
    try:
        if src_lang == dest_lang:
            return text
        result = translator.translate(text, src=src_lang, dest=dest_lang)
        return result.text
    except Exception as e:
        print(f"Translation error: {e}")
        return text 

def check_toxic_content(text: str) -> bool:
#toxic contnt check
    text_lower = text.lower()
    for pattern in TOXIC_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    return False

def detect_query_intent(message: str) -> List[str]:
    message_lower = message.lower()
    
    intent_keywords = {
        'career_guidance': ['career', 'job', 'profession', 'work', 'business', 'employment'],
        'love_guidance': ['love', 'relationship', 'marriage', 'partner', 'romance', 'dating'],
        'spiritual_guidance': ['spiritual', 'meditation', 'karma', 'dharma', 'moksha', 'temple'],
        'planetary_impacts': ['planet', 'mercury', 'venus', 'mars', 'jupiter', 'saturn', 'transit'],
        'nakshatra_mapping': ['nakshatra', 'star', 'lunar mansion'],
        'zodiac_traits': ['trait', 'personality', 'characteristic', 'nature', 'zodiac', 'sign']
    }
    
    relevant_sources = []
    
    for source, keywords in intent_keywords.items():
        if any(keyword in message_lower for keyword in keywords):
            relevant_sources.append(source)
    
    # Always include zodiac traits as base context
    if 'zodiac_traits' not in relevant_sources:
        relevant_sources.append('zodiac_traits')
    
    # If no specific intent detected, use career as default
    if len(relevant_sources) == 1:  # Only zodiac_traits
        relevant_sources.append('career_guidance')
    
    return relevant_sources

def get_complete_zodiac_data(zodiac: str) -> str:
    #Get the complete zodiac traits for a specific sign
    if 'zodiac_traits' not in vector_stores:
        return ""
    
    documents = vector_stores['zodiac_traits']['documents']
    
    # Find the document for this zodiac sign
    for doc in documents:
        if zodiac.lower() in doc.lower():
            return doc
    return ""

def retrieve_smart_context(query: str, zodiac: str, top_k: int = 3) -> tuple:

    global embedding_model, vector_stores
    
    # Detect relevant sources
    relevant_sources = detect_query_intent(query)
    
    # Create query embedding
    enhanced_query = f"{zodiac} {query}"
    query_embedding = embedding_model.encode([enhanced_query])
    query_embedding = np.array(query_embedding).astype('float32')
    
    all_contexts = []
    retrieval_scores = []
    context_used = []
    
    # Retrieve from each relevant source
    for source in relevant_sources:
        if source not in vector_stores:
            continue
        
        store = vector_stores[source]
        index = store['index']
        documents = store['documents']
        
        # Search this index
        distances, indices = index.search(query_embedding, min(top_k, len(documents)))
        
        for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
            if idx < len(documents):
                doc = documents[idx]
                # Convert L2 distance to similarity score (0-1)
                confidence = 1 / (1 + distance)
                
                all_contexts.append(f"[{source.upper()}]\n{doc}")
                retrieval_scores.append(RetrievalScore(
                    rank=len(retrieval_scores) + 1,
                    confidence=round(float(confidence), 4),
                    category=source,
                    distance=round(float(distance), 4)
                ))
                context_used.append(source)
    
    # Get COMPLETE zodiac data for this sign
    zodiac_data = get_complete_zodiac_data(zodiac)
    if zodiac_data:
        all_contexts.insert(0, f"[COMPLETE ZODIAC DATA FOR {zodiac}]\n{zodiac_data}")
    
    combined_context = "\n\n".join(all_contexts)
    return combined_context, list(set(context_used)), retrieval_scores

@lru_cache(maxsize=100)
def get_cached_context(query_hash: str):
    #caching similar replies
    return prompt_cache.get(query_hash)

def query_llm(prompt: str, max_retries: int = 2) -> str:
    
   #add api key
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if openai_key:
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {openai_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4o-mini",  # Fast and cheap
                    "messages": [
                        {"role": "system", "content": "You are an expert Vedic astrologer."},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 400,
                    "temperature": 0.7
                },
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                generated = result['choices'][0]['message']['content'].strip()
                print(f"✅ Using OpenAI (GPT-4o-mini)")
                return generated
            else:
                print(f"OpenAI error: {response.status_code}")
        except Exception as e:
            print(f"OpenAI error: {e}")
    

    return "API key for LLM not found or all LLM services failed."




def build_improved_prompt(user_profile: UserProfile, message: str, 
                         context: str, conversation_history: List[Dict]) -> str:
    
    zodiac = calculate_zodiac_sign(user_profile.birth_date)
    birth_year = datetime.strptime(user_profile.birth_date, '%Y-%m-%d').year
    age = datetime.now().year - birth_year

    history_text = ""
    if conversation_history:
        recent_history = conversation_history[-3:]  # Last 3 exchanges
        for entry in recent_history:
            history_text += f"User: {entry['user']}\nAstrologer: {entry['assistant']}\n\n"
    
    prompt = f"""<s>[INST] You are an expert Vedic astrologer with deep knowledge of Jyotish shastra, nakshatras, and planetary influences. You provide personalized, insightful guidance based on birth charts and cosmic energies.

CLIENT PROFILE:
- Name: {user_profile.name}
- Sun Sign (Rashi): {zodiac}
- Birth Date: {user_profile.birth_date} (Age: {age} years)
- Birth Time: {user_profile.birth_time}
- Birth Place: {user_profile.birth_place}

RELEVANT ASTROLOGICAL KNOWLEDGE:
{context}

PREVIOUS CONVERSATION:
{history_text if history_text else "This is the first interaction."}

CURRENT QUESTION:
{message}

INSTRUCTIONS FOR YOUR RESPONSE:
1. You have been given COMPLETE zodiac data for ALL signs
2. Only use information relevant to {zodiac} - ignore other zodiac signs
3. Address {user_profile.name} personally and warmly
4. Reference their {zodiac} nature and characteristics specifically
5. Mention relevant planetary influences affecting them currently
6. Provide practical, actionable guidance
7. Keep response focused (3-5 sentences maximum)
8. Be empathetic and encouraging while being truthful
9. If discussing challenges, always provide remedial suggestions
10. Do NOT make up information - only use the provided astrological knowledge
11. Do NOT mention other zodiac signs in your response
12. Speak as a wise, compassionate Vedic astrologer would

Provide your {zodiac}-focused astrological insight now: [/INST]</s>"""
    
    return prompt

def validate_and_filter_response(response: str) -> tuple:
   #Validate LLM response and calculate confidence

    response = re.sub(r'<s>|</s>|\[INST\]|\[/INST\]', '', response)
    response = re.sub(r'\s+', ' ', response).strip()
    
    #hallucination indicators
    hallucination_phrases = [
        "i don't have", "i cannot", "i'm not sure", "i don't know",
        "as an ai", "language model", "i apologize"
    ]
    
    response_lower = response.lower()
    hallucination_score = sum(1 for phrase in hallucination_phrases 
                             if phrase in response_lower)
    
    # Check length-too short might indicate error
    length_score = 1.0 if len(response) > 50 else 0.5
    
    # Check if response contains toxic content
    is_toxic = check_toxic_content(response)
    
    # Calculate confidence (0-1)
    confidence = max(0.0, 1.0 - (hallucination_score * 0.2)) * length_score
    
    if is_toxic:
        confidence = 0.0
        response = "I'm unable to provide that type of guidance. Please ask about positive astrological insights."
    
    return response, round(confidence, 4)



@app.get("/")
async def root():
    #Health check endpoint
    return {
        "status": "active",
        "service": "Astro Conversational Insight Agent",
        "version": "2.0.0"
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):

    
    try:
        # Validate input for toxic content
        if check_toxic_content(request.message):
            raise HTTPException(status_code=400, 
                              detail="Message contains inappropriate content")
        
        # Get or create session
        if request.session_id not in session_memory:
            session_memory[request.session_id] = {
                'history': [],
                'user_profile': request.user_profile.dict()
            }
        
        session = session_memory[request.session_id]
        
        # Detect message language
        detected_lang = detect_language(request.message)
        message_english = request.message
        
        # Translate to English if Hindi for better vector search
        if detected_lang == "hi":
            message_english = translate_text(request.message, "hi", "en")
            print(f"🌐 Translated query (Hi→En): {message_english}")
        
        # Calculate zodiac
        zodiac = calculate_zodiac_sign(request.user_profile.birth_date)
        
        # Smart retrieval from multiple indexes
        context, context_used, retrieval_scores = retrieve_smart_context(
            message_english, zodiac, top_k=1
        )
        
        # Build prompt with conversation history
        prompt = build_improved_prompt(
            request.user_profile,
            message_english,
            context,
            session['history']
        )
        

        response = query_llm(prompt)

        response, confidence = validate_and_filter_response(response)
        
        # Translate back to Hindi if user prefers Hindi
        if request.user_profile.preferred_language == "hi":
            response = translate_text(response, "en", "hi")
            print(f"🌐 Translated response (En→Hi)")
        
        # Update session memory with conversation history
        session['history'].append({
            'user': request.message,
            'assistant': response
        })
        
        # Keep only last 10 exchanges to manage memory
        if len(session['history']) > 10:
            session['history'] = session['history'][-10:]
        
        return ChatResponse(
            response=response,
            context_used=context_used,
            zodiac=zodiac,
            confidence_score=confidence,
            retrieval_scores=retrieval_scores
        )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, 
                          detail=f"Internal server error: {str(e)}")

@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """Retrieve session history"""
    if session_id not in session_memory:
        raise HTTPException(status_code=404, detail="Session not found")
    return session_memory[session_id]

@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear session history"""
    if session_id in session_memory:
        del session_memory[session_id]
        return {"message": "Session cleared successfully"}
    raise HTTPException(status_code=404, detail="Session not found")

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "indexes_loaded": len(vector_stores),
        "active_sessions": len(session_memory),
        "llm_backend": "Hugging Face Inference API",
        "features": {
            "multi_index_retrieval": True,
            "hindi_support": True,
            "session_memory": True,
            "confidence_scoring": True,
            "intent_detection": True
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

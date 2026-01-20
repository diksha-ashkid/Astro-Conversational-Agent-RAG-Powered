# Astro-Conversational-Agent-RAG-Powered# Astro Conversational Insight Agent

## Technical Implementation Documentation

## Executive Summary

The **Astro Conversational Insight Agent** is a conversational AI system designed to deliver personalized astrological consultations. It combines modern Natural Language Processing (NLP), vector similarity search, intent detection, and multilingual support to provide accurate, context-aware astrological guidance.

The system is built using **FastAPI** and integrates FAISS-based vector databases, session-aware conversation handling, safety validation layers, and external large language model APIs to ensure high-quality, grounded responses.

---

## System Architecture Overview

The application follows a modular, scalable architecture composed of the following core layers:

* FastAPI-based REST API
* Multi-index FAISS vector database
* Intent detection and smart retrieval engine
* Multilingual processing pipeline (English & Hindi)
* Session and conversation memory management
* Content safety and hallucination prevention layers
* External language model integration

---

## Core Components

## 1. Vector Database Creation and Management

### 1.1 Multi-Index Architecture

Instead of a single monolithic vector database, the system uses a **multi-index FAISS architecture**. Each index corresponds to a distinct astrological knowledge domain, allowing faster and more precise retrieval.

#### Knowledge Domains

Six separate FAISS indexes are created:

* `zodiac_traits` – Personality traits of all zodiac signs
* `career_guidance` – Career and professional insights
* `love_guidance` – Relationships and compatibility
* `spiritual_guidance` – Spiritual growth, karma, meditation
* `planetary_impacts` – Planetary transits and influences
* `nakshatra_mapping` – Nakshatra (lunar mansion) details

#### Processing Pipeline

Each data source follows this pipeline:

1. **Data Extraction**
   JSON and text sources are parsed into structured content.

2. **Text Preprocessing**
   Contextual markers and formatting are applied.

3. **Embedding Generation**
   Uses `sentence-transformers/all-MiniLM-L6-v2` to generate 384-dimensional embeddings.

4. **Index Construction**
   FAISS `IndexFlatL2` indexes are built for similarity search.

5. **Metadata Persistence**
   Documents and metadata are stored using `pickle`.

#### File Structure

For each knowledge domain:

* `faiss_<category>.bin` – FAISS index
* `faiss_<category>_meta.pkl` – Metadata store

Additionally:

* `vector_store_registry.pkl` – Central index registry

#### Advantages

* Reduced search space and faster retrieval
* Domain-specific relevance scoring
* Independent index updates and maintenance
* Flexible weighted multi-source retrieval

---

### 1.2 Embedding Model Selection

The `all-MiniLM-L6-v2` model was chosen due to:

* Compact size (~80 MB)
* Fast inference
* Balanced semantic performance
* Strong community support
* 384-dimensional dense embeddings

---

## 2. Intent Detection and Smart Retrieval

### 2.1 Query Intent Classification

A keyword-based intent detection system identifies relevant knowledge domains for each query.

#### Keyword Mapping

* **career_guidance**: career, job, profession, work, business
* **love_guidance**: love, relationship, marriage, partner
* **spiritual_guidance**: spiritual, meditation, karma, dharma
* **planetary_impacts**: planet, mercury, venus, mars, transit
* **nakshatra_mapping**: nakshatra, star, lunar mansion
* **zodiac_traits**: personality, trait, zodiac, sign

#### Logic Flow

* Query normalized to lowercase
* Keywords matched across domains
* Zodiac traits always included
* Career guidance added if no intent detected

This guarantees zodiac-specific personalization in every response.

---

### 2.2 Multi-Source Context Retrieval

Once intent is identified, the system retrieves information from multiple FAISS indexes in parallel.

#### Retrieval Steps

1. Query enhanced with zodiac sign
2. Embedding generated
3. Parallel FAISS searches executed
4. L2 distance converted to confidence score
   `confidence = 1 / (1 + distance)`
5. Retrieved documents assembled with labels
6. Full zodiac profile prepended to context

#### Configuration

* Top-K per index: 3
* Max context size: ~2000 tokens
* No minimum confidence threshold

---

## 3. Hallucination Prevention and Response Validation

### 3.1 Prompt Engineering Strategy

The system uses structured prompts to ensure grounded responses.

#### Key Techniques

* Explicit instruction to use only provided knowledge
* Zodiac-specific filtering
* Conversation history inclusion
* Response length and format constraints
* Explicit prohibition of fabrication

#### Prompt Structure

1. System role definition (Vedic astrologer)
2. User profile (zodiac, age)
3. Retrieved astrological context
4. Conversation history
5. Current user query
6. Response constraints and instructions

---

### 3.2 Response Validation Layer

Model outputs undergo post-processing validation.

#### Validation Steps

* Instruction artifact removal
* Hallucination phrase detection
* Minimum length check (50 characters)
* Toxic content scan
* Confidence score calculation

#### Confidence Formula

```
base_confidence = 1.0 - (hallucination_count * 0.2)
length_multiplier = 1.0 if length > 50 else 0.5
final_confidence = max(0.0, base_confidence * length_multiplier)
```

Toxic responses are replaced with safe fallback text and assigned a confidence score of `0.0`.

---

## 4. Toxic Content Filtering

### 4.1 Pattern-Based Detection

A two-stage safety mechanism is implemented.

#### Stage 1: Input Validation

User messages are scanned using regex patterns for:

* Violence
* Self-harm
* Illegal activities
* Weapons or terrorism

Violations result in an immediate `HTTP 400`.

#### Stage 2: Output Validation

Model responses are re-scanned:

* Toxic responses are discarded
* Confidence score set to `0.0`
* Safe guidance returned
* Incident logged

---

### 4.2 Design Rationale

* Fast rejection of harmful content
* Defense against prompt injection
* No exposure of internal rules
* Security audit trail

---

## 5. Session Management and Conversation History

### 5.1 In-Memory Session Store

Sessions are stored using a Python dictionary.

#### Schema

```
{
  "session_id": {
    "history": [
      {"user": "...", "assistant": "..."}
    ],
    "user_profile": {...}
  }
}
```

#### Lifecycle

* Created on first request
* Updated on every exchange
* Limited to last 10 messages
* Retrievable via GET
* Deletable via DELETE

---

### 5.2 Conversation Context Integration

* Last 3 exchanges included in prompt
* Enables follow-up questions
* Supports anaphora resolution
* Ensures coherent multi-turn dialogue

---

### 5.3 Memory Management

* Automatic pruning
* No persistence across restarts (privacy-first)
* Client-managed session IDs

---

## 6. Caching Strategy

### 6.1 LRU Cache

Uses Python `lru_cache` for:

* Context retrieval
* Prompt generation

```
@lru_cache(maxsize=100)
def get_cached_context(query_hash):
    ...
```

### 6.2 Benefits

* Faster responses
* Reduced embedding computation
* Lower API usage

Limitations include lack of invalidation and potential hash collisions.

---

## 7. Multi-Language Support

### 7.1 Translation Pipeline

Supports **English and Hindi**.

#### Hindi Flow

1. Unicode-based language detection
2. Translation to English
3. Vector search in English
4. Response generation
5. Translation back to Hindi

#### Implementation

* Library: `googletrans==4.0.0-rc1`
* Free Google Translate API
* Graceful fallback on failure
* Translation logging enabled

---

### 7.2 Language Preference Handling

User profile includes `preferred_language`:

* `en`: Full English processing
* `hi`: English processing, Hindi output

This ensures semantic accuracy with natural user responses.

---

## 8. Language Model Integration

### 8.1 API Architecture

* Model: `GPT-4o-mini`
* Max tokens: 400
* Temperature: 0.7
* Timeout: 15 seconds

#### Error Handling

* Retries with exponential backoff
* Graceful fallback responses
* Full error logging

---

### 8.2 Prompt Construction

Each request includes:

* System role definition
* User profile
* Retrieved context
* Conversation history
* Current question
* Response instructions

---

## 9. Performance Optimizations

### 9.1 Vector Search

* FAISS IndexFlatL2
* Parallel multi-index search
* Asynchronous FastAPI endpoints

### 9.2 Response Time Targets

| Component     | Latency  |
| ------------- | -------- |
| Health check  | < 100 ms |
| Vector search | < 200 ms |
| LLM API       | 2–5 s    |
| Translation   | +500 ms  |
| End-to-end    | 3–7 s    |

---

## 10. Data Flow Architecture

1. Client sends POST `/chat`
2. Request validation (Pydantic)
3. Toxic content scan
4. Session retrieval/creation
5. Language detection
6. Translation if required
7. Zodiac calculation
8. Intent detection
9. FAISS retrieval
10. Prompt construction
11. LLM invocation
12. Response validation
13. Translation (if needed)
14. Confidence calculation
15. Session update
16. Response returned

---

## 11. Error Handling Strategy

* **422**: Validation errors
* **400**: Content policy violations
* **404**: Session not found
* **500**: Internal server errors

Errors are logged server-side with safe client messaging.

---

## 12. Testing and Validation

### Unit Testing

* Zodiac calculation
* Intent detection
* Validation logic
* FAISS retrieval (mocked)

### Integration Testing

* End-to-end API flow
* Multi-language support
* Session lifecycle

### Load Testing

* Concurrent requests
* Memory usage
* API rate limits

---

## 13. Deployment Requirements

### Dependencies

* Python 3.8+
* FastAPI
* Pydantic
* FAISS-CPU
* sentence-transformers
* googletrans==4.0.0-rc1
* numpy
* requests
* boto3 (optional)

### Environment Variables

* `OPENAI_API_KEY`

### Pre-Deployment Steps

1. Run `vector_store_setup.py`
2. Verify all 6 FAISS indexes
3. Set environment variables
4. Start server using `uvicorn`
5. Run health check


## Conclusion

The **Astro Conversational Agent** is a production-ready AI system that combines astrological knowledge with modern NLP engineering best practices. Its multi-index vector architecture, robust safety mechanisms, multilingual support, and session-aware design ensure accurate, reliable, and user-safe consultations.

The modular architecture enables easy scalability and future enhancements, providing a strong foundation for real-world deployment.


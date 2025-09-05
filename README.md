# Memory-Enhanced AI System

A sophisticated conversational AI system with Neo4j graph memory and intelligent retrieval augmentation.

## Quick Setup

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Set up Environment Variables

1. Create a `.env` file in the root directory:
```
GEMINI_API_KEY=your_google_gemini_api_key_here
NEO4J_URI=neo4j+s://your-neo4j-instance.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password
```

2. Get your Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
3. Set up a Neo4j AuraDB instance at [Neo4j Aura](https://console.neo4j.io/)

### Step 3: Test Database Setup

You can test your setup by running the main application:
```bash
streamlit run app.py
```

This will:
- Test Neo4j graph database connection
- Verify embedding service functionality
- Test Gemini API connection
- Validate the complete system

### Step 4: Launch the Application

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## Database Configuration

### Neo4j Graph Database
- **Production**: Use Neo4j AuraDB (cloud) for best performance
- **Local**: Install Neo4j Desktop for local development  
- The system uses vector indexing, fulltext indexing, and graph traversal

## Architecture

### Prerequisites
- Python 3.8+
- Google Gemini API key
- Neo4j database (AuraDB recommended)

### Quick Setup

1. **Clone and setup**:
   ```bash
   git clone <your-repo-url>
   cd memory
   ```

2. **Configure environment**:
   - Create `.env` file with required API keys and database credentials
   - See environment setup section above for details

3. **Start the application**:
   ```bash
   # Create virtual environment
   python -m venv venv
   venv\Scripts\activate     # Windows
   source venv/bin/activate  # Unix/Linux/MacOS
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Run the application
   streamlit run app.py
   ```

## 🚀 Usage

### Basic Conversation
1. Open the Streamlit interface (usually at http://localhost:8501)
2. Start chatting! The system will automatically:
   - Remember your conversation
   - Search relevant memories when needed
   - Provide contextual responses

### Memory Management
- **🗑️ Clear Chat**: Removes current session but keeps stored memories
- **🧹 Clear Memory**: Completely wipes all stored memories for the conversation
- **🆕 New Conversation**: Starts fresh but can still access previous memories

### Analytics Features
- **� Basic Analytics**: View conversation statistics and activity patterns
- **� Activity Charts**: Visual representation of conversation trends
- **� Trending Topics**: See most discussed topics from your conversations
- **Conversation History**: Browse, search, and analyze past conversations
- **Data Management**: Export conversations for backup or analysis
- **Topic Analytics**: Track conversation themes and trending topics
- **Timeline Views**: Explore conversations across different time periods

## 📁 Project Structure

```
memory/
├── src/
│   ├── components/          # Core system components
│   │   ├── gemini_service.py       # Gemini LLM integration
│   │   ├── memory_orchestrator.py  # LangGraph orchestration
│   │   ├── qdrant_manager.py       # Vector database management
│   │   ├── retrieval_fusion.py     # Hybrid search fusion
│   │   └── whoosh_manager.py       # Lexical search management
│   ├── utils/              # Utility modules
│   │   ├── embedding_service.py    # Text embedding generation
│   │   └── text_chunker.py         # Text chunking logic
│   └── config.py           # Configuration management
├── data/                   # Data storage directory
├── .env                   # Environment variables
├── app.py                 # Streamlit UI application (main entry point)
├── run.bat                # Windows startup script
├── run.sh                 # Unix/Linux startup script
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## ⚙️ Configuration

### Environment Variables (.env)
```env
# Required
GEMINI_API_KEY=your_gemini_api_key

# Optional (defaults shown)
QDRANT_HOST=localhost
QDRANT_PORT=6333
WHOOSH_INDEX_DIR=data/whoosh_index
```

### System Parameters (src/config.py)
- **Chunk Size**: 300-500 tokens with 10-20% overlap
- **Memory Buffer**: Last 8-10 conversation turns
- **Retrieval**: Top 20 semantic + Top 20 lexical → Top 5-10 final results
- **Models**: Gemini 1.5 Pro for generation, Flash for compression

## 🔍 How It Works

### 1. Query Processing
1. **User Input**: User types a query in the Streamlit interface
2. **Analysis**: LangGraph agent analyzes if memory retrieval is needed
3. **Transformation**: Gemini rewrites vague queries into clear, searchable forms

### 2. Memory Retrieval
1. **Semantic Search**: Qdrant finds conceptually similar conversations
2. **Lexical Search**: Whoosh finds keyword matches using BM25
3. **Fusion**: RRF algorithm combines and ranks results

### 3. Response Generation
1. **Compression**: Gemini Flash summarizes retrieved context
2. **Generation**: Gemini Pro creates final response using context + short-term memory
3. **Storage**: New conversation turn is chunked and stored for future retrieval

### 4. Memory Management
- **Short-term**: Last 8-10 turns kept in session state
- **Long-term**: All conversations chunked, embedded, and stored persistently
- **Cross-session**: Memories survive browser refresh and new sessions

## 🛠️ Development

### Running Tests
```bash
### Running Tests
```bash
# Test individual components
python -m src.utils.text_chunker
python -m src.components.neo4j_manager
python -m src.components.conversation_history_manager
```

### Adding New Features
1. **New Retrieval Method**: Extend `graph_retrieval.py`
2. **Different LLM**: Modify `gemini_service.py` 
3. **UI Enhancements**: Update `app.py`
4. **Database Schema**: Modify Neo4j constraints and indexes

## 🔧 Troubleshooting

### Common Issues

**"Import errors"**
- Run `pip install -r requirements.txt`
- Ensure virtual environment is activated

**"Neo4j connection failed"**
- Check database URI and credentials in `.env`
- Verify Neo4j instance is running
- Test connection by launching the Streamlit app

**"Gemini API errors"**
- Verify API key in `.env` file
- Check quota limits at https://aistudio.google.com/

**"No memories found"**
- Have a few conversations first to build memory
- Check system status in analytics for database statistics

### Performance Tips
- **Large Conversations**: System automatically manages memory chunks
- **Slow Responses**: Reduce retrieval limit values in search functions
- **Memory Usage**: Clear old conversations using the analytics panel

## 📊 System Metrics

The system tracks various metrics visible in the analytics panel:
- **Memory Storage**: Number of messages and conversations stored
- **Daily Activity**: Message count trends over time
- **Trending Topics**: Most discussed conversation topics
- **Processing Performance**: Response time and system health

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Make changes and test thoroughly
4. Submit pull request with detailed description

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Google Gemini**: For powerful LLM capabilities
- **Neo4j**: For robust graph database and vector indexing
- **LangGraph**: For conversation orchestration
- **Streamlit**: For beautiful and interactive UI
- **Sentence Transformers**: For high-quality text embeddings

## 📞 Support

For questions or issues:
1. Check the troubleshooting section above
2. Review system status in the analytics panel
3. Open an issue on GitHub with detailed information

---

**Happy conversing with your memory-enhanced AI! 🧠✨**
*Advanced graph-based memory architecture for human-like conversations*

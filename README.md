# Memory-Enhanced AI System

A sophisticated 8-layer conversational AI system with vector memory and retrieval augmentation.

## Quick Setup

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Set up Environment Variables

1. Create a `.env` file in the root directory:
```
GEMINI_API_KEY=your_google_gemini_api_key_here
```

2. Get your Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)

### Step 3: Test Database Setup

Run the setup script to test all components:
```bash
python setup_databases.py
```

This will:
- Test Qdrant vector database (uses in-memory mode by default)
- Test Whoosh search engine (creates local index files)
- Verify Gemini API connection
- Test the complete orchestrator

### Step 4: Launch the Application

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## Database Configuration

### Qdrant Vector Database
- **Default**: In-memory mode (no setup required)
- **Production**: Install Docker and run `docker run -p 6333:6333 qdrant/qdrant`
- The system automatically falls back to in-memory if server unavailable

### Whoosh Search Engine
- **Automatic**: Creates `whoosh_index/` directory automatically
- **No setup required**: File-based search works out of the box

## Architecture

### Prerequisites
- Python 3.8+
- Docker (for Qdrant)
- Google Gemini API key

### Quick Setup

1. **Clone and setup**:
   ```bash
   git clone <your-repo-url>
   cd memory
   python setup.py
   ```

2. **Configure API key**:
   - Edit `.env` file
   - Add your Gemini API key: `GEMINI_API_KEY=your_key_here`
   - Get API key from: https://makersuite.google.com/app/apikey

3. **Start the application**:
   ```bash
   # Option 1: Use the convenient startup script
   run.bat           # Windows
   ./run.sh          # Unix/Linux/MacOS
   
   # Option 2: Manual startup
   venv\Scripts\activate     # Windows
   source venv/bin/activate  # Unix/Linux/MacOS
   streamlit run app.py
   ```

### Manual Installation

1. **Create virtual environment**:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # Unix/Linux/MacOS
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start Qdrant**:
   ```bash
   docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
   ```

4. **Configure environment**:
   ```bash
   # Create .env file with your Gemini API key
   echo "GEMINI_API_KEY=your_api_key_here" > .env
   ```

5. **Run the application**:
   ```bash
   # Option 1: Use startup script
   run.bat           # Windows
   ./run.sh          # Unix/Linux/MacOS
   
   # Option 2: Direct command
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

### Advanced Features
- **Debug Mode**: Toggle to see retrieval details and processing information
- **Retrieved Context**: View the specific memory chunks used for each response
- **System Status**: Monitor database connections and memory statistics

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
# Test individual components
python -m src.utils.text_chunker
python -m src.components.qdrant_manager
python -m src.components.whoosh_manager
```

### Adding New Features
1. **New Retrieval Method**: Extend `retrieval_fusion.py`
2. **Different LLM**: Modify `gemini_service.py`
3. **UI Enhancements**: Update `app.py`
4. **Configuration**: Add settings to `config.py`

### Debugging
- Enable debug mode in the UI to see:
  - Retrieved context chunks
  - Processing metadata
  - System status information
  - Error details

## 🔧 Troubleshooting

### Common Issues

**"Import errors"**
- Run `pip install -r requirements.txt`
- Ensure virtual environment is activated

**"Qdrant connection failed"**
- Check if Docker is running: `docker ps`
- Start Qdrant: `docker run -d --name qdrant -p 6333:6333 qdrant/qdrant`

**"Gemini API errors"**
- Verify API key in `.env` file
- Check quota limits at https://makersuite.google.com/

**"No memories found"**
- Have a few conversations first to build memory
- Check system status in sidebar for database statistics

### Performance Tips
- **Large Conversations**: System automatically manages memory chunks
- **Slow Responses**: Reduce retrieval top_k values in config
- **Memory Usage**: Clear old conversations periodically

## 📊 System Metrics

The system tracks various metrics visible in the debug panel:
- **Memory Storage**: Number of chunks stored per conversation
- **Retrieval Performance**: Semantic vs lexical match rates
- **Processing Time**: Time spent in each system component
- **Error Rates**: Failed operations and retry statistics

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Make changes and test thoroughly
4. Submit pull request with detailed description

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Google Gemini**: For powerful LLM capabilities
- **Qdrant**: For efficient vector database operations
- **Whoosh**: For lexical search functionality
- **LangGraph**: For conversation orchestration
- **Streamlit**: For beautiful and interactive UI

## 📞 Support

For questions or issues:
1. Check the troubleshooting section above
2. Review system status in the UI debug panel
3. Open an issue on GitHub with detailed information

---

**Happy conversing with your memory-enhanced AI! 🧠✨**
refined and advanced memory architecture for human like conversations

# LawBuddy 🤖⚖️
A powerful Thai legal assistant with RAG techniques.

🚀 Installation
- Create an .env file with the following content:
    ```bash
    OPENAI_API_KEY=YOUR_OPENAI_API_KEY
    ```
- Install the package using pip:
    ```bash
    git clone https://github.com/BetterACS/LawBuddy
    cd LawBuddy
    pip install -e .
    ```

💡 Quick Start
Using OpenAI Model
```python
from lawbuddy.rag import IterativeQueryChunking
pipeline = IterativeQueryChunking.from_openai_model(model="gpt-3.5-turbo")

pipeline.load_vector_store(path="spaces/iterative_query_chunking")

query = "โดนโกง 300 ล้านบาทไทย แต่คนโกงไม่โดนฟ้องควรทำยังไง"
response = pipeline.query(query)
```

📚 Vector Store Management
- Creating a New Vector Store
    ```python
    # Create vector store from CSV files
    pipeline.create_vector_store(
        csv_paths=["laws.csv"],
        save_dir="spaces/iterative_query_chunking"
    )
    ```

- Loading Existing Vector Store
    ```python
    pipeline.load_vector_store(path="spaces/iterative_query_chunking")
    ```

🤖 Load local model
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from lawbuddy.rag import IterativeQueryChunking
# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("openthaigpt/openthaigpt1.5-7b-instruct")
tokenizer = AutoTokenizer.from_pretrained("openthaigpt/openthaigpt1.5-7b-instruct")

# Load specialized legal adapter
model.load_adapter("betteracs/lawbuddy-7b")

# Initialize pipeline with local model
pipeline = IterativeQueryChunking.from_local_model(
    model_name="openthaigpt/openthaigpt1.5-7b-instruct",
    model=model
)
```

🔧 Advanced Configuration
The system supports various configurations for both OpenAI and local models. You can customize:

Chunk sizes for document processing
Vector store parameters
Model-specific settings
Query processing parameters

🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.
📝 License
MIT License
📬 Contact
For support or queries, please open an issue in the GitHub repository.

Made with ❤️ for the LawBuddy team.

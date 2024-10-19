from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from dotenv import load_dotenv
from llama_index.readers.file import PagedCSVReader

from llama_index.llms.huggingface import HuggingFaceLLM
from transformers import BitsAndBytesConfig
import torch
from llama_index.core import StorageContext, load_index_from_storage, Response
from llama_index.llms.openai import OpenAI

class BasePipeline:
    SYSTEM_PROMPT = "คุณคือผู้ช่วยตอบคำถามที่ฉลาดและซื่อสัตย์"

    @staticmethod
    def messages_to_prompt(messages):
        raise NotImplementedError
    
    @staticmethod
    def completion_to_prompt(completion):
        return f'<|im_start|>system\n{BasePipeline.SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{completion}<|im_end|>\n<|im_start|>assistant\n'

    def __init__(self, model):
        self.llm = model

    @classmethod
    def from_local_model(cls, model_name: str, model, quantization_config=None, **kwargs):
        if quantization_config is None:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        
        llm = HuggingFaceLLM(
            model_name=model_name,
            tokenizer_name=model_name,
            model=model,
            # context_window=3900,
            max_new_tokens=2048,
            model_kwargs={"quantization_config": quantization_config},
            # generate_kwargs={"temperature": 0.7, "top_k": 50, "top_p": 0.95},
            messages_to_prompt=BasePipeline.messages_to_prompt,
            completion_to_prompt=BasePipeline.completion_to_prompt,
            device_map="auto",
        )
        return cls(model=llm, **kwargs)

    @classmethod
    def from_openai_model(cls, **kwargs):
        return cls(model=OpenAI(**kwargs))

    def create_vector_store(self, csv_paths: str, save_dir: str = "./index"):
        load_dotenv()
        csv_reader = PagedCSVReader()
        reader = SimpleDirectoryReader( 
            input_files=csv_paths,
            file_extractor= {".csv": csv_reader}
        )

        self.docs = reader.load_data()
        self.vector_store_index = VectorStoreIndex.from_documents(self.docs)
        self.vector_store_index.storage_context.persist(save_dir)

        self.query_engine = self.vector_store_index.as_query_engine(similarity_top_k=3, llm=self.llm)

    def load_vector_store(self, path: str = "./index"):
        storage_context = StorageContext.from_defaults(persist_dir=path)
        self.vector_store_index = load_index_from_storage(storage_context)

        self.query_engine = self.vector_store_index.as_query_engine(similarity_top_k=3, llm=self.llm)

    def query(self, query: str) -> Response:
        streaming_response = self.query_engine.query(query)
        return streaming_response

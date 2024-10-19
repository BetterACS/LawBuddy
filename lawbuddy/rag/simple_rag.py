from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from dotenv import load_dotenv
from llama_index.readers.file import PagedCSVReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from lawbuddy.rag.base_pipeline import BasePipeline

class SimpleRagPipeline(BasePipeline):
    VECTOR_SPACE_PATH = "spaces/simple_rag"
    SYSTEM_PROMPT = "คุณคือนักกฏหมายที่เข้าใจกฏหมายเป็นอย่างดี"

    @staticmethod
    def completion_to_prompt(completion):
        return f'<|im_start|>system\n{BasePipeline.SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{completion}<|im_end|>\n<|im_start|>assistant\n'

    def __init__(self, model_name: str ="openthaigpt/openthaigpt1.5-7b-instruct" , quantization_config=None, **kwargs):
        super().__init__(model_name, quantization_config, **kwargs)

    def create_vector_store(self, csv_paths: str, save_dir: str = VECTOR_SPACE_PATH):
        load_dotenv()
        csv_reader = PagedCSVReader()
        reader = SimpleDirectoryReader( 
            input_files=csv_paths,
            file_extractor= {".csv": csv_reader}
        )

        self.docs = reader.load_data()
        self.pipeline = IngestionPipeline(transformations=[SentenceSplitter(chunk_size=256, chunk_overlap=64)])
        self.nodes = self.pipeline.run(documents=self.docs, show_progress=True)
        self.vector_store_index = VectorStoreIndex(self.nodes)
        self.vector_store_index.storage_context.persist(save_dir)

        self.query_engine = self.vector_store_index.as_query_engine(similarity_top_k=3, llm=self.llm)

    def load_vector_store(self, path: str = VECTOR_SPACE_PATH):
        super().load_vector_store(path)

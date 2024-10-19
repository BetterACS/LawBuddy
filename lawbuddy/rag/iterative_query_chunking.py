from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Response
from dotenv import load_dotenv
from llama_index.readers.file import PagedCSVReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from lawbuddy.rag.base_pipeline import BasePipeline
from llama_index.core import QueryBundle, Document
from llama_index.core.prompts import PromptTemplate
from llama_index.core.postprocessor import LLMRerank

class IterativeQueryChunking(BasePipeline):
    VECTOR_SPACE_PATH = "spaces/iterative_query_chunking"
    SYSTEM_PROMPT = PromptTemplate(
        "นี่คือเนื้อหา\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "จงใช้เนื้อหาต่อไปนี้ในการตอบคำถาม\n"
        "Question: {question}\n"
        "Answer: "
    )

    def __init__(self, model):
        self.reranker = LLMRerank(choice_batch_size=5, top_n=5)
        super().__init__(model)
    
    def create_vector_store(self, csv_paths: str, save_dir: str = VECTOR_SPACE_PATH):
        load_dotenv()
        csv_reader = PagedCSVReader()
        reader = SimpleDirectoryReader( 
            input_files=csv_paths,
            file_extractor= {".csv": csv_reader}
        )

        self.docs = reader.load_data()
        self.pipeline = IngestionPipeline()
        self.nodes = self.pipeline.run(documents=self.docs, show_progress=True)
        self.vector_store_index = VectorStoreIndex(self.nodes)
        self.vector_store_index.storage_context.persist(save_dir)

        self.retriever = self.vector_store_index.as_retriever(similarity_top_k=3)

    def load_vector_store(self, path: str = VECTOR_SPACE_PATH):
        super().load_vector_store(path)
        self.retriever = self.vector_store_index.as_retriever(similarity_top_k=3)


    def query(self, query: str, chunk_size=512, chunk_overlap=128) -> Response:        
        nodes = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap).get_nodes_from_documents(
            [Document(text=query)])
        
        retrieved_doc = []
        for sentence in nodes:
            retrieved_doc.extend(self.retriever.retrieve(sentence.text))

        query_bundle = QueryBundle(query)
        retrieved_nodes = self.reranker.postprocess_nodes(retrieved_doc, query_bundle)

        retrieved_contents = set([n.node.get_content() for n in retrieved_nodes])
        context_str = "\n\n".join(list(retrieved_contents))

        response = self.llm.complete(self.SYSTEM_PROMPT.format(context_str=context_str, question=query))
        return response


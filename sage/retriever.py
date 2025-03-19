import logging
import os
from typing import Dict, List, Optional

import Levenshtein
from anytree import Node, RenderTree
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.schema import BaseRetriever, Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_voyageai import VoyageAIEmbeddings
from pydantic import Field

from sage.code_symbols import get_code_symbols
from sage.data_manager import DataManager, GitHubRepoManager
from sage.llm import build_llm_via_langchain
from sage.reranker import build_reranker
from sage.vector_store import build_vector_store_from_args

from ollama import chat
from ollama import ChatResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Use a local LLM through Ollama.
LOCAL_MODEL_CONTEXT_SIZE = 32768  # Adjust as appropriate for your model

def call_ollama_chat(model: str, messages: List[Dict]) -> ChatResponse:
    """
    Calls the local model via the Ollama Python library.
    `messages` is a list of message dicts, e.g. [{'role': 'user', 'content': '...'}].
    """
    response: ChatResponse = chat(model=model, messages=messages)
    return response

class LLMRetriever(BaseRetriever):
    """
    Custom Langchain retriever based on an LLM.
    
    Builds a representation of the folder structure of the repo, feeds it to a local LLM via Ollama,
    and asks the LLM for the most relevant files for a particular user query—making decisions based solely on file names.
    """

    repo_manager: GitHubRepoManager = Field(...)
    top_k: int = Field(...)
    local_model: str = ""

    cached_repo_metadata: List[Dict] = Field(...)
    cached_repo_files: List[str] = Field(...)
    cached_repo_hierarchy: str = Field(...)

    def __init__(self, repo_manager: GitHubRepoManager, top_k: int, local_model: str):
        super().__init__()
        self.repo_manager = repo_manager
        self.top_k = top_k
        self.local_model = local_model

        # Manually cache these fields.
        self.cached_repo_metadata = None
        self.cached_repo_files = None
        self.cached_repo_hierarchy = None

    @property
    def repo_metadata(self):
        if not self.cached_repo_metadata:
            self.cached_repo_metadata = [metadata for metadata in self.repo_manager.walk(get_content=False)]
            # For small codebases, add code symbols.
            small_codebase = len(self.repo_files) <= 200
            if small_codebase:
                for metadata in self.cached_repo_metadata:
                    file_path = metadata["file_path"]
                    content = self.repo_manager.read_file(file_path)
                    metadata["code_symbols"] = get_code_symbols(file_path, content)
        return self.cached_repo_metadata

    @property
    def repo_files(self):
        if not self.cached_repo_files:
            self.cached_repo_files = set(metadata["file_path"] for metadata in self.repo_metadata)
        return self.cached_repo_files

    @property
    def repo_hierarchy(self):
        """
        Produces a string describing the structure of the repository.
        May include class and method names.
        """
        if self.cached_repo_hierarchy is None:
            render = LLMRetriever._render_file_hierarchy(self.repo_metadata, include_classes=True, include_methods=True)
            # Reserve some tokens for the rest of the prompt.
            max_tokens = LOCAL_MODEL_CONTEXT_SIZE - 500

            def count_tokens(text: str) -> int:
                # Simple approximation: count words as tokens.
                return len(text.split())

            if count_tokens(render) > max_tokens:
                logging.info("File hierarchy is too large; excluding methods.")
                render = LLMRetriever._render_file_hierarchy(self.repo_metadata, include_classes=True, include_methods=False)
                if count_tokens(render) > max_tokens:
                    logging.info("File hierarchy is still too large; excluding classes.")
                    render = LLMRetriever._render_file_hierarchy(self.repo_metadata, include_classes=False, include_methods=False)
                    if count_tokens(render) > max_tokens:
                        logging.info("File hierarchy is still too large; truncating.")
                        tokens = render.split()[:max_tokens]
                        render = " ".join(tokens)
            self.cached_repo_hierarchy = render
        return self.cached_repo_hierarchy

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        filenames = self._ask_llm_to_retrieve(user_query=query, top_k=self.top_k)
        documents = []
        for filename in filenames:
            document = Document(
                page_content=self.repo_manager.read_file(filename),
                metadata={"file_path": filename, "url": self.repo_manager.url_for_file(filename)},
            )
            documents.append(document)
        return documents

    def _ask_llm_to_retrieve(self, user_query: str, top_k: int) -> List[str]:
        repo_hierarchy = str(self.repo_hierarchy)
        sys_prompt = f"""
You are a retriever system. You will be given a user query and a list of files in a GitHub repository, along with the class names in each file.

For example:
folder1
    folder2
        folder3
            file123.py
                ClassName1
                ClassName2
                ClassName3
indicates there is a file at folder1/folder2/folder3/file123.py containing classes ClassName1, ClassName2, and ClassName3.

Your task is to determine the top {top_k} files most relevant to the user query.
DO NOT RESPOND TO THE USER QUERY DIRECTLY. Instead, respond with full paths to the files that might contain the answer.
Say nothing else.

Here is the file hierarchy of the repository:

{repo_hierarchy}
"""
        augmented_user_query = f"""
User query: {user_query}

DO NOT RESPOND TO THE USER QUERY DIRECTLY. Instead, respond with full paths to relevant files.
Say nothing else.
"""
        response = LLMRetriever._call_via_ollama_with_prompt(sys_prompt, augmented_user_query, self.local_model)
        # Access the response via the Ollama API.
        files_from_llm = response.message.content.strip().split("\n")
        validated_files = []
        for filename in files_from_llm:
            if filename not in self.repo_files:
                if "/" not in filename:
                    continue
                filename = LLMRetriever._fix_filename(filename, self.repo_manager.repo_id)
                if filename not in self.repo_files:
                    filename = LLMRetriever._find_closest_filename(filename, self.repo_files)
            if filename in self.repo_files:
                validated_files.append(filename)
        return validated_files

    @staticmethod
    def _call_via_ollama_with_prompt(system_prompt: str, user_prompt: str, local_model: str) -> ChatResponse:
        """
        Combines the system prompt and user query, calls the local LLM via the Ollama Python library,
        and returns the response.
        """
        combined_prompt = f"{system_prompt}\n{user_prompt}"
        messages = [{"role": "user", "content": combined_prompt}]
        response = call_ollama_chat(model=local_model, messages=messages)
        logging.info("Ollama response: %s", response.message.content)
        return response

    @staticmethod
    def _render_file_hierarchy(
        repo_metadata: List[Dict], include_classes: bool = True, include_methods: bool = True
    ) -> str:
        """
        Produces a visualization of the file hierarchy from repo metadata.
        Optionally includes class and method names.
        """
        nodepath_to_node = {}

        for metadata in repo_metadata:
            path = metadata["file_path"]
            paths = [path]
            if include_classes or include_methods:
                for class_name, method_name in metadata.get("code_symbols", []):
                    if include_classes and class_name:
                        paths.append(f"{path}/{class_name}")
                    if include_methods and method_name and not method_name.startswith("_"):
                        if class_name:
                            paths.append(f"{path}/{class_name}/{method_name}")
                        else:
                            paths.append(f"{path}/{method_name}")
            for p in paths:
                items = p.split("/")
                nodepath = ""
                parent_node = None
                for item in items:
                    nodepath = f"{nodepath}/{item}"
                    if nodepath in nodepath_to_node:
                        node = nodepath_to_node[nodepath]
                    else:
                        node = Node(item, parent=parent_node)
                        nodepath_to_node[nodepath] = node
                    parent_node = node

        root_path = "/" + repo_metadata[0]["file_path"].split("/")[0]
        full_render = ""
        root_node = nodepath_to_node[root_path]
        for pre, fill, node in RenderTree(root_node):
            render = f"{pre}{node.name}\n"
            render = render.replace("└", " ").replace("├", " ").replace("│", " ").replace("─", " ")
            full_render += render
        return full_render

    @staticmethod
    def _fix_filename(filename: str, repo_id: str) -> str:
        """
        Attempts to fix a filename output by the LLM by adding missing org/repo prefixes.
        """
        if filename.startswith("/"):
            filename = filename[1:]
        org_name, repo_name = repo_id.split("/")
        items = filename.split("/")
        if filename.startswith(org_name) and not filename.startswith(repo_id):
            new_items = [org_name, repo_name] + items[1:]
            return "/".join(new_items)
        if not filename.startswith(org_name) and filename.startswith(repo_name):
            return f"{org_name}/{filename}"
        if not filename.startswith(org_name) and not filename.startswith(repo_name):
            return f"{org_name}/{repo_name}/{filename}"
        return filename

    @staticmethod
    def _find_closest_filename(filename: str, repo_filenames: List[str], max_edit_distance: int = 10) -> Optional[str]:
        """
        Returns the repo path with the smallest edit distance from `filename`.
        Returns None if the closest path exceeds max_edit_distance.
        """
        distances = [(path, Levenshtein.distance(filename, path)) for path in repo_filenames]
        distances.sort(key=lambda x: x[1])
        if distances[0][1] <= max_edit_distance:
            return distances[0][0]
        return None

class RerankerWithErrorHandling(BaseRetriever):
    """
    Wraps a ContextualCompressionRetriever to catch errors during inference.
    If an error occurs, returns the documents in the original order.
    """

    def __init__(self, reranker: ContextualCompressionRetriever):
        self.reranker = reranker

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        try:
            return self.reranker._get_relevant_documents(query, run_manager=run_manager)
        except Exception as e:
            logging.error(f"Error in reranker; preserving original document order. {e}")
            return self.reranker.base_retriever._get_relevant_documents(query, run_manager=run_manager)

def build_retriever_from_args(args, data_manager: Optional[DataManager] = None):
    """
    Builds a retriever (with optional reranking) from command-line arguments.
    """
    if args.llm_retriever:
        print("LOCAL_MODEL =", args.llm_model)
        retriever = LLMRetriever(GitHubRepoManager.from_args(args), top_k=args.retriever_top_k, local_model=args.llm_model)
    else:
        if args.embedding_provider == "openai":
            embeddings = OpenAIEmbeddings(model=args.embedding_model)
        elif args.embedding_provider == "voyage":
            embeddings = VoyageAIEmbeddings(model=args.embedding_model)
        elif args.embedding_provider == "gemini":
            embeddings = GoogleGenerativeAIEmbeddings(model=args.embedding_model)
        else:
            embeddings = None

        retriever = build_vector_store_from_args(args, data_manager).as_retriever(
            top_k=args.retriever_top_k, embeddings=embeddings, namespace=args.index_namespace
        )

    if args.multi_query_retriever:
        retriever = MultiQueryRetriever.from_llm(
            retriever=retriever, llm=build_llm_via_langchain(args.llm_provider, args.llm_model)
        )

    reranker = build_reranker(args.reranker_provider, args.reranker_model, args.reranker_top_k)
    if reranker:
        retriever = ContextualCompressionRetriever(base_compressor=reranker, base_retriever=retriever)
    return retriever

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import asyncio

from models.models import (
    Document,
    DocumentChunk,
    DocumentMetadata,
    DocumentMetadataFilter,
    Query,
    QueryResult,
    QueryWithEmbedding,
)
from services.chunks import get_document_chunks
from services.openai import get_embeddings


class DataStore(ABC):
    async def upsert(
        self, documents: List[Document], chunk_token_size: Optional[int] = None
    ) -> List[str]:
        """
        Takes in a list of documents and inserts them into the database.
        First deletes all the existing vectors with the document id (if necessary, depends on the vector db), then inserts the new ones.
        Return a list of document ids.
        """
        # Delete any existing vectors for documents with the input document ids
        await asyncio.gather(
            *[
                self.delete(
                    filter=DocumentMetadataFilter(
                        document_id=document.id,
                    ),
                    delete_all=False,
                )
                for document in documents
                if document.id
            ]
        )

        chunks = get_document_chunks(documents, chunk_token_size)

        return await self._upsert(chunks)

    @abstractmethod
    async def _upsert(self, chunks: Dict[str, List[DocumentChunk]]) -> List[str]:
        """
        Takes in a list of document chunks and inserts them into the database.
        Return a list of document ids.
        """

        raise NotImplementedError

    async def query(self, queries: List[Query]) -> List[QueryResult]:
        """
        Takes in a list of queries and filters and returns a list of query results with matching document chunks and scores.
        """
        # get a list of just the queries from the Query list
        query_texts = [query.query for query in queries]
        query_embeddings = get_embeddings(query_texts)
        # hydrate the queries with embeddings
        queries_with_embeddings = [
            QueryWithEmbedding(**query.dict(), embedding=embedding)
            for query, embedding in zip(queries, query_embeddings)
        ]
        return await self._query(queries_with_embeddings)

    @abstractmethod
    async def _query(self, queries: List[QueryWithEmbedding]) -> List[QueryResult]:
        """
        Takes in a list of queries with embeddings and filters and returns a list of query results with matching document chunks and scores.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete(
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[DocumentMetadataFilter] = None,
        delete_all: Optional[bool] = None,
    ) -> bool:
        """
        Removes vectors by ids, filter, or everything in the datastore.
        Multiple parameters can be used at once.
        Returns whether the operation was successful.
        """
        raise NotImplementedError

    async def update_text(self, document_id: str, new_text: str) -> List[str]:
        """
        Updates the text content of an existing document while preserving metadata.
        Returns a list of updated document chunk ids.
        """
        # Query for existing document chunks to get metadata
        query = Query(
            query="", 
            filter=DocumentMetadataFilter(document_id=document_id),
            top_k=1000  # Get all chunks for this document
        )
        result = await self.query([query])
        
        if not result or not result[0].results:
            raise ValueError(f"Document with ID {document_id} not found")
        
        # Extract metadata from the first chunk (should be consistent across chunks)
        first_chunk = result[0].results[0]
        metadata = first_chunk.metadata
        
        # Create new document with updated text and preserved metadata
        updated_document = Document(
            id=document_id,
            text=new_text,
            metadata=DocumentMetadata(
                source=metadata.source,
                source_id=metadata.source_id,
                url=metadata.url,
                created_at=metadata.created_at,
                author=metadata.author,
            )
        )
        
        # Use upsert to replace the document (it will delete old chunks and create new ones)
        return await self.upsert([updated_document])

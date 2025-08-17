import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing_extensions import List, TypedDict
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import dotenv
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import re
from datetime import datetime, timedelta

dotenv.load_dotenv()


class MedicalRetriever:
    def __init__(self, vectordb: Chroma, base_k: int = 5):
        self.vectordb = vectordb
        self.base_k = base_k
    
    def retrieve_with_fallback(self, query: str, k: int = None) -> List[Document]:
        """
        Retrieve documents with fallback
        """
        k = k if k is not None else self.base_k

        docs = self.vectordb.similarity_search(query, k=k)

        if len(docs) < k // 2:
            # Extract keywords from the query
            keywords = self._extract_keywords(query)
            if keywords:
                keyword_query = " ".join(keywords)
                additional_docs = self.vectordb.similarity_search(keyword_query, k=k-len(docs))
                docs.extend(additional_docs)
        if len(docs) >= k:
            docs = self.vectordb.max_marginal_relevance_search(query, k=k, fetch_k=k*2)
        
        return self._remove_duplicates(docs)

    
    def _extract_keywords(self, query: str) -> List[str]:
        """
        Simple keyword extraction from the query
        """
        medical_terms = [
            'symptoms', 'treatment', 'diagnosis', 'disease', 'medication', 
            'therapy', 'syndrome', 'condition', 'patient', 'clinical',
            'medical', 'health', 'doctor', 'hospital', 'medicine'
        ]
        words = re.findall(r'\b\w+\b', query.lower())
        return [word for word in words if len(word) > 3 and word in medical_terms]


    def _remove_duplicates(self, docs: List[Document]) -> List[Document]:
        """Remove duplicate documents based on content similarity"""
        unique_docs = []
        seen_content = set()
        
        for doc in docs:
            # Use first 200 characters as fingerprint
            fingerprint = doc.page_content[:200]
            if fingerprint not in seen_content:
                seen_content.add(fingerprint)
                unique_docs.append(doc)
        
        return unique_docs


class MedicalQuery(BaseModel):
    """Structured representation of a medical query"""
    intent: str = Field(description="Type of medical query: diagnosis, treatment, symptoms, general_info, emergency")
    medical_terms: List[str] = Field(description="Medical terms mentioned in the query")
    urgency_level: str = Field(description="Urgency level: low, medium, high, emergency")
    requires_disclaimer: bool = Field(description="Whether response needs medical disclaimer")


class QueryProcessor:
    def __init__(self, llm):
        self.llm = llm
        self.parser = PydanticOutputParser(pydantic_object=MedicalQuery)
        
        self.analysis_prompt = PromptTemplate(
            template="""
            Analyze this medical query and extract structured information:
            
            Query: {query}
            
            Determine:
            1. Intent (diagnosis, treatment, symptoms, general_info, emergency)
            2. Medical terms mentioned
            3. Urgency level (low, medium, high, emergency)
            4. Whether a medical disclaimer is needed
            
            {format_instructions}
            """,
            input_variables=["query"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
    
    def process_query(self, query: str) -> MedicalQuery:
        """Process and analyze the medical query"""
        try:
            chain = self.analysis_prompt | self.llm | self.parser
            return chain.invoke({"query": query})
        except:
            # Fallback to basic analysis
            return MedicalQuery(
                intent="general_info",
                medical_terms=[],
                urgency_level="low",
                requires_disclaimer=True
            )
    

class MedicalPromptManager:
    def __init__(self):
        self.base_system_prompt = """
        You are a medical assistant that ONLY answers based on the retrieved medical information.
        You MUST prioritize the context provided below when forming your answer.
        - If the context does not have enough information, clearly say: "The retrieved medical information does not fully answer this question."
        - Never invent medical facts.
        - Always remain professional and cautious.

        
        IMPORTANT GUIDELINES:
        1. Always recommend consulting healthcare professionals for medical concerns
        2. Never provide specific medical diagnoses
        3. Focus on educational information and general guidance
        4. Clearly state when information is from recent sources vs. general knowledge
        5. If unsure about anything, acknowledge uncertainty
        """

        self.intent_prompts = {
            "symptoms": """
            The user is asking about symptoms. Provide:
            1. General information about the symptoms mentioned
            2. Common associated conditions (without diagnosing)
            3. When to seek medical attention
            4. General self-care recommendations where appropriate
            """,
            
            "treatment": """
            The user is asking about treatments. Provide:
            1. Overview of common treatment approaches
            2. Evidence-based information from recent sources
            3. Importance of professional medical guidance
            4. Potential considerations or side effects
            """,
            
            "emergency": """
            This appears to be an emergency or urgent medical question. 
            IMMEDIATELY recommend seeking emergency medical care while providing 
            basic information. Use urgent, clear language.
            """
        }

    
    def get_prompt_for_query(self, query_analysis: MedicalQuery, conversation_history: str = ""):
        """Generate prompt for appropriate query"""
        system_content = self.base_system_prompt

        if query_analysis.intent in self.intent_prompts:
            system_content += "\n" + self.intent_prompts[query_analysis.intent]
        
        if query_analysis.urgency_level in ['high', 'emergency']:
            system_content += "\nIMPORTANT: This query has high urgency. Respond accordingly."
        
        if conversation_history:
            user_template = """
            Previous conversation:
            {history}
            
            Retrieved medical information:
            {context}
            
            Current question: {question}
            
            Please provide a helpful, accurate response following the guidelines above.
            """
        else:
            user_template = """
            Retrieved medical information:
            {context}
            
            Question: {question}
            
            Please provide a helpful, accurate response following the guidelines above.
            """
        
        user_template = PromptTemplate(
        template="""
        Retrieved medical information:
        {context}

        Question: {question}

        Previous conversation (if relevant):
        {history}

        Please provide a helpful, accurate response following the guidelines above.
        """,
        input_variables=["context", "question", "history"]
        )

        return ChatPromptTemplate.from_messages([
            SystemMessage(content=system_content),
            HumanMessagePromptTemplate(prompt=user_template)
        ])


#Main enhanced medical agent
class EnhancedMedicalAgent:
    def __init__(self, vectordb_path: str = '../chroma_langchain_db'):
        self.embeddings = GoogleGenerativeAIEmbeddings(model = "models/gemini-embedding-001")
        self.vectordb = Chroma(persist_directory=vectordb_path, embedding_function=self.embeddings)

        self.llm =  ChatGoogleGenerativeAI(
            model="models/gemini-2.5-flash",
            temperature=0.1,
            max_output_tokens=1024
        )

        self.memory = ConversationBufferWindowMemory(
            memory_key = "history",
            return_messages = "true",
            k = 10
        )

        self.retriever = MedicalRetriever(self.vectordb)
        self.query_processor = QueryProcessor(self.llm)
        self.prompt_manager = MedicalPromptManager()

        self.response_metadata = {}
    
    def _format_context(self, docs: List[Document]) -> str:
        """Format retrieved documents into a readable context"""
        formatted_context = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get('source', 'Unknown source')
            # Extract domain for credibility
            if 'cdc.gov' in source: credibility = "High (CDC)"
            elif 'nih.gov' in source: credibility = "High (NIH)"
            elif 'who.int' in source: credibility = "High (WHO)"
            else: credibility = "Moderate"
            
            formatted_context.append(f"""
            Source {i} [{credibility}]:
            {doc.page_content[:500]}...
            (Source: {source})
            """)
    
        return "\n".join(formatted_context)

    def ask(self, query: str) -> Dict[str, Any]:
        """
        Main method to ask agent a question
        """
        start_time = datetime.now()

        #Step 1 Process and analyze the query
        query_analysis = self.query_processor.process_query(query)

        docs = self.retriever.retrieve_with_fallback(query, k=5)

        history = self.memory.buffer_as_str if hasattr(self.memory, 'buffer_as_str') else ""

        prompt = self.prompt_manager.get_prompt_for_query(query_analysis, history)

        context = self._format_context(docs)

        chain = (
            RunnableParallel({
                "context": lambda x: context,
                "question": lambda x: query,
                "history": lambda x: history
            })
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke({})

        if query_analysis.requires_disclaimer:
            response += "\n\nIMPORTANT: This information is for educational purposes only. Always consult a healthcare professional for medical advice."
        
        self.memory.save_context({"input": query}, {"output": response})

        processing_time = (datetime.now() - start_time).total_seconds()

        metadata = {
            "processing_time": processing_time,
            "sources_used": len(docs),
            "query_intent": query_analysis.intent,
            "urgency_level": query_analysis.urgency_level,
            "sources": [doc.metadata.get('source', 'Unknown') for doc in docs]
        }
    
        return {
            "response": response,
            "metadata": metadata,
            "query_analysis": query_analysis
        }
    
    def get_conversation_history(self) -> str:
        """Get formatted conversation history"""
        return self.memory.buffer_as_str if hasattr(self.memory, 'buffer_as_str') else ""
    
    def clear_memory(self) -> str:
        self.memory.clear()
    
def main():
    # Initialize the enhanced medical agent
    agent = EnhancedMedicalAgent()
    
    # Example queries
    test_queries = [
        "What are the symptoms of diabetes?",
        "I have chest pain and shortness of breath",  # Should detect as emergency
        "What are the latest treatments for hypertension?",
        "How can I improve my sleep quality?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print('='*50)
        
        result = agent.ask(query)
        
        print(f"Response: {result['response']}")
        print(f"\nMetadata:")
        for key, value in result['metadata'].items():
            print(f"  {key}: {value}")
        
        print(f"\nQuery Analysis:")
        analysis = result['query_analysis']
        print(f"  Intent: {analysis.intent}")
        print(f"  Urgency: {analysis.urgency_level}")
        print(f"  Medical Terms: {analysis.medical_terms}")

if __name__ == "__main__":
    main()
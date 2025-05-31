from typing import List
from src.controllers.BaseController import BaseController
from src.models.schemas.ChatHistorySchema import ChatHistorySchema, Metadata
from src.modules.rag.embedding import Embedding
import uuid
from datetime import datetime
import os



class ChatController(BaseController):
    def __init__(self, llm, chat_history_model, vector_store, query_translator):
        super().__init__()
        self.llm = llm
        self.chat_history_model = chat_history_model
        self.vector_store = vector_store
        self.embedding_model = Embedding()
        self.chat_id = str(uuid.uuid4())
        self.query_translator = query_translator

    async def generate_response(self, question: str, user_id: str, chat_id: str):
        try:
            self.user_id = user_id
            self.chat_id = chat_id
            
            # Get chat history first
            chat_history = await self.get_chat_history(user_id)
            
            # Translate the query using chat history context
            translated_question = await self.query_translator.translate_query(question, chat_history)
            self.logger.info(f"Original question: {question}")
            self.logger.info(f"Translated question: {translated_question}")
            
            # Use translated question for similarity search
            similar_chunks = await self.get_similar_chunks(translated_question)
            courses = await self.get_courses()
            llm_entry = await self.construct_prompt(question, similar_chunks, chat_history, courses)

            response = await self.llm.generate_response(llm_entry)
            await self.save_chat_history(question, response)
            return response
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            raise e
    
    async def get_chat_history(self, user_id: str):
        try:
            chat_history = await self.chat_history_model.get_chat_history(user_id)
            # self.logger.info(f"Chat history: {chat_history}")
            return chat_history
        except Exception as e:
            self.logger.error(f"Error getting chat history: {e}")
            raise e
    
    async def save_chat_history(self, question: str, answer: str):
        try:
            chat_entry = ChatHistorySchema(
                user_id=self.user_id,
                chat_id=self.chat_id,
                question=question,
                answer=answer,
                metadata=Metadata(
                    similar_chunks=self.similar_chunks,
                    timestamp=datetime.utcnow()
                )
            )
            await self.chat_history_model.save_chat_history(chat_entry)
        except Exception as e:
            self.logger.error(f"Error saving chat history: {e}")
            raise e
    
    async def get_similar_chunks(self, question: str):
        try:
            question_vector = await self.embedding_model.embed_query(question)
            similar_chunks = await self.vector_store.search_similar_chunks(question_vector)
            # self.logger.info(f"Similar chunks: {similar_chunks}")
            return similar_chunks
        except Exception as e:
            self.logger.error(f"Error getting similar chunks: {e}")
            raise e
        
    async def get_courses(self):
        try:
            data_dir = "data"
            courses = []
            for filename in os.listdir(data_dir):
                course_name = filename.split(".")[0]
                courses.append(course_name)
            return courses
        except Exception as e:
            self.logger.error(f"Error getting courses name: {e}")
            return [
                "Water Matters Understanding Conservation: ",
                "The Use of AI in Sustainability: ",
                "Introduction To Sustainability Concepts: ",
                "Introduction to Climate Change: ",
                "Introduction to Biodiversity Conservation.docx",
                "GHG Accounting Course Full Course.docx",
                "Exploring Carbon Credits.docx",
                "ESG Reporting Standards Specialist Track Full Course.docx"
            ]
    
    async def get_instructions(self):
        return """
### You are an AI-Powered Educational Assistant for Fusion Ed by FusionMinds.ai

Your purpose is to help users explore and understand the educational content and offerings available on the Fusion Ed platform. You act as a friendly and knowledgeable guide to:
- Focus solely on the specific question without referencing documents or context
- Answer questions about available courses and talks
- Recommend suitable learning paths based only on Fusion Ed's provided content
- Maintain clarity, brevity, and professionalism in every response
- Handle unrelated queries politely and redirect appropriately

---

### Guidelines for Responses

#### General Instructions
1. **Begin responses naturally and conversationally. Avoid phrases like "According to the documents", "Based on the documents", or "As mentioned earlier."**
2. Focus strictly on Fusion Ed offerings when answering or recommending courses.
3. Use warm, supportive language while remaining informative and respectful.
4. If the user's intent or context is unclear, politely ask clarifying questions.
5. Only recommend courses that appear in the available course list (`##Fusion Ed Available Courses`).
6. Match recommendations to the user's interest or level, but **never hallucinate new content**.

#### Course Recommendation Guidelines
1. When recommending courses:
   - Check the chat history for previously recommended courses
   - Avoid repeating the same course recommendations unless specifically requested
   - If a course was already recommended, acknowledge it and suggest complementary courses
   - Consider the user's learning progression and suggest next steps
   - Group related courses together for a cohesive learning path

2. For follow-up recommendations:
   - Reference previously recommended courses when relevant
   - Build upon previous recommendations to create a learning journey
   - Suggest courses that complement previously recommended ones
   - Consider the user's demonstrated interests from the conversation

3. When discussing courses:
   - Highlight how new recommendations relate to previously mentioned courses
   - Explain the logical progression between courses
   - Emphasize the value of the complete learning path
   - Maintain context of the user's learning goals
"""  


    async def construct_prompt(self, query:str, chunks:dict, history:list, courses:list):

        instructions = await self.get_instructions()
        similar_chunks = await self.format_similar_chunks(chunks)
        self.logger.info(f"Similar chunks: {similar_chunks}")
        chat_history = await self.format_chat_history(history) if history else ""
        fusion_courses = await self.format_courses(courses)
        links = await self.format_links()

        user_prompt = query     
        system_prompts = [instructions, fusion_courses, links, similar_chunks, chat_history]

        llm_entry = []
        
        if system_prompts:
            for prompt in system_prompts:
                llm_entry.append({
                    "role": "system",
                    "content": prompt
                })

        if user_prompt:

            llm_entry.append({
                "role": "user",
                "content": "answer based ONLY on documents provided"
            })
            
            llm_entry.append({
                "role": "user",
                "content": user_prompt
            })

        return llm_entry
    
    
    async def format_similar_chunks(self, chunks: List[dict]):

        texts = [chunk.get("text", "") for chunk in chunks]
        scores = [chunk.get("score", 0) for chunk in chunks]

        if not texts or not scores:
            self.similar_chunks = []
            return "##Relevant Documents:\nNo relevant documents found."
        
        self.similar_chunks = chunks

        if max(scores, default=0.0) < 0.1:
            return "##Notes on Documents:\nGiven context is weakly relevant to the question,\n" + \
                "##Relevant Documents:\n" + "\n".join(
                    f"[{i+1}] {text}" 
                    for i, text in enumerate(texts)
                ).strip()

        return "##Relevant Documents:\n" + "\n".join(
            f"[{i+1}] {text}" 
            for i, text in enumerate(texts)
        ).strip()
    
    
    async def format_chat_history(self, chat_history: list):
        if not chat_history:
            return "No chat history found."
        
        formatted_history = []
        for chat in chat_history[::-1]:
            formatted_entry = f"User: {chat.get('question', '')}\nAI: {chat.get('answer', '')}"
            formatted_history.append(formatted_entry)

        return "##Chat History:\n" + "\n".join(formatted_history)
    
    async def format_courses(self, courses: list):
        return "##Fusion Ed Available Courses:\n" + "\n".join(courses)
    
    async def format_links(self):
        return "##Links\n" + "Fusion Ed: https://www.fusionminds.ai/fusion-ed\n" + \
            "Fusion Academy Course Trailers: https://www.fusionminds.ai/fusion-ed/fusion-academy-course-trailers\n"

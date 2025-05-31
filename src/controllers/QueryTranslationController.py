from src.controllers.BaseController import BaseController

class QueryTranslationController(BaseController):
    def __init__(self, llm):
        super().__init__()
        self.llm = llm

    async def translate_query(self, question: str, chat_history: list) -> str:
        if not chat_history:
            return question

        formatted_history = await self.format_chat_history(chat_history)
        instructions = await self.get_instructions()
        llm_entry = await self.construct_prompt(question, [instructions, formatted_history])

        response = await self.llm.generate_response(llm_entry)

        return response.strip()


    async def get_instructions(self):
        return """You are a Query Translation Expert. Your task is to transform follow-up questions into complete, self-contained questions by incorporating relevant context from the chat history.

### Core Principles:
1. Maintain the original question's intent and focus
2. Only include context that is directly relevant to the current question
3. Keep the translated query concise and clear
4. Preserve any specific terminology or technical terms from the original question

### Guidelines:
- If the current question is self-contained, return it unchanged
- If the question references previous context, incorporate only the essential context
- Avoid adding unnecessary information or assumptions
- Keep the translated query focused on a single main topic
- Maintain the original question's tone and formality level

### Examples:
Original: "What are its effects?"
Context: Previous question about climate change
Translated: "What are the effects of climate change?"

Original: "How does it work?"
Context: Previous question about carbon credits
Translated: "How do carbon credits work?"

### Output Format:
- Return only the translated question
- No explanations or additional text
- Keep the response concise and direct"""
    

    async def format_chat_history(self, chat_history: list):
        
        chat_history = chat_history[:3] if len(chat_history) > 3 else chat_history

        formatted_history = []
        for chat in chat_history[::-1]:
            formatted_entry = f"User: {chat.get('question', '')}\nAI: {chat.get('answer', '')}"
            formatted_history.append(formatted_entry)

        return "##Chat History:\n" + "\n".join(formatted_history)
    

    async def construct_prompt(self, user_prompt: str, system_prompts: list):

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
                "content": user_prompt
            })

        return llm_entry
        
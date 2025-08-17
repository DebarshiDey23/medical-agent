from langgraph import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chat_models.base import init_chat_model


def check_safety(query: str):
    PROMPT = ChatPromptTemplate([
        ("system", """You are a medical safety classifier. Analyze the user's query and classify it into exactly one category:

EMERGENCY: User is describing current symptoms of life-threatening conditions requiring immediate medical attention (chest pain, difficulty breathing, severe bleeding, loss of consciousness, signs of stroke/heart attack, overdose, severe allergic reactions, suicidal ideation)

CONCERNING: User is describing current symptoms that are worrying but not immediately life-threatening (persistent pain, unusual symptoms, medication concerns)

INFORMATION: User is seeking general medical information, research, or asking hypothetical questions

Respond with ONLY the classification word: EMERGENCY, CONCERNING, or INFORMATION

Examples:
"I'm having severe chest pain right now" → EMERGENCY
"What causes chest pain?" → INFORMATION  
"I've had this weird rash for a week" → CONCERNING
"My friend overdosed, what do I do?" → EMERGENCY
"What are the side effects of aspirin?" → INFORMATION"""),
        ("user", query)
    ])
    
    messages = PROMPT.format_messages(query=query)
    chat = init_chat_model("google_vertexai:gemini-2.5-flash", temperature=0)
    response = chat.invoke(messages)
    
    classification = response.content.strip().upper()
    
    if classification == "EMERGENCY":
        return False, "emergency"  # Unsafe, block query
    elif classification == "CONCERNING":
        return True, "concerning"  # Safe but add disclaimers
    elif classification == "INFORMATION":
        return True, "information"  # Safe, normal processing
    else:
        # Fallback - if unclear response, err on side of caution
        return False, "unclear"
    


def process_user_query(query: str):
    is_safe, classification = check_safety(query)
    
    if not is_safe:
        return {
            "status": "error",
            "message": "Your query contains potentially unsafe content. Please seek immediate medical attention."
        }
    
    if classification == "concerning":
        return {
            "status": "warning",
            "message": "Your query raises some concerns. Please consult a healthcare professional for advice."
        }
    
    # For information queries, proceed normally
    return {
        "status": "success",
        "message": "Your query is safe and will be processed."
    }
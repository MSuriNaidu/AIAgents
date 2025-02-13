from phi.agent import Agent
from phi.model.groq import Groq
from dotenv import load_dotenv

load_dotenv()

agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile")
)

#agent.print_response("Summarize and compare analyst recommendation and fundamentals for Tesla and Nvda")
agent.print_response("Write 5 lines about the Agentic AI")
import os

from pydantic import BaseModel
from outlines import models, generate
from llama_cpp import Llama

class Contact(BaseModel):
    email: str
    phone: str
    address: str

class Experience(BaseModel):
    company: str
    role: str
    start_date: str
    end_date: str
    description: str

class Education(BaseModel):
    institution: str
    degree: str
    graduation_year: int

class Resume(BaseModel):
    name: str
    contact: Contact
    experience: list[Experience]
    education: list[Education]
    skills: list[str]
    summary: str

llm = Llama(
    model_path=os.path.join(os.path.dirname(__file__), "mistral-7b-v0.1.Q5_K_M.gguf"),
    n_threads=4,
    n_ctx=16384,
    verbose=False
)

model = models.LlamaCpp(llm)

generator = generate.json(model, Resume)
resume_text = input("Please paste the resume text: ")

result = generator(resume_text)
print(result)
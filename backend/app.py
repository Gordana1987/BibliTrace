from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import health, analyze, corpora, search

app = FastAPI(
    title="BibliTrace API",
    description="Concept search over the Serbian New Testament (exact / lemma / semantic).",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(search.router)
app.include_router(analyze.router)
app.include_router(corpora.router)


@app.get("/")
async def root():
    return {"app": "BibliTrace", "docs": "/docs"}

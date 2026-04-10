Hi, I’ll quickly walk through the system I built for this assignment.

The goal here was to build something that can handle real-world investment documents — not just answer questions, but keep outputs **grounded**, **traceable**, and **reliable**.

**Setup (short):** PDFs are indexed **once** from the terminal (`python ingest.py ./folder` with `--company` / `--version` where needed), then this Streamlit app **only** retrieves and answers. Open the README mermaid: offline chunk → embed → DB, then question → retrieve → LLM.

So this is the interface. I’ve already **indexed** a set of PDFs, and I can query across all of them.

Let me start with a simple question. I’ll ask, what is the strategy of Digital Realty.

You can see the system retrieves relevant sections and generates an answer. What I focused on here is making sure the response is tied back to the source, so every part of the answer includes citations with the document name and page number. That way, it is not just generating text, but actually pointing to where the information is coming from.

**Ingest tip for the recording:** to show **two versions** of the same company, run two passes, e.g.  
`python ingest.py ./dlr_dec --company "Digital Realty" --version 2025-12` then  
`python ingest.py ./dlr_mar --company "Digital Realty" --version 2026-03`,  
then refresh Streamlit and check **Indexed documents** in the sidebar.

Now I’ll try something across documents. I’ll ask, compare EastGroup and Digital Realty.

Here, the system pulls context from multiple documents and gives a comparison. For example, EastGroup is focused more on physical logistics infrastructure, while Digital Realty is focused on data centers and AI driven demand. So instead of treating each document in isolation, it is able to reason across them.

One thing I was particularly careful about is handling conflicting information. Let me ask, are there conflicting data points across documents.

If there are differences, the system does not try to merge or resolve them automatically. Instead, it surfaces both and attributes them clearly. I felt that was important because in this kind of use case, it is better to expose differences than to risk introducing incorrect assumptions.

**Optional:** ingest with different `--client` values and use the sidebar **Client** filter to scope listing and retrieval.

Under the hood, the system is a straightforward pipeline: document ingestion, chunking, embeddings, retrieval, and then generation based strictly on the retrieved context. I tried to keep the design simple, but focused on making it reliable.

There are a few limitations. Charts and visual heavy pages are not fully extracted, so in those cases the system treats them as text where possible. Also, conflict resolution is not automated, it is surfaced instead. With more time, I would improve structured data extraction and retrieval quality further.

Overall, the idea was to build something that behaves more like a decision support system than a chatbot, where the emphasis is on grounding and traceability.

Happy to walk through any part in more detail.

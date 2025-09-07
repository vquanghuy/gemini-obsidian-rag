# Gemini Obsidian RAG

A local, private Retrieval-Augmented Generation (RAG) pipeline to chat with your entire Obsidian vault using the Google Gemini CLI. This project turns your personal knowledge base into a conversational assistant, allowing you to ask questions and get answers grounded in your own notes.

## The Problem

The Gemini CLI is a powerful tool, but its effectiveness with personal knowledge bases like Obsidian is limited by the model's context window. Manually loading a few notes at a time is inefficient, doesn't scale as your vault grows, and makes it impossible to reason across your entire collection of knowledge.

## The Solution

This project implements a complete RAG pipeline that runs entirely on your local machine, ensuring your notes remain private. It intelligently indexes your entire vault, creating a searchable knowledge base that Gemini can query in real-time.

### How It Works: Architecture Diagram

The system is split into two phases: a one-time setup/maintenance phase and a live query workflow.

1. **Setup & Maintenance:** Python scripts read your Obsidian vault, use a local embedding model (`nomic-ai/nomic-embed-text-v1.5`) to understand the content, and store this knowledge in a local ChromaDB vector database.
    
2. **Live Query Workflow:** When you ask a question in the Gemini CLI, it connects to a local MCP Server. This server queries the ChromaDB for relevant information, passes that context back to the Gemini model, and allows it to generate an accurate answer based on your notes.
    

## Features

- **100% Private:** Your notes and queries are never sent to a third party for indexing. Everything runs on your local machine.
- **Scalable:** Query thousands of notes as easily as you can query a dozen. The size of your vault is no longer limited by a context window. 
- **Intelligent Search:** Uses semantic search to find conceptually related notes, not just keyword matches.
- **Efficient:** Minimizes API usage by sending only the most relevant context to the Gemini model with each query.
- **Simple Setup:** Designed with minimal dependencies and an embedded vector database to get up and running quickly.
    

## Setup and Installation

Follow these steps to get the system running.

### Step 1: Clone the Repository

Clone this repository to your local machine:

```
git clone [https://github.com/your-username/gemini-obsidian-rag.git](https://github.com/your-username/gemini-obsidian-rag.git)
cd gemini-obsidian-rag
```

### Step 2: Install Dependencies

This project uses Python 3.9+. Install all required libraries from `requirements.txt`:

```
pip install -r requirements.txt
```

### Step 3: Configure the Scripts

You must tell the scripts where your Obsidian vault is located.

1. Open `1_create_index_langchain.py`, `2_run_mcp_server_langchain.py`, and `3_update_index_langchain.py` in a text editor.
2. In each file, change the `VAULT_PATH` variable to the **full, absolute path** of your Obsidian vault.
    
    ```
    # Example for macOS/Linux
    VAULT_PATH = "/Users/yourname/Documents/MyObsidianVault"
    
    # Example for Windows
    VAULT_PATH = "C:\\Users\\yourname\\Documents\\MyObsidianVault"
    ```
    
3. Save all three files.

### Step 4: Run the Initial Indexing

This is a one-time process that creates the vector database for your vault. In your terminal, run:

```
python 1_create_index_langchain.py
```

This will download the embedding model on the first run and then process all your notes. A new folder named `chroma_db_langchain` will be created.

## Usage

To use your new Obsidian assistant, follow these two steps.

### Step 1: Run the MCP Server

The MCP server must be running in the background to answer queries. In a terminal, navigate to the project folder and run:

```
python 2_run_mcp_server_langchain.py
```

Keep this terminal window open.

### Step 2: Configure and Run Gemini CLI

1. Open your Gemini CLI settings file, typically located at `~/.gemini/settings.json`.
2. Add the `mcpServers` configuration to point to your local server:
```
{
  "theme": "Default",
  "mcpServers": {
    "obsidian": {
      "httpUrl": "[http://127.0.0.1:8000/mcp](http://127.0.0.1:8000/mcp)"
    }
  }
}
```
3. Save the file.
4. Open a **new terminal window** and start the Gemini CLI. You can now ask questions about your vault!

## Keeping Your Knowledge Base Fresh

As you add and edit notes in Obsidian, your index will become outdated. To update it, simply run the `update` script.

### Manual Update

Run the following command in your terminal whenever you want to sync your latest notes:

```
python 3_update_index_langchain.py
```

This script will intelligently find and process only the new or modified files, making it much faster than a full re-index.

## Project Structure

- `requirements.txt`: A list of all the Python libraries needed for the project.
- `1_create_index_langchain.py`: Script to perform the initial, full indexing of the Obsidian vault.
- `2_run_mcp_server_langchain.py`: The local MCP server that listens for requests from the Gemini CLI and queries the database.
- `3_update_index_langchain.py`: Script to perform fast, incremental updates to the vector database.
- `README.md`: This file.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

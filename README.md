# Humanizer MCP Server

A [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server that rewrites AI-sounding English or Arabic text so it reads more naturally. It preserves meaning, validates the requested word range and required keywords, and returns a Markdown report with before-and-after analytics.

## Features

- Automatic English/Arabic language detection.
- Groq-powered rewriting with language-specific prompts.
- Guardrails for word count and keyword preservation, with automatic retries.
- Burstiness and heuristic AI-trigger analytics before and after rewriting.
- Optional semantic caching with Redis and `all-MiniLM-L6-v2` embeddings.
- Supports MCP `stdio`, `sse`, and `streamable-http` transports.

## Requirements

- Python 3.10 or newer
- A Groq API key
- Redis is optional; without it, the server continues without caching.

## Installation

From this directory, create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Set the required API key. Never commit API keys to source control.

Windows PowerShell:

```powershell
$env:GROQ_API_KEY = "your-groq-api-key"
```

macOS/Linux:

```bash
export GROQ_API_KEY="your-groq-api-key"
```

## Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `GROQ_API_KEY` | — | Required Groq API key |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Groq model name |
| `REDIS_URL` | `redis://localhost:6379/0` | Optional Redis connection URL |
| `HUMANIZER_EMBED_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer model used for semantic caching |
| `HUMANIZER_CACHE_TTL` | `604800` | Cache lifetime in seconds |
| `MCP_TRANSPORT` | `stdio` | MCP transport: `stdio`, `sse`, or `streamable-http` |

The embedding model is downloaded by `sentence-transformers` the first time semantic caching is used. Redis is not required for text generation.

## Run

Start the server with the default `stdio` transport:

```bash
python humanizer_server.py
```

For an HTTP-based MCP transport, set the transport before starting:

```powershell
$env:MCP_TRANSPORT = "streamable-http"
python humanizer_server.py
```

## Run in GitHub Codespaces

1. Open the repository on GitHub.
2. Select **Code** → **Codespaces** → **Create codespace on main**.
3. Wait for the development container to install the dependencies.
4. Add `GROQ_API_KEY` under the codespace's environment secrets, or set it in the terminal:

   ```bash
   export GROQ_API_KEY="your-groq-api-key"
   ```

5. Start the MCP server:

   ```bash
   python humanizer_server.py
   ```

The repository includes `.devcontainer/devcontainer.json`, so Codespaces uses a consistent Python environment and installs `requirements.txt` automatically. Codespaces is intended for interactive development and testing; it does not keep the server running after the codespace stops.

Configure the server in an MCP client such as Claude Desktop or another compatible host. Example `stdio` configuration:

```json
{
  "mcpServers": {
    "humanizer": {
      "command": "python",
      "args": ["C:\\path\\to\\humanizer\\humanizer_server.py"],
      "env": {
        "GROQ_API_KEY": "your-groq-api-key"
      }
    }
  }
}
```

### Claude Desktop on Windows

Claude Desktop must be installed on the same computer as the server for a `stdio` connection. A Codespace by itself is not a local process that Claude Desktop can launch.

1. Clone this repository locally, or use the existing local folder.
2. Create the virtual environment and install dependencies as described above.
3. Find the configuration file from Claude Desktop's **Settings → Developer → Edit Config** option. If needed, the Windows file is commonly located at:

   ```text
   %APPDATA%\Claude\claude_desktop_config.json
   ```

4. Copy the contents of `claude_desktop_config.example.json` into that file.
5. Replace both `C:\path\to\humanizer` values with the real absolute path to this repository.
6. Replace `replace-with-your-groq-api-key` with your Groq key. Do not commit the real key.
7. Fully quit and reopen Claude Desktop.

For example, if the repository is located at `D:\projects\humanizer-mcp-server`, use:

```json
{
  "mcpServers": {
    "humanizer": {
      "command": "D:\\projects\\humanizer-mcp-server\\.venv\\Scripts\\python.exe",
      "args": [
        "D:\\projects\\humanizer-mcp-server\\humanizer_server.py"
      ],
      "env": {
        "GROQ_API_KEY": "your-groq-api-key"
      }
    }
  }
}
```

After restarting Claude Desktop, look for the MCP tools/hammer icon and enable `humanize_text`. You can then ask Claude to humanize English or Arabic text and specify a word range or keywords to preserve.

To diagnose startup problems, run the same command in PowerShell:

```powershell
D:\projects\humanizer-mcp-server\.venv\Scripts\python.exe D:\projects\humanizer-mcp-server\humanizer_server.py
```

The server communicates over standard input/output, so normal logs are written to standard error. Do not add `--transport` arguments; the default `stdio` transport is the correct one for Claude Desktop.

## Tool

The server exposes one tool:

`humanize_text(text, min_words=50, max_words=500, keywords_to_preserve=[])`

- `text`: English or Arabic source text.
- `min_words`: Minimum allowed output length.
- `max_words`: Maximum allowed output length.
- `keywords_to_preserve`: Words or phrases that must remain verbatim.

The result includes the rewritten text, language, word counts, sentence counts, burstiness, heuristic AI detection risk, and keyword-preservation status.

## Helper script

`my_available_groq_models.py` lists models available to the configured Groq account:

```bash
python my_available_groq_models.py
```

It reads `GROQ_API_KEY` from the environment and does not contain credentials.

## Project structure

```text
.
├── humanizer_server.py
├── my_available_groq_models.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Limitations

- The AI detection score is a heuristic based on known trigger phrases; it is not a real AI detector.
- The default model and available Groq models can change over time.
- The first embedding-model load may take time and requires network access.

## License

No license has been specified yet. Add a license before distributing the project or accepting external contributions.

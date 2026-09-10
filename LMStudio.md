

LM Studio offers a powerful REST API with first-class support for local inference and model management. In addition to our native API, we provide OpenAI-compatible endpoints ([learn more](/docs/developer/openai-compat)) and Anthropic-compatible endpoints ([learn more](/docs/developer/anthropic-compat)).

What's new [#whats-new]

Previously, there was a [v0 REST API](/docs/developer/rest/endpoints). With LM Studio 0.4.0, we have officially released our native v1 REST API at `/api/v1/*` endpoints and recommend using it.

The v1 REST API includes enhanced features such as:

* [MCP via API](/docs/developer/core/mcp)
* [Stateful chats](/docs/developer/rest/stateful-chats)
* [Authentication](/docs/developer/core/authentication) configuration with API tokens
* Model [download](/docs/developer/rest/download), [load](/docs/developer/rest/load) and [unload](/docs/developer/rest/unload) endpoints

Supported endpoints [#supported-endpoints]

The following endpoints are available in LM Studio's v1 REST API.

| Endpoint | Method | Docs |
|---|---|---|
| `/api/v1/chat` | POST | [Chat](/docs/developer/rest/chat) |
| `/api/v1/models` | GET | [List Models](/docs/developer/rest/list) |
| `/api/v1/models/load` | POST | [Load](/docs/developer/rest/load) |
| `/api/v1/models/unload` | POST | [Unload](/docs/developer/rest/unload) |
| `/api/v1/models/download` | POST | [Download](/docs/developer/rest/download) |
| `/api/v1/models/download/status` | GET | [Download Status](/docs/developer/rest/download-status) |

Inference endpoint comparison [#inference-endpoint-comparison]

The table below compares the features of LM Studio's `/api/v1/chat` endpoint with OpenAI-compatible and Anthropic-compatible inference endpoints.

| Feature | `/api/v1/chat` | `/v1/chat/completions` (OpenAI) | `/v1/messages` (Anthropic) |
|---|---|---|---|
| Streaming | ✅ | ✅ | ✅ |
| Stateful chat | ✅ | ❌ | ❌ |
| Remote MCPs | ✅ | ❌ | ❌ |
| MCPs you have in LM Studio | ✅ | ❌ | ❌ |
| Custom tools | ❌ | ✅ | ✅ |
| Include assistant messages in the request | ❌ | ✅ | ✅ |
| Model load streaming events | ✅ | ❌ | ❌ |
| Prompt processing streaming events | ✅ | ❌ | ❌ |
| Specify context length in the request | ✅ | ❌ | ❌ |

***

Please report bugs by opening an issue on [Github](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues).

Start the server [#start-the-server]

[Install](/download) and launch LM Studio.

Then ensure the server is running through the toggle at the top left of the Developer page, or through [lms](/docs/cli) in the terminal:

```bash
lms server start
```

By default, the server is available at `http://localhost:1234`.

If you don't have a model downloaded yet, you can download the model:

```bash
lms get ibm/granite-4-micro
```

API Authentication [#api-authentication]

By default, the LM Studio API server does **not** require authentication. You can configure the server to require authentication by API token in the [server settings](/docs/developer/core/server/settings) for added security.

To authenticate API requests, generate an API token from the Developer page in LM Studio, and include it in the `Authorization` header of your requests as follows: `Authorization: Bearer $LM_API_TOKEN`. Read more about authentication [here](/docs/developer/core/authentication).

Chat with a model [#chat-with-a-model]

Use the chat endpoint to send a message to a model. By default, the model will be automatically loaded if it is not already.

The `/api/v1/chat` endpoint is stateful, which means you do not need to pass the full history in every request. Read more about it [here](/docs/developer/rest/stateful-chats).



  
#### curl

    ```bash
    curl http://localhost:1234/api/v1/chat \
      -H "Authorization: Bearer $LM_API_TOKEN" \
      -H "Content-Type: application/json" \
      -d '{
        "model": "ibm/granite-4-micro",
        "input": "Write a short haiku about sunrise."
      }'
    ```
  

  
#### Python

    ```python
    import os
    import requests
    import json

    response = requests.post(
      "http://localhost:1234/api/v1/chat",
      headers={
        "Authorization": f"Bearer {os.environ['LM_API_TOKEN']}",
        "Content-Type": "application/json"
      },
      json={
        "model": "ibm/granite-4-micro",
        "input": "Write a short haiku about sunrise."
      }
    )
    print(json.dumps(response.json(), indent=2))
    ```
  

  
#### TypeScript

    ```typescript
    const response = await fetch("http://localhost:1234/api/v1/chat", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${process.env.LM_API_TOKEN}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model: "ibm/granite-4-micro",
        input: "Write a short haiku about sunrise."
      })
    });
    const data = await response.json();
    console.log(data);
    ```
  

See the full [chat](/docs/developer/rest/chat) docs for more details.

Use MCP servers via API [#use-mcp-servers-via-api]

Enable the model interact with ephemeral Model Context Protocol (MCP) servers in `/api/v1/chat` by specifying servers in the `integrations` field.

  
    
      curl
    </CodeBlockTabsTrigger>

    
      Python
    </CodeBlockTabsTrigger>

    
      TypeScript
    </CodeBlockTabsTrigger>
  

  
#### curl

    ```bash
    curl http://localhost:1234/api/v1/chat \
      -H "Authorization: Bearer $LM_API_TOKEN" \
      -H "Content-Type: application/json" \
      -d '{
        "model": "ibm/granite-4-micro",
        "input": "What is the top trending model on hugging face?",
        "integrations": [
          {
            "type": "ephemeral_mcp",
            "server_label": "huggingface",
            "server_url": "https://huggingface.co/mcp",
            "allowed_tools": ["model_search"]
          }
        ],
        "context_length": 8000
      }'
    ```
  

  
#### Python

    ```python
    import os
    import requests
    import json

    response = requests.post(
      "http://localhost:1234/api/v1/chat",
      headers={
        "Authorization": f"Bearer {os.environ['LM_API_TOKEN']}",
        "Content-Type": "application/json"
      },
      json={
        "model": "ibm/granite-4-micro",
        "input": "What is the top trending model on hugging face?",
        "integrations": [
          {
            "type": "ephemeral_mcp",
            "server_label": "huggingface",
            "server_url": "https://huggingface.co/mcp",
            "allowed_tools": ["model_search"]
          }
        ],
        "context_length": 8000
      }
    )
    print(json.dumps(response.json(), indent=2))
    ```
  

  
#### TypeScript

    ```typescript
    const response = await fetch("http://localhost:1234/api/v1/chat", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${process.env.LM_API_TOKEN}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model: "ibm/granite-4-micro",
        input: "What is the top trending model on hugging face?",
        integrations: [
          {
            type: "ephemeral_mcp",
            server_label: "huggingface",
            server_url: "https://huggingface.co/mcp",
            allowed_tools: ["model_search"]
          }
        ],
        context_length: 8000
      })
    const data = await response.json();
    console.log(data);
    ```
  

You can also use locally configured MCP plugins (from your `mcp.json`) via the `integrations` field. Using locally run MCP plugins requires authentication via an API token passed through the `Authorization` header. Read more about authentication [here](/docs/developer/core/authentication).

  
    
      curl
    </CodeBlockTabsTrigger>

    
      Python
    </CodeBlockTabsTrigger>

    
      TypeScript
    </CodeBlockTabsTrigger>
  

  
#### curl

    ```bash
    curl http://localhost:1234/api/v1/chat \
      -H "Authorization: Bearer $LM_API_TOKEN" \
      -H "Content-Type: application/json" \
      -d '{
        "model": "ibm/granite-4-micro",
        "input": "Open lmstudio.ai",
        "integrations": [
          {
            "type": "plugin",
            "id": "mcp/playwright",
            "allowed_tools": ["browser_navigate"]
          }
        ],
        "context_length": 8000
      }'
    ```
  

  
#### Python

    ```python
    import os
    import requests
    import json

    response = requests.post(
      "http://localhost:1234/api/v1/chat",
      headers={
        "Authorization": f"Bearer {os.environ['LM_API_TOKEN']}",
        "Content-Type": "application/json"
      },
      json={
        "model": "ibm/granite-4-micro",
        "input": "Open lmstudio.ai",
        "integrations": [
          {
            "type": "plugin",
            "id": "mcp/playwright",
            "allowed_tools": ["browser_navigate"]
          }
        ],
        "context_length": 8000
      }
    )
    print(json.dumps(response.json(), indent=2))
    ```
  

  
#### TypeScript

    ```typescript
    const response = await fetch("http://localhost:1234/api/v1/chat", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${process.env.LM_API_TOKEN}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model: "ibm/granite-4-micro",
        input: "Open lmstudio.ai",
        integrations: [
          {
            type: "plugin",
            id: "mcp/playwright",
            allowed_tools: ["browser_navigate"]
          }
        ],
        context_length: 8000
      })
    });
    const data = await response.json();
    console.log(data);
    ```
  

See the full [chat](/docs/developer/rest/chat) docs for more details.

Download a model [#download-a-model]

Use the download endpoint to download models by identifier from the [LM Studio model catalog](https://lmstudio.ai/models), or by Hugging Face model URL.

  
    
      curl
    </CodeBlockTabsTrigger>

    
      Python
    </CodeBlockTabsTrigger>

    
      TypeScript
    </CodeBlockTabsTrigger>
  

  
#### curl

    ```bash
    curl http://localhost:1234/api/v1/models/download \
      -H "Authorization: Bearer $LM_API_TOKEN" \
      -H "Content-Type: application/json" \
      -d '{
        "model": "ibm/granite-4-micro"
      }'
    ```
  

  
#### Python

    ```python
    import os
    import requests
    import json

    response = requests.post(
      "http://localhost:1234/api/v1/models/download",
      headers={
        "Authorization": f"Bearer {os.environ['LM_API_TOKEN']}",
        "Content-Type": "application/json"
      },
      json={"model": "ibm/granite-4-micro"}
    )
    print(json.dumps(response.json(), indent=2))
    ```
  

  
#### TypeScript

    ```typescript
    const response = await fetch("http://localhost:1234/api/v1/models/download", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${process.env.LM_API_TOKEN}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model: "ibm/granite-4-micro"
      })
    });
    const data = await response.json();
    console.log(data);
    ```
  

The response will return a `job_id` that you can use to track download progress.

  
    
      curl
    </CodeBlockTabsTrigger>

    
      Python
    </CodeBlockTabsTrigger>

    
      TypeScript
    </CodeBlockTabsTrigger>
  

  
#### curl

    ```bash
    curl -H "Authorization: Bearer $LM_API_TOKEN" \
      http://localhost:1234/api/v1/models/download/status/{job_id}
    ```
  

  
#### Python

    ```python
    import os
    import requests
    import json

    job_id = "your-job-id"
    response = requests.get(
      f"http://localhost:1234/api/v1/models/download/status/{job_id}",
      headers={"Authorization": f"Bearer {os.environ['LM_API_TOKEN']}"}
    )
    print(json.dumps(response.json(), indent=2))
    ```
  

  
#### TypeScript

    ```typescript
    const jobId = "your-job-id";
    const response = await fetch(
      `http://localhost:1234/api/v1/models/download/status/${jobId}`,
      {
        headers: {
          "Authorization": `Bearer ${process.env.LM_API_TOKEN}`
        }
      }
    );
    const data = await response.json();
    console.log(data);
    ```
  

See the [download](/docs/developer/rest/download) and [download status](/docs/developer/rest/download-status) docs for more details.

The `/api/v1/chat` endpoint is stateful by default. This means you don't need to pass the full conversation history in every request — LM Studio automatically stores and manages the context for you.

How it works [#how-it-works]

When you send a chat request, LM Studio stores the conversation in a chat thread and returns a `response_id` in the response. Use this `response_id` in subsequent requests to continue the conversation.

```bash title="Start a new conversation"
curl http://localhost:1234/api/v1/chat \
  -H "Authorization: Bearer $LM_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ibm/granite-4-micro",
    "input": "My favorite color is blue."
  }'
```

The response includes a `response_id`:

> [!NOTE]
> **Info**
> Every response includes an unique `response_id` that you can use to reference that specific point in the conversation for future requests. This allows you to branch conversations.

```json title="Response"
{
  "model_instance_id": "ibm/granite-4-micro",
  "output": [
    {
      "type": "message",
      "content": "That's great! Blue is a beautiful color..."
    }
  ],
  "response_id": "resp_abc123xyz..."
}
```

Continue a conversation [#continue-a-conversation]

Pass the `previous_response_id` in your next request to continue the conversation. The model will remember the previous context.

```bash title="Continue the conversation"
curl http://localhost:1234/api/v1/chat \
  -H "Authorization: Bearer $LM_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ibm/granite-4-micro",
    "input": "What color did I just mention?",
    "previous_response_id": "resp_abc123xyz..."
  }'
```

The model can reference the previous message without you needing to resend it and will return a new `response_id` for further continuation.

Disable stateful storage [#disable-stateful-storage]

If you don't want to store the conversation, set `store` to `false`. The response will not include a `response_id`.

```bash title="Stateless chat"
curl http://localhost:1234/api/v1/chat \
  -H "Authorization: Bearer $LM_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ibm/granite-4-micro",
    "input": "Tell me a joke.",
    "store": false
  }'
```

This is useful for one-off requests where you don't need to maintain context.

Streaming events let you render chat responses incrementally over Server‑Sent Events (SSE). When you call `POST /api/v1/chat` with `stream: true`, the server emits a series of named events that you can consume. These events arrive in order and may include multiple deltas (for reasoning and message content), tool call boundaries and payloads, and any errors encountered. The stream always begins with `chat.start` and concludes with `chat.end`, which contains the aggregated result equivalent to a non‑streaming response.

List of event types that can be sent in an `/api/v1/chat` response stream:

* `chat.start`
* `model_load.start`
* `model_load.progress`
* `model_load.end`
* `prompt_processing.start`
* `prompt_processing.progress`
* `prompt_processing.end`
* `reasoning.start`
* `reasoning.delta`
* `reasoning.end`
* `tool_call.start`
* `tool_call.arguments`
* `tool_call.success`
* `tool_call.failure`
* `message.start`
* `message.delta`
* `message.end`
* `error`
* `chat.end`

Events will be streamed out in the following raw format:

```bash
event: <event type>
data: <JSON event data>
```

`chat.start` [#chatstart]

  
    An event that is emitted at the start of a chat response stream.

    
- **`model_instance_id`** (`string`): Unique identifier for the loaded model instance that will generate the response.
- **`type`** (`"chat.start"`): The type of the event. Always `chat.start`.

  

  
    ```json title="Example Event Data"
    {
      "type": "chat.start",
      "model_instance_id": "openai/gpt-oss-20b"
    }
    ```
  

`model_load.start` [#model_loadstart]

  
    Signals the start of a model being loaded to fulfill the chat request. Will not be emitted if the requested model is already loaded.

    
- **`model_instance_id`** (`string`): Unique identifier for the model instance being loaded.
- **`type`** (`"model_load.start"`): The type of the event. Always `model_load.start`.

  

  
    ```json title="Example Event Data"
    {
      "type": "model_load.start",
      "model_instance_id": "openai/gpt-oss-20b"
    }
    ```
  

`model_load.progress` [#model_loadprogress]

  
    Progress of the model load.

    
- **`model_instance_id`** (`string`): Unique identifier for the model instance being loaded.
- **`progress`** (`number`): Progress of the model load as a float between `0` and `1`.
- **`type`** (`"model_load.progress"`): The type of the event. Always `model_load.progress`.

  

  
    ```json title="Example Event Data"
    {
      "type": "model_load.progress",
      "model_instance_id": "openai/gpt-oss-20b",
      "progress": 0.65
    }
    ```
  

`model_load.end` [#model_loadend]

  
    Signals a successfully completed model load.

    
- **`model_instance_id`** (`string`): Unique identifier for the model instance that was loaded.
- **`load_time_seconds`** (`number`): Time taken to load the model in seconds.
- **`type`** (`"model_load.end"`): The type of the event. Always `model_load.end`.

  

  
    ```json title="Example Event Data"
    {
      "type": "model_load.end",
      "model_instance_id": "openai/gpt-oss-20b",
      "load_time_seconds": 12.34
    }
    ```
  

`prompt_processing.start` [#prompt_processingstart]

  
    Signals the start of the model processing a prompt.

    
- **`type`** (`"prompt_processing.start"`): The type of the event. Always `prompt_processing.start`.

  

  
    ```json title="Example Event Data"
    {
      "type": "prompt_processing.start"
    }
    ```
  

`prompt_processing.progress` [#prompt_processingprogress]

  
    Progress of the model processing a prompt.

    
- **`progress`** (`number`): Progress of the prompt processing as a float between `0` and `1`.
- **`type`** (`"prompt_processing.progress"`): The type of the event. Always `prompt_processing.progress`.

  

  
    ```json title="Example Event Data"
    {
      "type": "prompt_processing.progress",
      "progress": 0.5
    }
    ```
  

`prompt_processing.end` [#prompt_processingend]

  
    Signals the end of the model processing a prompt.

    
- **`type`** (`"prompt_processing.end"`): The type of the event. Always `prompt_processing.end`.

  

  
    ```json title="Example Event Data"
    {
      "type": "prompt_processing.end"
    }
    ```
  

`reasoning.start` [#reasoningstart]

  
    Signals the model is starting to stream reasoning content.

    
- **`type`** (`"reasoning.start"`): The type of the event. Always `reasoning.start`.

  

  
    ```json title="Example Event Data"
    {
      "type": "reasoning.start"
    }
    ```
  

`reasoning.delta` [#reasoningdelta]

  
    A chunk of reasoning content. Multiple deltas may arrive.

    
- **`content`** (`string`): Reasoning text fragment.
- **`type`** (`"reasoning.delta"`): The type of the event. Always `reasoning.delta`.

  

  
    ```json title="Example Event Data"
    {
      "type": "reasoning.delta",
      "content": "Need to"
    }
    ```
  

`reasoning.end` [#reasoningend]

  
    Signals the end of the reasoning stream.

    
- **`type`** (`"reasoning.end"`): The type of the event. Always `reasoning.end`.

  

  
    ```json title="Example Event Data"
    {
      "type": "reasoning.end"
    }
    ```
  

`tool_call.start` [#tool_callstart]

  
    Emitted when the model starts a tool call.

    
- **`tool`** (`string`): Name of the tool being called.
- **`provider_info`** (`object`): Information about the tool provider. Discriminated union upon possible provider types.
  - **`Plugin provider info`** (`object`): Present when the tool is provided by a plugin.
    - **`type`** (`"plugin"`): Provider type.
    - **`plugin_id`** (`string`): Identifier of the plugin.
  - **`Ephemeral MCP provider info`** (`object`): Present when the tool is provided by a ephemeral MCP server.
    - **`type`** (`"ephemeral_mcp"`): Provider type.
    - **`server_label`** (`string`): Label of the MCP server.
- **`type`** (`"tool_call.start"`): The type of the event. Always `tool_call.start`.

  

  
    ```json title="Example Event Data"
    {
      "type": "tool_call.start",
      "tool": "model_search",
      "provider_info": {
        "type": "ephemeral_mcp",
        "server_label": "huggingface"
      }
    }
    ```
  

`tool_call.arguments` [#tool_callarguments]

  
    Arguments streamed for the current tool call.

    
- **`tool`** (`string`): Name of the tool being called.
- **`arguments`** (`object`): Arguments passed to the tool. Can have any keys/values depending on the tool definition.
- **`provider_info`** (`object`): Information about the tool provider. Discriminated union upon possible provider types.
  - **`Plugin provider info`** (`object`): Present when the tool is provided by a plugin.
    - **`type`** (`"plugin"`): Provider type.
    - **`plugin_id`** (`string`): Identifier of the plugin.
  - **`Ephemeral MCP provider info`** (`object`): Present when the tool is provided by a ephemeral MCP server.
    - **`type`** (`"ephemeral_mcp"`): Provider type.
    - **`server_label`** (`string`): Label of the MCP server.
- **`type`** (`"tool_call.arguments"`): The type of the event. Always `tool_call.arguments`.

  

  
    ```json title="Example Event Data"
    {
      "type": "tool_call.arguments",
      "tool": "model_search",
      "arguments": {
        "sort": "trendingScore",
        "limit": 1
      },
      "provider_info": {
        "type": "ephemeral_mcp",
        "server_label": "huggingface"
      }
    }
    ```
  

`tool_call.success` [#tool_callsuccess]

  
    Result of the tool call, along with the arguments used.

    
- **`tool`** (`string`): Name of the tool that was called.
- **`arguments`** (`object`): Arguments that were passed to the tool.
- **`output`** (`string`): Raw tool output string.
- **`provider_info`** (`object`): Information about the tool provider. Discriminated union upon possible provider types.
  - **`Plugin provider info`** (`object`): Present when the tool is provided by a plugin.
    - **`type`** (`"plugin"`): Provider type.
    - **`plugin_id`** (`string`): Identifier of the plugin.
  - **`Ephemeral MCP provider info`** (`object`): Present when the tool is provided by a ephemeral MCP server.
    - **`type`** (`"ephemeral_mcp"`): Provider type.
    - **`server_label`** (`string`): Label of the MCP server.
- **`type`** (`"tool_call.success"`): The type of the event. Always `tool_call.success`.

  

  
    ```json title="Example Event Data"
    {
      "type": "tool_call.success",
      "tool": "model_search",
      "arguments": {
        "sort": "trendingScore",
        "limit": 1
      },
      "output": "[{\"type\":\"text\",\"text\":\"Showing first 1 models...\"}]",
      "provider_info": {
        "type": "ephemeral_mcp",
        "server_label": "huggingface"
      }
    }
    ```
  

`tool_call.failure` [#tool_callfailure]

  
    Indicates that the tool call failed.

    
- **`reason`** (`string`): Reason for the tool call failure.
- **`metadata`** (`object`): Metadata about the invalid tool call.
  - **`type`** (`"invalid_name" | "invalid_arguments"`): Type of error that occurred.
  - **`tool_name`** (`string`): Name of the tool that was attempted to be called.
  - **`arguments`** (`object`): *(Optional)* Arguments that were passed to the tool (only present for `invalid_arguments` errors).
  - **`provider_info`** (`object`): *(Optional)* Information about the tool provider (only present for `invalid_arguments` errors).
    - **`type`** (`"plugin" | "ephemeral_mcp"`): Provider type.
    - **`plugin_id`** (`string`): *(Optional)* Identifier of the plugin (when `type` is `"plugin"`).
    - **`server_label`** (`string`): *(Optional)* Label of the MCP server (when `type` is `"ephemeral_mcp"`).
- **`type`** (`"tool_call.failure"`): The type of the event. Always `tool_call.failure`.

  

  
    ```json title="Example Event Data"
    {
      "type": "tool_call.failure",
      "reason": "Cannot find tool with name open_browser.",
      "metadata": {
        "type": "invalid_name",
        "tool_name": "open_browser"
      }
    }
    ```
  

`message.start` [#messagestart]

  
    Signals the model is about to stream a message.

    
- **`type`** (`"message.start"`): The type of the event. Always `message.start`.

  

  
    ```json title="Example Event Data"
    {
      "type": "message.start"
    }
    ```
  

`message.delta` [#messagedelta]

  
    A chunk of message content. Multiple deltas may arrive.

    
- **`content`** (`string`): Message text fragment.
- **`type`** (`"message.delta"`): The type of the event. Always `message.delta`.

  

  
    ```json title="Example Event Data"
    {
      "type": "message.delta",
      "content": "The current"
    }
    ```
  

`message.end` [#messageend]

  
    Signals the end of the message stream.

    
- **`type`** (`"message.end"`): The type of the event. Always `message.end`.

  

  
    ```json title="Example Event Data"
    {
      "type": "message.end"
    }
    ```
  

`error` [#error]

  
    An error occurred during streaming. The final payload will still be sent in `chat.end` with whatever was generated.

    
- **`error`** (`object`): Error information.
  - **`type`** (`"invalid_request" | "unknown" | "mcp_connection_error" | "plugin_connection_error" | "not_implemented" | "model_not_found" | "job_not_found" | "internal_error"`): High-level error type.
  - **`message`** (`string`): Human-readable error message.
  - **`code`** (`string`): *(Optional)* More detailed error code (e.g., validation issue code).
  - **`param`** (`string`): *(Optional)* Parameter associated with the error, if applicable.
- **`type`** (`"error"`): The type of the event. Always `error`.

  

  
    ```json title="Example Event Data"
    {
      "type": "error",
      "error": {
        "type": "invalid_request",
        "message": "\"model\" is required",
        "code": "missing_required_parameter",
        "param": "model"
      }
    }
    ```
  

`chat.end` [#chatend]

  
    Final event containing the full aggregated response, equivalent to the non-streaming `POST /api/v1/chat` response body.

    
- **`result`** (`object`): Final response with `model_instance_id`, `output`, `stats`, and optional `response_id`. See [non-streaming chat docs](/docs/developer/rest/chat) for more details.
- **`type`** (`"chat.end"`): The type of the event. Always `chat.end`.

  

  
    ```json title="Example Event Data"
    {
      "type": "chat.end",
      "result": {
        "model_instance_id": "openai/gpt-oss-20b",
        "output": [
          { "type": "reasoning", "content": "Need to call function." },
          {
            "type": "tool_call",
            "tool": "model_search",
            "arguments": { "sort": "trendingScore", "limit": 1 },
            "output": "[{\"type\":\"text\",\"text\":\"Showing first 1 models...\"}]",
            "provider_info": { "type": "ephemeral_mcp", "server_label": "huggingface" }
          },
          { "type": "message", "content": "The current top‑trending model is..." }
        ],
        "stats": {
          "input_tokens": 329,
          "total_output_tokens": 268,
          "reasoning_output_tokens": 5,
          "tokens_per_second": 43.73,
          "time_to_first_token_seconds": 0.781
        },
        "response_id": "resp_02b2017dbc06c12bfc353a2ed6c2b802f8cc682884bb5716"
      }
    }
    ```
  

  
    `POST /api/v1/chat`

    **Request body**

    
- **`model`** (`string`): Unique identifier for the model to use.
- **`input`** (`string | array<object>`): Message to send to the model.
  - **`Input text`** (`string`): Text content of the message.
  - **`Input object`** (`object`): Object representing a message with additional metadata.
    - **`Text Input`** (`object`): *(Optional)* Text input to provide user messages
      - **`type`** (`"text"`): Type of input item.
      - **`content`** (`string`): Text content of the message.
    - **`Image Input`** (`object`): *(Optional)* Image input to provide user messages
      - **`type`** (`"image"`): Type of input item.
      - **`data_url`** (`string`): Image data as a base64-encoded data URL.
- **`system_prompt`** (`string`): *(Optional)* System message that sets model behavior or instructions.
- **`integrations`** (`array<string | object>`): *(Optional)* List of integrations (plugins, ephemeral MCP servers, etc...) to enable for this request.
  - **`Plugin id`** (`string`): Unique identifier of a plugin to use. Plugins contain `mcp.json` installed MCP servers (id `mcp/<server_label>`). Shorthand for plugin object with no custom configuration.
  - **`Plugin`** (`object`): Specification of a plugin to use. Plugins contain `mcp.json` installed MCP servers (id `mcp/<server_label>`).
    - **`type`** (`"plugin"`): Type of integration.
    - **`id`** (`string`): Unique identifier of the plugin.
    - **`allowed_tools`** (`array<string>`): *(Optional)* List of tool names the model can call from this plugin. If not provided, all tools from the plugin are allowed.
  - **`Ephemeral MCP server specification`** (`object`): Specification of an ephemeral MCP server. Allows defining MCP servers on-the-fly without needing to pre-configure them in your `mcp.json`.
    - **`type`** (`"ephemeral_mcp"`): Type of integration.
    - **`server_label`** (`string`): Label to identify the MCP server.
    - **`server_url`** (`string`): URL of the MCP server.
    - **`allowed_tools`** (`array<string>`): *(Optional)* List of tool names the model can call from this server. If not provided, all tools from the server are allowed.
    - **`headers`** (`object`): *(Optional)* Custom HTTP headers to send with requests to the server.
- **`stream`** (`boolean`): *(Optional)* Whether to stream partial outputs via SSE. Default `false`. See [streaming events](/docs/developer/rest/streaming-events) for more information.
- **`temperature`** (`number`): *(Optional)* Randomness in token selection. 0 is deterministic, higher values increase creativity [0,1].
- **`top_p`** (`number`): *(Optional)* Minimum cumulative probability for the possible next tokens [0,1].
- **`top_k`** (`integer`): *(Optional)* Limits next token selection to top-k most probable tokens.
- **`min_p`** (`number`): *(Optional)* Minimum base probability for a token to be selected for output [0,1].
- **`repeat_penalty`** (`number`): *(Optional)* Penalty for repeating token sequences. 1 is no penalty, higher values discourage repetition.
- **`max_output_tokens`** (`integer`): *(Optional)* Maximum number of tokens to generate.
- **`reasoning`** (`"off" | "low" | "medium" | "high" | "on"`): *(Optional)* Reasoning setting. Will error if the model being used does not support the reasoning setting using. Defaults to the automatically chosen setting for the model.
- **`context_length`** (`integer`): *(Optional)* Number of tokens to consider as context. Higher values recommended for MCP usage.
- **`store`** (`boolean`): *(Optional)* Whether to store the chat. If set, response will return a `"response_id"` field. Default `true`.
- **`previous_response_id`** (`string`): *(Optional)* Identifier of existing response to append to. Must start with `"resp_"`.

  

  
    
      
        
          Request with MCP
        </CodeBlockTabsTrigger>

        
          Request with Images
        </CodeBlockTabsTrigger>
      

      
#### Request with MCP

        ```bash
        curl http://localhost:1234/api/v1/chat \
          -H "Authorization: Bearer $LM_API_TOKEN" \
          -H "Content-Type: application/json" \
          -d '{
            "model": "ibm/granite-4-micro",
            "input": "Tell me the top trending model on hugging face and navigate to https://lmstudio.ai",
            "integrations": [
              {
                "type": "ephemeral_mcp",
                "server_label": "huggingface",
                "server_url": "https://huggingface.co/mcp",
                "allowed_tools": [
                  "model_search"
                ]
              },
              {
                "type": "plugin",
                "id": "mcp/playwright",
                "allowed_tools": [
                  "browser_navigate"
                ]
              }
            ],
            "context_length": 8000,
            "temperature": 0
          }'
        ```
      

      
#### Request with Images

        ```bash
        # Image is a small red square encoded as a base64 data URL
        curl http://localhost:1234/api/v1/chat \
          -H "Authorization: Bearer $LM_API_TOKEN" \
          -H "Content-Type: application/json" \
          -d '{
            "model": "qwen/qwen3-vl-4b",
            "input": [
              {
                "type": "text",
                "content": "Describe this image in two sentences"
              },
              {
                "type": "image",
                "data_url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAFUlEQVR42mP8z8BQz0AEYBxVSF+FABJADveWkH6oAAAAAElFTkSuQmCC"
              }
            ],
            "context_length": 2048,
            "temperature": 0
          }'
        ```
      
    
  

***

  
    **Response fields**

    
- **`model_instance_id`** (`string`): Unique identifier for the loaded model instance that generated the response.
- **`output`** (`array<object>`): Array of output items generated. Each item can be one of three types.
  - **`Message`** (`object`): A text message from the model.
    - **`type`** (`"message"`): Type of output item.
    - **`content`** (`string`): Text content of the message.
  - **`Tool call`** (`object`): A tool call made by the model.
    - **`type`** (`"tool_call"`): Type of output item.
    - **`tool`** (`string`): Name of the tool called.
    - **`arguments`** (`object`): Arguments passed to the tool. Can have any keys/values depending on the tool definition.
    - **`output`** (`string`): Result returned from the tool.
    - **`provider_info`** (`object`): Information about the tool provider.
      - **`type`** (`"plugin" | "ephemeral_mcp"`): Provider type.
      - **`plugin_id`** (`string`): *(Optional)* Identifier of the plugin (when `type` is `"plugin"`).
      - **`server_label`** (`string`): *(Optional)* Label of the MCP server (when `type` is `"ephemeral_mcp"`).
  - **`Reasoning`** (`object`): Reasoning content from the model.
    - **`type`** (`"reasoning"`): Type of output item.
    - **`content`** (`string`): Text content of the reasoning.
  - **`Invalid tool call`** (`object`): An invalid tool call made by the model - due to invalid tool name or tool arguments.
    - **`type`** (`"invalid_tool_call"`): Type of output item.
    - **`reason`** (`string`): Reason why the tool call was invalid.
    - **`metadata`** (`object`): Metadata about the invalid tool call.
      - **`type`** (`"invalid_name" | "invalid_arguments"`): Type of error that occurred.
      - **`tool_name`** (`string`): Name of the tool that was attempted to be called.
      - **`arguments`** (`object`): *(Optional)* Arguments that were passed to the tool (only present for `invalid_arguments` errors).
      - **`provider_info`** (`object`): *(Optional)* Information about the tool provider (only present for `invalid_arguments` errors).
        - **`type`** (`"plugin" | "ephemeral_mcp"`): Provider type.
        - **`plugin_id`** (`string`): *(Optional)* Identifier of the plugin (when `type` is `"plugin"`).
        - **`server_label`** (`string`): *(Optional)* Label of the MCP server (when `type` is `"ephemeral_mcp"`).
- **`stats`** (`object`): Token usage and performance metrics.
  - **`input_tokens`** (`number`): Number of input tokens. Includes formatting, tool definitions, and prior messages in the chat.
  - **`total_output_tokens`** (`number`): Total number of output tokens generated.
  - **`reasoning_output_tokens`** (`number`): Number of tokens used for reasoning.
  - **`tokens_per_second`** (`number`): Generation speed in tokens per second.
  - **`time_to_first_token_seconds`** (`number`): Time in seconds to generate the first token.
  - **`model_load_time_seconds`** (`number`): *(Optional)* Time taken to load the model for this request in seconds. Present only if the model was not already loaded.
- **`response_id`** (`string`): *(Optional)* Identifier of the response for subsequent requests. Starts with `"resp_"`. Present when `store` is `true`.

  

  
    
      
        
          Request with MCP
        </CodeBlockTabsTrigger>

        
          Request with Images
        </CodeBlockTabsTrigger>
      

      
#### Request with MCP

        ```json
        {
          "model_instance_id": "ibm/granite-4-micro",
          "output": [
            {
              "type": "tool_call",
              "tool": "model_search",
              "arguments": {
                "sort": "trendingScore",
                "query": "",
                "limit": 1
              },
              "output": "...",
              "provider_info": {
                "server_label": "huggingface",
                "type": "ephemeral_mcp"
              }
            },
            {
              "type": "message",
              "content": "..."
            },
            {
              "type": "tool_call",
              "tool": "browser_navigate",
              "arguments": {
                "url": "https://lmstudio.ai"
              },
              "output": "...",
              "provider_info": {
                "plugin_id": "mcp/playwright",
                "type": "plugin"
              }
            },
            {
              "type": "message",
              "content": "**Top Trending Model on Hugging Face** ... Below is a quick snapshot of what’s on the landing page ... more details on the model or LM Studio itself!"
            }
          ],
          "stats": {
            "input_tokens": 646,
            "total_output_tokens": 586,
            "reasoning_output_tokens": 0,
            "tokens_per_second": 29.753900615398926,
            "time_to_first_token_seconds": 1.088,
            "model_load_time_seconds": 2.656
          },
          "response_id": "resp_4ef013eba0def1ed23f19dde72b67974c579113f544086de"
        }
        ```
      

      
#### Request with Images

        ```json
        {
          "model_instance_id": "qwen/qwen3-vl-4b",
          "output": [
            {
              "type": "message",
              "content": "This image is a solid, vibrant red square that fills the entire frame, with no discernible texture, pattern, or other elements. It presents a minimalist, uniform visual field of pure red, evoking a sense of boldness or urgency."
            }
          ],
          "stats": {
            "input_tokens": 17,
            "total_output_tokens": 50,
            "reasoning_output_tokens": 0,
            "tokens_per_second": 51.03762685242662,
            "time_to_first_token_seconds": 0.814
          },
          "response_id": "resp_0182bd7c479d7451f9a35471f9c26b34de87a7255856b9a4"
        }
        ```
      
    
  

  
    `GET /api/v1/models`

    This endpoint has no request parameters.
  

  
    ```bash title="Example Request"
    curl http://localhost:1234/api/v1/models \
      -H "Authorization: Bearer $LM_API_TOKEN"
    ```
  

***

  
    **Response fields**

    
- **`models`** (`array`): List of available models (both LLMs and embedding models).
  - **`type`** (`"llm" | "embedding"`): Type of model.
  - **`publisher`** (`string`): Model publisher name.
  - **`key`** (`string`): Unique identifier for the model.
  - **`display_name`** (`string`): Human-readable model name.
  - **`architecture`** (`string | null`): *(Optional)* Model architecture (e.g., "llama", "mistral"). Absent for embedding models.
  - **`quantization`** (`object | null`): Quantization information for the model.
    - **`name`** (`string | null`): Quantization method name.
    - **`bits_per_weight`** (`number | null`): Bits per weight for the quantization.
  - **`size_bytes`** (`number`): Size of the model in bytes.
  - **`params_string`** (`string | null`): Human-readable parameter count (e.g., "7B", "13B").
  - **`loaded_instances`** (`array`): List of currently loaded instances of this model.
    - **`id`** (`string`): Unique identifier for the loaded model instance.
    - **`config`** (`object`): Configuration for the loaded instance.
      - **`context_length`** (`number`): The maximum context length for the model in number of tokens.
      - **`eval_batch_size`** (`number`): *(Optional)* Number of input tokens to process together in a single batch during evaluation. Absent for embedding models.
      - **`parallel`** (`number`): *(Optional)* Maximum number of parallel predictions the instance can handle. Absent for embedding models.
      - **`flash_attention`** (`boolean`): *(Optional)* Whether Flash Attention is enabled for optimized attention computation. Absent for embedding models.
      - **`num_experts`** (`number`): *(Optional)* Number of experts for MoE (Mixture of Experts) models. Absent for embedding models.
      - **`offload_kv_cache_to_gpu`** (`boolean`): *(Optional)* Whether KV cache is offloaded to GPU memory. Absent for embedding models.
  - **`max_context_length`** (`number`): Maximum context length supported by the model in number of tokens.
  - **`format`** (`"gguf" | "mlx" | null`): Model file format.
  - **`capabilities`** (`object`): *(Optional)* Model capabilities. Absent for embedding models.
    - **`vision`** (`boolean`): Whether the model supports vision/image inputs.
    - **`trained_for_tool_use`** (`boolean`): Whether the model was trained for tool/function calling.
    - **`reasoning`** (`object`): *(Optional)* Public reasoning configuration for the model. Absent when no reasoning config is exposed.
      - **`allowed_options`** (`("off" | "on" | "low" | "medium" | "high")[]`): Allowed public reasoning settings for the model.
      - **`default`** (`"off" | "on" | "low" | "medium" | "high"`): Default public reasoning setting for the model.
  - **`description`** (`string | null`): *(Optional)* Model description. Absent for embedding models.
  - **`variants`** (`array`): *(Optional)* List of available quantization variant names for this model. Present for multi-variant models.
  - **`selected_variant`** (`string`): *(Optional)* The currently selected variant name. Present when `variants` is present.

  

  
    ```json title="Response"
    {
      "models": [
        {
          "type": "llm",
          "publisher": "google",
          "key": "google/gemma-4-26b-a4b",
          "display_name": "Gemma 4 26B A4B",
          "architecture": "gemma4",
          "quantization": {
            "name": "Q4_K_M",
            "bits_per_weight": 4
          },
          "size_bytes": 17990911801,
          "params_string": "26B-A4B",
          "loaded_instances": [
            {
              "id": "google/gemma-4-26b-a4b",
              "config": {
                "context_length": 4096,
                "eval_batch_size": 512,
                "parallel": 4,
                "flash_attention": true,
                "num_experts": 8,
                "offload_kv_cache_to_gpu": true
              }
            }
          ],
          "max_context_length": 262144,
          "format": "gguf",
          "capabilities": {
            "vision": true,
            "trained_for_tool_use": true,
            "reasoning": {
              "allowed_options": [
                "off",
                "on"
              ],
              "default": "on"
            }
          },
          "description": null,
          "variants": [
            "google/gemma-4-26b-a4b@q4_k_m"
          ],
          "selected_variant": "google/gemma-4-26b-a4b@q4_k_m"
        },
          {
            "type": "llm",
            "publisher": "deepseek",
            "key": "deepseek-r1",
            "display_name": "DeepSeek R1",
            "architecture": "deepseek",
            "quantization": {
              "name": "Q4_K_M",
              "bits_per_weight": 4
            },
            "size_bytes": 40492610355,
            "params_string": "671B",
            "loaded_instances": [],
            "max_context_length": 131072,
            "format": "gguf",
            "capabilities": {
              "vision": false,
              "trained_for_tool_use": true,
              "reasoning": {
                "allowed_options": ["on"],
                "default": "on"
              }
            },
            "description": null
          },
          {
            "type": "embedding",
            "publisher": "gaianet",
            "key": "text-embedding-nomic-embed-text-v1.5-embedding",
            "display_name": "Nomic Embed Text v1.5",
            "quantization": {
              "name": "F16",
              "bits_per_weight": 16
            },
            "size_bytes": 274290560,
            "params_string": null,
            "loaded_instances": [],
            "max_context_length": 2048,
            "format": "gguf"
          }
      ]
    }
    ```
  

  
    `POST /api/v1/models/load`

    **Request body**

    
- **`model`** (`string`): Unique identifier for the model to load. Can be an LLM or embedding model.
- **`context_length`** (`number`): *(Optional)* Maximum number of tokens that the model will consider.
- **`eval_batch_size`** (`number`): *(Optional)* Number of input tokens to process together in a single batch during evaluation. Will only have an effect on LLMs loaded by LM Studio's [llama.cpp](https://github.com/ggml-org/llama.cpp)-based engine.
- **`flash_attention`** (`boolean`): *(Optional)* Whether to optimize attention computation. Can decrease memory usage and improved generation speed. Will only have an effect on LLMs loaded by LM Studio's [llama.cpp](https://github.com/ggml-org/llama.cpp)-based engine.
- **`num_experts`** (`number`): *(Optional)* Number of expert to use during inference for MoE (Mixture of Experts) models. Will only have an effect on MoE LLMs loaded by LM Studio's [llama.cpp](https://github.com/ggml-org/llama.cpp)-based engine.
- **`offload_kv_cache_to_gpu`** (`boolean`): *(Optional)* Whether KV cache is offloaded to GPU memory. If false, KV cache is stored in CPU memory/RAM. Will only have an effect on LLMs loaded by LM Studio's [llama.cpp](https://github.com/ggml-org/llama.cpp)-based engine.
- **`echo_load_config`** (`boolean`): *(Optional)* If true, echoes the final load configuration in the response under `"load_config"`. Default `false`.

  

  
    ```bash title="Example Request"
    curl http://localhost:1234/api/v1/models/load \
      -H "Authorization: Bearer $LM_API_TOKEN" \
      -H "Content-Type: application/json" \
      -d '{
        "model": "openai/gpt-oss-20b",
        "context_length": 16384,
        "flash_attention": true,
        "echo_load_config": true
      }'
    ```
  

***

  
    **Response fields**

    
- **`type`** (`"llm" | "embedding"`): Type of the loaded model.
- **`instance_id`** (`string`): Unique identifier for the loaded model instance.
- **`load_time_seconds`** (`number`): Time taken to load the model in seconds.
- **`status`** (`"loaded"`): Load status.
- **`load_config`** (`object`): *(Optional)* The final configuration applied to the loaded model. This may include settings that were not specified in the request. Included only when `"echo_load_config"` is `true` in the request.
  - **`LLM load config`** (`object`): Configuration parameters specific to LLM models. `load_config` will be this type when `"type"` is `"llm"`. Only parameters that applied to the load will be present.
    - **`context_length`** (`number`): Maximum number of tokens that the model will consider.
    - **`eval_batch_size`** (`number`): *(Optional)* Number of input tokens to process together in a single batch during evaluation. Only present for models loaded with LM Studio's [llama.cpp](https://github.com/ggml-org/llama.cpp)-based engine.
    - **`flash_attention`** (`boolean`): *(Optional)* Whether Flash Attention is enabled for optimized attention computation. Only present for models loaded with LM Studio's [llama.cpp](https://github.com/ggml-org/llama.cpp)-based engine.
    - **`num_experts`** (`number`): *(Optional)* Number of experts for MoE (Mixture of Experts) models. Only present for MoE models loaded with LM Studio's [llama.cpp](https://github.com/ggml-org/llama.cpp)-based engine.
    - **`offload_kv_cache_to_gpu`** (`boolean`): *(Optional)* Whether KV cache is offloaded to GPU memory. Only present for models loaded with LM Studio's [llama.cpp](https://github.com/ggml-org/llama.cpp)-based engine.
  - **`Embedding model load config`** (`object`): Configuration parameters specific to embedding models. `load_config` will be this type when `"type"` is `"embedding"`. Only parameters that applied to the load will be present.
    - **`context_length`** (`number`): Maximum number of tokens that the model will consider.

  

  
    ```json title="Response"
    {
      "type": "llm",
      "instance_id": "openai/gpt-oss-20b",
      "load_time_seconds": 9.099,
      "status": "loaded",
      "load_config": {
        "context_length": 16384,
        "eval_batch_size": 512,
        "flash_attention": true,
        "offload_kv_cache_to_gpu": true,
        "num_experts": 4
      }
    }
    ```
  

  
    `POST /api/v1/models/download`

    **Request body**

    
- **`model`** (`string`): The model to download. Accepts [model catalog](https://lmstudio.ai/models) identifiers (e.g., `openai/gpt-oss-20b`) and exact Hugging Face links (e.g., `https://huggingface.co/lmstudio-community/gpt-oss-20b-GGUF`)
- **`quantization`** (`string`): *(Optional)* Quantization level of the model to download (e.g., `Q4_K_M`). Only supported for Hugging Face links.

  

  
    ```bash title="Example Request"
    curl http://localhost:1234/api/v1/models/download \
      -H "Authorization: Bearer $LM_API_TOKEN" \
      -H "Content-Type: application/json" \
      -d '{
        "model": "ibm/granite-4-micro"
      }'
    ```
  

  
    **Response fields**

    Returns a download job status object. The response varies based on the download status.

    
- **`job_id`** (`string`): *(Optional)* Unique identifier for the download job. Absent when `status` is `already_downloaded`.
- **`status`** (`"downloading" | "paused" | "completed" | "failed" | "already_downloaded"`): Current status of the download.
- **`completed_at`** (`string`): *(Optional)* Download completion time in ISO 8601 format. Present when `status` is `completed`.
- **`total_size_bytes`** (`number`): *(Optional)* Total size of the download in bytes. Absent when `status` is `already_downloaded`.
- **`started_at`** (`string`): *(Optional)* Download start time in ISO 8601 format. Absent when `status` is `already_downloaded`.

  

  
    ```json title="Response"
    {
      "job_id": "job_493c7c9ded",
      "status": "downloading",
      "total_size_bytes": 2279145003,
      "started_at": "2025-10-03T15:33:23.496Z"
    }
    ```
  

  
    `POST /api/v1/models/unload`

    **Request body**

    
- **`instance_id`** (`string`): Unique identifier of the model instance to unload.

  

  
    ```bash title="Example Request"
    curl http://localhost:1234/api/v1/models/unload \
      -H "Authorization: Bearer $LM_API_TOKEN" \
      -H "Content-Type: application/json" \
      -d '{
        "instance_id": "openai/gpt-oss-20b"
      }'
    ```
  

***

  
    **Response fields**

    
- **`instance_id`** (`string`): Unique identifier for the unloaded model instance.

  

  
    ```json title="Response"
    {
      "instance_id": "openai/gpt-oss-20b"
    }
    ```
  

  
    `GET /api/v1/models/download/status/:job_id`

    **Path parameters**

    
- **`job_id`** (`string`): The unique identifier of the download job. `job_id` is returned by the [download](/docs/developer/rest/download) endpoint when a download is initiated.

  

  
    ```bash title="Example Request"
    curl -H "Authorization: Bearer $LM_API_TOKEN" \
      http://localhost:1234/api/v1/models/download/status/job_493c7c9ded
    ```
  

  
    **Response fields**

    Returns a single download job status object. The response varies based on the download status.

    
- **`job_id`** (`string`): Unique identifier for the download job.
- **`status`** (`"downloading" | "paused" | "completed" | "failed"`): Current status of the download.
- **`bytes_per_second`** (`number`): *(Optional)* Current download speed in bytes per second. Present when `status` is `downloading`.
- **`estimated_completion`** (`string`): *(Optional)* Estimated completion time in ISO 8601 format. Present when `status` is `downloading`.
- **`completed_at`** (`string`): *(Optional)* Download completion time in ISO 8601 format. Present when `status` is `completed`.
- **`total_size_bytes`** (`number`): *(Optional)* Total size of the download in bytes.
- **`downloaded_bytes`** (`number`): *(Optional)* Number of bytes downloaded so far.
- **`started_at`** (`string`): *(Optional)* Download start time in ISO 8601 format.

  

  
    ```json title="Response"
    {
      "job_id": "job_493c7c9ded",
      "status": "completed",
      "total_size_bytes": 2279145003,
      "downloaded_bytes": 2279145003,
      "started_at": "2025-10-03T15:33:23.496Z",
      "completed_at": "2025-10-03T15:43:12.102Z"
    }
    ```
  

> [!WARNING]
> **Heads Up**
> LM Studio now has a [v1 REST API](/docs/developer/rest)! We recommend using the v1 API for new projects!

Requires LM Studio 0.3.6 or newer. [#requires-lm-studio-036-or-newer]

LM Studio now has its own REST API, in addition to OpenAI-compatible endpoints ([learn more](/docs/developer/openai-compat)) and Anthropic-compatible endpoints ([learn more](/docs/developer/anthropic-compat)).

The REST API includes enhanced stats such as Token / Second and Time To First Token (TTFT), as well as rich information about models such as loaded vs unloaded, max context, quantization, and more.

Supported API Endpoints [#supported-api-endpoints]

* [`GET /api/v0/models`](#get-apiv0models) - List available models
* [`GET /api/v0/models/{model}`](#get-apiv0modelsmodel) - Get info about a specific model
* [`POST /api/v0/chat/completions`](#post-apiv0chatcompletions) - Chat Completions (messages -> assistant response)
* [`POST /api/v0/completions`](#post-apiv0completions) - Text Completions (prompt -> completion)
* [`POST /api/v0/embeddings`](#post-apiv0embeddings) - Text Embeddings (text -> embedding)

***

Start the REST API server [#start-the-rest-api-server]

To start the server, run the following command:

```bash
lms server start
```

> [!TIP]
> **Pro Tip**
> You can run LM Studio as a service and get the server to auto-start on boot without launching the GUI. [Learn about Headless Mode](/docs/developer/core/headless).

Endpoints [#endpoints]

`GET /api/v0/models` [#get-apiv0models]

List all loaded and downloaded models

**Example request**

```bash
curl -H "Authorization: Bearer $LM_API_TOKEN" http://localhost:1234/api/v0/models
```

**Response format**

```json
{
  "object": "list",
  "data": [
    {
      "id": "qwen2-vl-7b-instruct",
      "object": "model",
      "type": "vlm",
      "publisher": "mlx-community",
      "arch": "qwen2_vl",
      "compatibility_type": "mlx",
      "quantization": "4bit",
      "state": "not-loaded",
      "max_context_length": 32768
    },
    {
      "id": "meta-llama-3.1-8b-instruct",
      "object": "model",
      "type": "llm",
      "publisher": "lmstudio-community",
      "arch": "llama",
      "compatibility_type": "gguf",
      "quantization": "Q4_K_M",
      "state": "not-loaded",
      "max_context_length": 131072
    },
    {
      "id": "text-embedding-nomic-embed-text-v1.5",
      "object": "model",
      "type": "embeddings",
      "publisher": "nomic-ai",
      "arch": "nomic-bert",
      "compatibility_type": "gguf",
      "quantization": "Q4_0",
      "state": "not-loaded",
      "max_context_length": 2048
    }
  ]
}
```

***

`GET /api/v0/models/{model}` [#get-apiv0modelsmodel]

Get info about one specific model

**Example request**

```bash
curl -H "Authorization: Bearer $LM_API_TOKEN" http://localhost:1234/api/v0/models/qwen2-vl-7b-instruct
```

**Response format**

```json
{
  "id": "qwen2-vl-7b-instruct",
  "object": "model",
  "type": "vlm",
  "publisher": "mlx-community",
  "arch": "qwen2_vl",
  "compatibility_type": "mlx",
  "quantization": "4bit",
  "state": "not-loaded",
  "max_context_length": 32768
}
```

***

`POST /api/v0/chat/completions` [#post-apiv0chatcompletions]

Chat Completions API. You provide a messages array and receive the next assistant response in the chat.

**Example request**

```bash
curl http://localhost:1234/api/v0/chat/completions \
  -H "Authorization: Bearer $LM_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "granite-3.0-2b-instruct",
    "messages": [
      { "role": "system", "content": "Always answer in rhymes." },
      { "role": "user", "content": "Introduce yourself." }
    ],
    "temperature": 0.7,
    "max_tokens": -1,
    "stream": false
  }'
```

**Response format**

```json
{
  "id": "chatcmpl-i3gkjwthhw96whukek9tz",
  "object": "chat.completion",
  "created": 1731990317,
  "model": "granite-3.0-2b-instruct",
  "choices": [
    {
      "index": 0,
      "logprobs": null,
      "finish_reason": "stop",
      "message": {
        "role": "assistant",
        "content": "Greetings, I'm a helpful AI, here to assist,\nIn providing answers, with no distress.\nI'll keep it short and sweet, in rhyme you'll find,\nA friendly companion, all day long you'll bind."
      }
    }
  ],
  "usage": {
    "prompt_tokens": 24,
    "completion_tokens": 53,
    "total_tokens": 77
  },
  "stats": {
    "tokens_per_second": 51.43709529007664,
    "time_to_first_token": 0.111,
    "generation_time": 0.954,
    "stop_reason": "eosFound"
  },
  "model_info": {
    "arch": "granite",
    "quant": "Q4_K_M",
    "format": "gguf",
    "context_length": 4096
  },
  "runtime": {
    "name": "llama.cpp-mac-arm64-apple-metal-advsimd",
    "version": "1.3.0",
    "supported_formats": ["gguf"]
  }
}
```

***

`POST /api/v0/completions` [#post-apiv0completions]

Text Completions API. You provide a prompt and receive a completion.

**Example request**

```bash
curl http://localhost:1234/api/v0/completions \
  -H "Authorization: Bearer $LM_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "granite-3.0-2b-instruct",
    "prompt": "the meaning of life is",
    "temperature": 0.7,
    "max_tokens": 10,
    "stream": false,
    "stop": "\n"
  }'
```

**Response format**

```json
{
  "id": "cmpl-p9rtxv6fky2v9k8jrd8cc",
  "object": "text_completion",
  "created": 1731990488,
  "model": "granite-3.0-2b-instruct",
  "choices": [
    {
      "index": 0,
      "text": " to find your purpose, and once you have",
      "logprobs": null,
      "finish_reason": "length"
    }
  ],
  "usage": {
    "prompt_tokens": 5,
    "completion_tokens": 9,
    "total_tokens": 14
  },
  "stats": {
    "tokens_per_second": 57.69230769230769,
    "time_to_first_token": 0.299,
    "generation_time": 0.156,
    "stop_reason": "maxPredictedTokensReached"
  },
  "model_info": {
    "arch": "granite",
    "quant": "Q4_K_M",
    "format": "gguf",
    "context_length": 4096
  },
  "runtime": {
    "name": "llama.cpp-mac-arm64-apple-metal-advsimd",
    "version": "1.3.0",
    "supported_formats": ["gguf"]
  }
}
```

***

`POST /api/v0/embeddings` [#post-apiv0embeddings]

Text Embeddings API. You provide a text and a representation of the text as an embedding vector is returned.

**Example request**

```bash
curl http://localhost:1234/api/v0/embeddings \
  -H "Authorization: Bearer $LM_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "text-embedding-nomic-embed-text-v1.5",
    "input": "Some text to embed"
  }
```

**Example response**

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [
        -0.016731496900320053,
        0.028460891917347908,
        -0.1407836228609085,
        ... (truncated for brevity) ...,
        0.02505224384367466,
        -0.0037634256295859814,
        -0.04341062530875206
      ],
      "index": 0
    }
  ],
  "model": "text-embedding-nomic-embed-text-v1.5@q4_k_m",
  "usage": {
    "prompt_tokens": 0,
    "total_tokens": 0
  }
}
```

***

Please report bugs by opening an issue on [Github](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues).

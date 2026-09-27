# Third-party notices

## AgentDojo 0.1.35

The harness installs AgentDojo 0.1.35 as a pinned, unmodified dependency
(`requirements.txt`); the package itself is not redistributed here. Several
functions in this repository reproduce or adapt source code from that release,
and those portions remain subject to the AgentDojo MIT License reproduced below.

- Project: AgentDojo (https://github.com/ethz-spylab/agentdojo)
- Version: 0.1.35
- License: MIT
- Copyright (c) 2024 Edoardo Debenedetti, Jie Zhang, Mislav Balunovic, Luca
  Beurer-Kellner, Marc Fischer, and Florian Tramèr

### Adapted code

| File in this repository | Function or class | Source in AgentDojo 0.1.35 | Relationship |
|---|---|---|---|
| `src/adapters_baseten.py` | `_corrected_request` | `agentdojo/agent_pipeline/llms/openai_llm.py`: `chat_completion_request` | Retry decorator and request body reproduced; the temperature argument is changed from `temperature or NOT_GIVEN` to `temperature if temperature is not None else NOT_GIVEN`. |
| `src/adapters_openai.py` | `_responses_request` | `agentdojo/agent_pipeline/llms/openai_llm.py`: `chat_completion_request` | Retry decorator reproduced; the request is rewritten for the OpenAI Responses API. |
| `src/adapters_openai.py` | `_messages_to_responses_input` | `agentdojo/agent_pipeline/llms/openai_llm.py`: `_message_to_openai`, `_tool_call_to_openai` | Role dispatch, tool-call argument serialisation and error handling adapted to Responses API input items. |
| `src/adapters_openai.py` | `_function_to_responses_tool` | `agentdojo/agent_pipeline/llms/openai_llm.py`: `_function_to_openai` | The same fields, emitted in the flat Responses tool shape. |
| `src/adapters_openai.py` | `_response_to_assistant_message` | `agentdojo/agent_pipeline/llms/openai_llm.py`: `_openai_to_assistant_message`, `_openai_to_tool_call` | The same message construction, reading Responses output items. |
| `src/adapters_openai.py` | `ResponsesLLM` (`__init__`, `query`) | `agentdojo/agent_pipeline/llms/openai_llm.py`: `OpenAILLM` | The same pipeline-element structure and `query` flow, calling the Responses API. |
| `src/run_sweep.py` | `QuietLogger.__enter__`, `QuietLogger.__exit__` | `agentdojo/logging.py`: `Logger.__enter__`, `Logger.__exit__` | The same logger-stack push and pop; listed for completeness. |

The other adapter modules, `src/adapters.py` and `src/adapters_anthropic.py`,
wrap AgentDojo functions at run time and call the originals; they do not
reproduce AgentDojo source.

### AgentDojo license

The text below is reproduced byte for byte from
`agentdojo-0.1.35.dist-info/licenses/LICENSE` of the installed package.

```text
MIT License

Copyright (c) 2024 Edoardo Debenedetti, Jie Zhang, Mislav Balunovic, Luca Beurer-Kellner, Marc Fischer, and Florian Tramèr

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

# Generative Agents for Social Behavior Simulation

This code marks the implementation part of the master thesis **["Generative agents for simulation of social behaviour"](https://dspace.cuni.cz/handle/20.500.11956/209608)**.

This library provides tools and frameworks for using generative agents to simulate social behavior. It is designed to enhance traditional rule-based agent paradigms with large language models, enabling the creation of agents with memory, action selection mechanisms, and reasoning capabilities.

## Features

- **Generative Agents**: Create agents with memory and reasoning capabilities.
- **Social Behavior Simulation**: Simulate communication and interactions in small groups.
- **Experimentation**: Run experiments with real-world data based on sociological surveys.

## Installation

### Prerequisites

- Python 3.13 or higher

This example uses `uv`, an extremely fast python package manager, but feel free to use your favorite as `pip` etc.

### Steps

1. **Create and activate a virtual environment**:

   ```bash
   uv venv venv --seed
   source venv/bin/activate
   ```

2. **Install the library**:
   - For editable mode (development):

     ```bash
     uv pip install -e .
     ```

   - For regular installation:

     ```bash
     uv pip install .
     ```

3. **Install dependencies for experiments**:

   ```bash
   uv pip install -e .[dev]
   ```

4. **Install sentence-transformers for local embedding support (optional)**:

   ```bash
   uv pip install .[embedding]
   ```

## Configuration

- **Environment Variables**: Optionally, configure environment variables in the `.env` file to match the naming used in your experiments. This is only required for running the example experiments.

## Running a Sample Experiment

You can run a sample experiment using the following command:

```bash
python experiments/valentine_party.py
```

Sample experiments are located in the `experiments` directory.

### Overriding the prompts

You can override the default provided prompts from the model by subclassing the `DefaultPromptBuilder`, replacing the methods and overriding the config by provided decorator.

```python
from generative_agents import DefaultPromptBuilder, default_builder

class CustomPromptBuilder(DefaultPromptBuilder):
    ...

@default_builder.override(CustomPromptBuilder())
async def main():
    ...

with default_builder.override(CustomPromptBuilder()):
    ...
```

### Configuring LLM Parameters

The framework provides three parameter sets out of the box:

- `factual_llm_params` - lower temperature (0.3) for deterministic, factual responses
- `neutral_default_llm_params` - balanced temperature (0.6) for general use
- `creative_llm_params` - higher temperature (1.0) for creative generation

You can customize them in several ways, from most to least convenient:

#### 1. Constructor Parameters (Recommended)

Pass custom params when constructing `DefaultConfig` (or its subclass):

```python
from generative_agents import DefaultConfig, create_completion_params

config = DefaultConfig(
    creative_llm_params=create_completion_params(temperature=0.8, top_p=0.95),
    factual_llm_params=create_completion_params(temperature=0.4),
)
```

#### 2. Via OverridableContextVar

Override the global config for a scoped block:

```python
from generative_agents import default_config, DefaultConfig, create_completion_params

custom_config = DefaultConfig(
    creative_llm_params=create_completion_params(temperature=0.9)
)

with default_config.override(custom_config):
    # All agents created here use the overridden params
    agent = LLMConversationAgent(...)
```

#### 3. Subclassing

Create a subclass to set custom defaults:

```python
from generative_agents import DefaultConfig, create_completion_params

class WarmCreativeConfig(DefaultConfig):
    def __init__(self):
        super().__init__(
            creative_llm_params=create_completion_params(
                temperature=0.9, top_p=1.0, frequency_penalty=0.5
            ),
        )
```

Then override using `OverridableContextVar`.

#### 4. Direct Property Access

Modify parameters on an existing instance:

```python
from generative_agents import default_config, create_completion_params

# Access current parameters
current_creative = default_config().creative_llm_params
current_factual = default_config().factual_llm_params

# Modify parameters directly
default_config().creative_llm_params = create_completion_params(
    temperature=0.5, top_p=1.0
)
```

## Example Experiment: Valentine Party

The `valentine_party.py` experiment demonstrates how information spreads through a social network of generative agents. Below is a simplified walkthrough of the key concepts.
Other experiments are defined in `experiments/` folder.

### 1. Define Your Agent Model

Subclass `AgentModelBase` to define the attributes of your agents:

```python
from generative_agents import AgentModelBase
from pydantic import Field
from typing import Literal

class ExperimentAgent(AgentModelBase):
    first_name: str = Field(..., description="First name")
    last_name: str = Field(..., description="Last name")
    sex: Literal["F", "M"] = Field(..., description="Sex")
    description: str = Field(..., description="Agent characteristics and description")

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"
```

### 2. Create an LLM Backend

The backend handles all LLM and embedding calls:

```python
from generative_agents import LLMBackend, OpenAIEmbeddingProvider
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key="...")
context = LLMBackend(
    client=client,
    model="...",
    RPS=10,
    embedding_provider=OpenAIEmbeddingProvider(client=client, model="..."),
)
```

### 3. Configure Agent Behaviors

Agents can be composed with multiple behaviors such as memory updating, BDI planning, and memory forgetting:

```python
from generative_agents import (
    CompositeBehaviorMemoryManager,
    ConversationMemoryUpdatingBehavior,
    BDIPlanningBehavior,
    ConversationMemoryForgettingBehavior,
    EmbeddingMemory,
    get_record_removal_linear_probability,
    mean_std_count_strategy_factory,
)

behaviors = [
    ConversationMemoryUpdatingBehavior(),
    BDIPlanningBehavior(),
    ConversationMemoryForgettingBehavior(
        get_record_removal_linear_probability(0.5), seed=seed
    ),
]

agent = LLMConversationAgent(
    data,
    context,
    lambda agent: CompositeBehaviorMemoryManager(
        EmbeddingMemory(context, count_selector=mean_std_count_strategy_factory(0.5)),
        agent,
        context,
        behaviors,
    ),
)
```

### 4. Build a Social Network

Use `networkx` to define the social graph that determines who can talk to whom:

```python
import networkx as nx

graph = nx.Graph()
graph.add_edges_from([(agent_a, agent_b), (agent_b, agent_c)])
```

### 5. Run the Simulation

Use `ConversationManager` and a conversation selector to drive the simulation:

```python
from generative_agents import ConversationManager, SequentialConversationSelector

selector = SequentialConversationSelector(structure=graph, seed=seed, initial_conversation=[(agent_a, agent_b)])
manager = ConversationManager(conversation_selector=selector, max_conversation_utterances=4)

for epoch in range(2):
    await manager.run_simulation_epoch()

# Query agents afterward
answer = await agent.ask_agent("When is the party happening?")
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or suggestions, please open an issue or contact the maintainers.

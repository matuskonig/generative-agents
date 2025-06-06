# Generative Agents for Social Behavior Simulation

This library provides tools and frameworks for using generative agents to simulate social behavior. It is designed to enhance traditional rule-based agent paradigms with large language models, enabling the creation of agents with memory, action selection mechanisms, and reasoning capabilities.

## Features

- **Generative Agents**: Create agents with memory and reasoning capabilities.
- **Social Behavior Simulation**: Simulate communication and interactions in small groups.
- **Experimentation**: Run experiments with real-world data based on sociological surveys.

## Installation

### Prerequisites

- Python 3.12 or higher

This example uses `uv`, an extremely fast python package manager, but feel free to
use your favourite as `pip` etc.

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

## Configuration

- **Environment Variables**: Optionally, configure environment variables in the `.env` file to match the naming used in your experiments. This is only required for running the example experiments.

## Running a Sample Experiment

You can run a sample experiment using the following command:

```bash
python experiments/valentine_party.py
```

Sample experiments are located in the `experiments` directory.

### Overriding the prompts
You can override the default provided prompts from the model by subclassing the `DefaultPromptBuilder`, replacing the methods 
and overriding the config by provided decorator.

```
@default_builder.override(DefaultPromptBuilder())
async def main():
   ...

with default_builder.override(DefaultPromptBuilder()):
   ...
```


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or suggestions, please open an issue or contact the maintainers.

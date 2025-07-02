import networkx as nx
import pydantic
from openai import AsyncOpenAI
from typing import Literal, Mapping
from generative_agents import AgentModelBase


class SocietyAgent(AgentModelBase):
    first_name: str = pydantic.Field(description="First name")
    last_name: str = pydantic.Field(description="Last name")
    sex: Literal["F", "M"] = pydantic.Field(description="Sex")
    age: int = pydantic.Field(description="Age of the agent in years")
    nationality: str = pydantic.Field(
        description="Nationality of the agent, e.g. 'American', 'Russian', 'Chinese'"
    )
    education: str = pydantic.Field(
        description="Education of the agent, e.g. 'PhD in Computer Science', 'High School Diploma'",
    )
    political_views: str = pydantic.Field(description="Political views of the agent")
    religion: str = pydantic.Field(
        description="Religion of the agent, e.g. 'Christianity', 'Islam', 'Atheism'",
    )
    family_status: str = pydantic.Field(
        description="Family status of the agent, e.g. 'Single', 'Married with children', 'Divorced'",
    )
    location: str = pydantic.Field(
        description="Where the agent lives, e.g. 'house in the suburbs', 'apartment in the city center'. Omit exact geographical place.",
    )
    hobbies: str = pydantic.Field(
        description="Hobbies of the agent, e.g. 'Reading', 'Sports', 'Traveling', 'Cooking'",
    )
    occupation: str = pydantic.Field(
        description="Occupation of the agent, e.g. 'Software Engineer', 'Doctor', 'Construction Worker'",
    )
    description: str = pydantic.Field(
        description="Agent characteristics and description"
    )

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"

    @property
    def agent_characteristics(self) -> str:
        return self.model_dump_json()


async def generate_agents[T](
    graph: nx.Graph,
    centrality_mapping: Mapping[T, float],
    client: AsyncOpenAI,
    model: str,
):
    assert nx.is_connected(graph), "Graph must be connected for agent generation."

    result_mapping: dict[T, SocietyAgent] = {}

    async def generate_single(node):
        established_neighbors = [
            result_mapping[neighbor].model_dump_json()
            for neighbor in graph.neighbors(node)
            if neighbor in result_mapping
        ]
        generated_samples = [
            {
                "name": agent.full_name,
                "age": agent.age,
                "sex": agent.sex,
                # "occupation": agent.occupation,
            }
            for agent in result_mapping.values()
        ]

        prompt = f"""You are tasked with generating a detailed agent persona for a social network simulation.

CONTEXT:
- This agent will have {sum(1 for _ in graph.neighbors(node))} connections in the network
- Already established neighbors: {established_neighbors}
- These agents will engage in conversations and exchange information in future experiments
- Additional agents will be generated until the entire network is populated
- The network currently consists of: {generated_samples}

REQUIREMENTS:
Generate a diverse, realistic agent that will:
- Use their background and characteristics during conversations
- Have distinct opinions and perspectives
- Form believable social connections with neighbors
- Naturally participate in information sharing within their social network
- Avoid using names that already exist in the network
- Use common names without excessive diversity
- Mix nationalities slightly, simulating a realistic social network

DIVERSITY GUIDELINES:
- Embrace variety in demographics, backgrounds, and viewpoints
- Consider how real social connections form: through family, work, education, hobbies, or chance encounters
- Make connections feel authentic and purposeful
- Avoid stereotypes while maintaining realistic characteristics
- Ensure reasonable diversity to make the social network believable

PERSONA DETAILS TO CONSIDER:
- Personal background and life experiences
- Professional occupation and career history
- Educational background and skills
- Social network and relationship patterns
- Interests, hobbies, and lifestyle choices
- Values, beliefs, and worldview
- Geographic and cultural context (keep locations generic)

RESPONSE FORMAT:
Provide detailed, nuanced descriptions where applicable. The richer the persona, the more authentic future conversations and information sharing will be.

Generate the agent using this schema: {SocietyAgent.model_json_schema()}"""

        result = await client.beta.chat.completions.parse(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format=SocietyAgent,
            temperature=1.5,
            frequency_penalty=0.8,
        )
        response = result.choices[0].message.parsed
        assert response is not None, "Response parsing failed."
        return response

    (init_node, v) = max(centrality_mapping.items(), key=lambda item: item[1])

    result_mapping[init_node] = await generate_single(init_node)
    while len(result_mapping) < len(graph.nodes):
        possible_choices = {
            node
            for generated_node in result_mapping
            for node in graph.neighbors(generated_node)
            if node not in result_mapping
        }
        # Extend the node with max centrality
        extension_node = max(possible_choices, key=lambda n: centrality_mapping[n])
        result_mapping[extension_node] = await generate_single(extension_node)
    return result_mapping


async def create_information_seed[T](
    graph: nx.Graph,
    agent_mapping: dict[T, SocietyAgent],
    seed_node: T,
    client: AsyncOpenAI,
    model: str,
):
    generated_samples = [
        {
            "name": agent.full_name,
            "age": agent.age,
            "sex": agent.sex,
            "occupation": agent.occupation,
        }
        for agent in agent_mapping.values()
    ]

    class InformationSeedResponse(pydantic.BaseModel):
        information_injected: str
        agent: SocietyAgent

    prompt = f"""You are tasked with injecting specific information into a social network agent to study how information spreads through conversations.

TARGET AGENT:
{agent_mapping[seed_node]}

CONNECTED AGENTS:
{[agent_mapping[node].model_dump_json() for node in graph.neighbors(seed_node)]}

NETWORK AGENTS:
{generated_samples}

OBJECTIVE:
Modify the target agent to possess specific information that can naturally spread through conversations with connected agents. This information will be tracked as it propagates through the social network.

INFORMATION TYPES (choose one that best fits the agent):
1. Event Organization: Planning a party, community event, or gathering
2. Major Life Development: Starting a business, political campaign, or significant project  
3. Personal News: Sharing a secret, family update, or important personal development
4. Professional Announcement: Job change, promotion, or career milestone
5. Community Information: Local news, opportunities, or important community updates

REQUIREMENTS:
- The information should feel natural for this specific agent to possess and share
- It should be conversation-worthy and likely to be discussed with others
- Include specific, memorable details that can be tracked as the information spreads
- Integrate the information seamlessly into the agent's updated description
- Maintain the agent's core personality while adding this new informational element

CONVERSATION CONTEXT:
This agent will discuss this information in natural conversations with neighbors. The information should:
- Be relevant and interesting to the agent's social circle
- Motivate natural sharing behavior during conversations
- Be specific enough to track its spread through the network
- Feel authentic to the agent's character and circumstances
- Be something the agent would naturally want to share

AGENT GUIDELINES:
- Maintain diversity and realistic characteristics
- Consider existing social connections and how they naturally formed
- Keep geographic references generic
- Provide detailed, nuanced descriptions
- Ensure the agent remains believable and relatable

Generate the updated agent and specify the injected information using this schema: {InformationSeedResponse.model_json_schema()}.

IMPORTANT: Format the injected information as a complete, clear sentence containing all relevant details. This exact phrasing will be used in future conversation experiments to track information spread."""

    llm_response = await client.beta.chat.completions.parse(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format=InformationSeedResponse,
    )
    response = llm_response.choices[0].message.parsed
    assert response is not None, "Response parsing failed."

    new_seed_node = response.agent
    information_injected = response.information_injected

    mapping_copy = dict(agent_mapping)
    mapping_copy[seed_node] = new_seed_node
    return mapping_copy, information_injected


class Dataset(pydantic.BaseModel):
    agents: list[SocietyAgent]
    edges: list[tuple[int, int]]
    information_seed_agent: int
    injected_information: str


async def generate_dataset_from_graph[T](
    graph: nx.Graph,
    centrality_mapping: Mapping[T, float],
    client: AsyncOpenAI,
    model: str,
):
    centrality_mapping = nx.betweenness_centrality(graph)
    agent_mapping = await generate_agents(
        graph=graph,
        centrality_mapping=centrality_mapping,
        client=client,
        model=model,
    )
    seed_node = max(agent_mapping, key=lambda n: centrality_mapping[n])
    agent_mapping, injected_information = await create_information_seed(
        graph=graph,
        agent_mapping=agent_mapping,
        seed_node=seed_node,
        client=client,
        model=model,
    )

    node_to_id = {node: id for (id, (node, agent)) in enumerate(agent_mapping.items())}
    agents = list(agent_mapping.values())
    edges = [(node_to_id[u], node_to_id[v]) for (u, v) in graph.edges]
    return Dataset(
        agents=agents,
        edges=edges,
        information_seed_agent=node_to_id[seed_node],
        injected_information=injected_information,
    )

from typing import Literal, Mapping

import networkx as nx
import pydantic
from openai import AsyncOpenAI
import time
from generative_agents import AgentModelBase
from generative_agents.llm_backend import rate_limit_repeated


class GenerationStats(pydantic.BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    time_seconds: float = 0.0

    def __add__(self, other: "GenerationStats") -> "GenerationStats":
        return GenerationStats(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            time_seconds=self.time_seconds + other.time_seconds,
        )


class SocietyAgent(AgentModelBase):
    first_name: str = pydantic.Field(description="First name")
    last_name: str = pydantic.Field(description="Last name")
    sex: Literal["F", "M"] = pydantic.Field(description="Sex")
    age: int = pydantic.Field(description="Age of the agent in years")
    personality_type: str = pydantic.Field(
        description="Personality type of the agent, e.g. 'Introvert', 'Extrovert', 'Ambivert'"
    )
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

    # Communication & Interests
    communication_style: str = pydantic.Field(
        description="How the agent communicates, e.g. 'Direct', 'Diplomatic', 'Reserved', 'Humorous', 'Blunt'"
    )
    topics_of_interest: str = pydantic.Field(
        description="Topics the agent enjoys discussing, e.g. 'Politics, Technology, Sports, Music'"
    )
    favorite_music_genres: str = pydantic.Field(
        description="Favorite music genres, e.g. 'Rock, Jazz, Classical, Hip-hop'"
    )
    favorite_movies: str = pydantic.Field(
        description="Favorite movie genres or specific movies, e.g. 'Action, Comedy, Drama, Sci-Fi'"
    )
    favorite_books_genres: str = pydantic.Field(
        description="Favorite book genres, e.g. 'Mystery, Sci-Fi, Biography, Romance'"
    )
    favorite_sports: str = pydantic.Field(
        description="Favorite sports (as player or spectator), e.g. 'Basketball, Soccer, Tennis'"
    )

    # Economic
    income_bracket: str = pydantic.Field(
        description="Income bracket, e.g. 'Lower class', 'Middle class', 'Upper middle class', 'Wealthy'"
    )
    home_owner: bool = pydantic.Field(
        description="Whether the agent owns their residence"
    )
    vehicle_price_range: str = pydantic.Field(
        description="Vehicle price range, e.g. 'No car', 'Budget (under $10k)', 'Economy ($10-25k)', 'Mid-range ($25-50k)', 'Luxury ($50k+)'"
    )

    # Values & Psychology
    core_values: str = pydantic.Field(
        description="Core values of the agent, e.g. 'Family, Integrity, Success, Freedom'"
    )
    environmental_views: str = pydantic.Field(
        description="Environmental views, e.g. 'Climate activist', 'Skeptical', 'Neutral', 'Pragmatic'"
    )
    life_goals: str = pydantic.Field(
        description="Life goals of the agent, e.g. 'Career success, Family, Financial independence, Travel the world'"
    )
    biggest_fears: str = pydantic.Field(
        description="Biggest fears of the agent, e.g. 'Failure, Loneliness, Health issues, Financial ruin'"
    )
    major_life_events: str = pydantic.Field(
        description="Major past life events that shaped the agent, e.g. 'Divorce, Career change, Moved abroad, Lost a loved one'"
    )

    # Lifestyle
    sleep_schedule: str = pydantic.Field(
        description="Sleep schedule, e.g. 'Early riser', 'Night owl', 'Irregular', 'Standard 9-5'"
    )
    daily_routine: str = pydantic.Field(
        description="Daily routine pattern, e.g. 'Work-focused', 'Family-oriented', 'Leisure-heavy'"
    )
    exercise_frequency: str = pydantic.Field(
        description="Exercise frequency, e.g. 'Daily', 'Weekly', 'Rarely', 'Never'"
    )
    dietary_preferences: str = pydantic.Field(
        description="Dietary preferences, e.g. 'Vegetarian', 'Vegan', 'No restrictions', 'Keto', 'Organic only'"
    )

    # Reality Show Simulation
    dramatic_tendency: int = pydantic.Field(
        description="How conflict-seeking the agent is on a scale of 1-10"
    )
    popularity_seeking: int = pydantic.Field(
        description="Desire for attention and fame on a scale of 1-10"
    )
    emotional_expressiveness: int = pydantic.Field(
        description="How openly the agent shows emotions on a scale of 1-10"
    )
    competitiveness: int = pydantic.Field(
        description="Competitiveness level on a scale of 1-10"
    )
    strategic_thinking: int = pydantic.Field(
        description="Strategic thinking ability on a scale of 1-10"
    )
    entertainment_value: int = pydantic.Field(
        description="How entertaining the agent is on a scale of 1-10"
    )
    controversial_opinions: str = pydantic.Field(
        description="Controversial opinions the agent holds that could create drama"
    )

    # Information Spread Simulation
    trust_level: int = pydantic.Field(
        description="How trusting the agent is of others on a scale of 1-10"
    )
    rumor_spreading_tendency: int = pydantic.Field(
        description="Likelihood to share unverified information on a scale of 1-10"
    )
    influence_level: int = pydantic.Field(
        description="How much others listen to and follow the agent's views on a scale of 1-10"
    )
    social_media_activity: str = pydantic.Field(
        description="Social media activity level, e.g. 'High', 'Medium', 'Low', 'None'"
    )

    # Economic Games Simulation
    risk_tolerance: int = pydantic.Field(
        description="Willingness to take risks on a scale of 1-10"
    )
    cooperation_tendency: int = pydantic.Field(
        description="Likelihood to cooperate vs defect in games on a scale of 1-10"
    )
    generosity_level: int = pydantic.Field(
        description="Willingness to share resources with others on a scale of 1-10"
    )

    # Additional Realism
    languages_spoken: str = pydantic.Field(
        description="Languages the agent speaks, e.g. 'English', 'English, Spanish', 'English, Mandarin, French'"
    )
    pet_peeves: str = pydantic.Field(
        description="Things that annoy or irritate the agent"
    )
    health_status: str = pydantic.Field(
        description="Health status, e.g. 'Excellent', 'Good', 'Fair', 'Has chronic condition'"
    )
    travel_experiences: str = pydantic.Field(
        description="Travel experiences and countries visited, e.g. 'Never traveled abroad', 'Has visited 10+ countries'"
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
) -> tuple[dict[T, SocietyAgent], GenerationStats]:
    assert nx.is_connected(graph), "Graph must be connected for agent generation."

    result_mapping: dict[T, SocietyAgent] = {}

    @rate_limit_repeated()
    async def generate_single(node):
        start = time.perf_counter()
        stats = GenerationStats()
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
                "occupation": agent.occupation,
                "personality_type": agent.personality_type,
                "communication_style": agent.communication_style,
                "income_bracket": agent.income_bracket,
                "hobbies": agent.hobbies,
                "core_values": agent.core_values,
                "family_status": agent.family_status,
                "location": agent.location,
                "topics_of_interest": agent.topics_of_interest,
                "political_views": agent.political_views,
                "trust_level": agent.trust_level,
                "favorite_music_genres": agent.favorite_music_genres,
                "risk_tolerance": agent.risk_tolerance,
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
- IMPORTANT: Do not repeat previous answers or patterns - each agent should be unique
- Strive for originality in names, backgrounds, occupations, and personalities
- Vary the types of information you generate - avoid clustering similar agents together, unless neccesary due to shared background

PERSONA DETAILS TO CONSIDER:
- Personal background and life experiences, including major life events
- Professional occupation and career history
- Educational background and skills
- Social network and relationship patterns (family status, how connections formed)
- Interests, hobbies, lifestyle choices (sleep schedule, exercise, diet, hobbies)
- Communication style and topics of interest
- Economic status (income bracket, home ownership, vehicle)
- Values, beliefs, and worldview (core values, life goals, fears, political views, religion, environmental views)
- Geographic and cultural context (keep locations generic)
- Simulation-specific traits: trust level, influence level, social media activity, risk tolerance, competitiveness

RESPONSE FORMAT:
Provide detailed, nuanced descriptions where applicable. The richer the persona, the more authentic future conversations and information sharing will be.

Generate the agent using this schema: {SocietyAgent.model_json_schema()}"""

        result = await client.beta.chat.completions.parse(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format=SocietyAgent,
            temperature=0.3,
            top_p=0.7,
            frequency_penalty=0.2,
        )
        response = result.choices[0].message.parsed
        assert response is not None, "Response parsing failed."
        if result.usage:
            stats.input_tokens += result.usage.prompt_tokens
            stats.output_tokens += result.usage.completion_tokens
            stats.total_tokens += result.usage.total_tokens
        stats.time_seconds += time.perf_counter() - start
        return response, stats

    init_node, v = max(centrality_mapping.items(), key=lambda item: item[1])

    init_response, stats = await generate_single(init_node)
    result_mapping[init_node] = init_response

    while len(result_mapping) < len(graph.nodes):
        possible_choices = {
            node
            for generated_node in result_mapping
            for node in graph.neighbors(generated_node)
            if node not in result_mapping
        }
        # Extend the node with max centrality
        extension_node = max(possible_choices, key=lambda n: centrality_mapping[n])

        response, single_stats = await generate_single(extension_node)
        print(f"{len(result_mapping)} / {len(graph.nodes)} agents generated. Using {single_stats.input_tokens} / {single_stats.output_tokens} in {single_stats.time_seconds:.2f} s.")
        result_mapping[extension_node] = response
        stats += single_stats
    return result_mapping, stats


@rate_limit_repeated()
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
            "personality_type": agent.personality_type,
            "communication_style": agent.communication_style,
            "income_bracket": agent.income_bracket,
            "hobbies": agent.hobbies,
            "core_values": agent.core_values,
            "family_status": agent.family_status,
            "location": agent.location,
            "topics_of_interest": agent.topics_of_interest,
            "political_views": agent.political_views,
            "trust_level": agent.trust_level,
            "favorite_music_genres": agent.favorite_music_genres,
            "risk_tolerance": agent.risk_tolerance,
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

SEED INFORMATION PRINCIPLE:
The injected information should be a "north star" from the agent's background/CV - something meaningful and worth sharing that stems from their life experience, career, or personal journey. Choose one of the following types:

INFORMATION TYPES (choose one that best fits the agent):
1. Event Organization: Planning a party, community event, or gathering
2. Major Life Development: Starting a business, political campaign, or significant project  
3. Personal News: Sharing a secret, family update, or important personal development
4. Professional Announcement: Job change, promotion, or career milestone
5. Community Information: Local news, opportunities, or important community updates

The chosen information should:
- Be something the agent genuinely cares about and is proud of
- Have value to others and naturally spark conversation
- Be a meaningful piece of their story they'd want to share

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

    stats = GenerationStats()
    start = time.perf_counter()
    llm_response = await client.beta.chat.completions.parse(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format=InformationSeedResponse,
    )
    response = llm_response.choices[0].message.parsed
    assert response is not None, "Response parsing failed."

    new_seed_node = response.agent
    information_injected = response.information_injected

    if llm_response.usage:
        stats.input_tokens += llm_response.usage.prompt_tokens
        stats.output_tokens += llm_response.usage.completion_tokens
        stats.total_tokens += llm_response.usage.total_tokens
    stats.time_seconds += time.perf_counter() - start

    mapping_copy = dict(agent_mapping)
    mapping_copy[seed_node] = new_seed_node
    return (mapping_copy, information_injected, stats)


class Dataset(pydantic.BaseModel):
    agents: list[SocietyAgent]
    edges: list[tuple[int, int]]
    information_seed_agent: int
    injected_information: str
    dataset_generation_stats: GenerationStats


async def generate_dataset_from_graph[T](
    graph: nx.Graph,
    centrality_mapping: Mapping[T, float],
    client: AsyncOpenAI,
    model: str,
):
    agent_mapping, agent_stats = await generate_agents(
        graph=graph,
        centrality_mapping=centrality_mapping,
        client=client,
        model=model,
    )
    seed_node = max(agent_mapping, key=lambda n: centrality_mapping[n])
    agent_mapping, injected_information, seed_stats = await create_information_seed(
        graph=graph,
        agent_mapping=agent_mapping,
        seed_node=seed_node,
        client=client,
        model=model,
    )

    total_stats = agent_stats + seed_stats

    node_to_id = {node: id for (id, (node, agent)) in enumerate(agent_mapping.items())}
    agents = list(agent_mapping.values())
    edges = [(node_to_id[u], node_to_id[v]) for (u, v) in graph.edges]
    return Dataset(
        agents=agents,
        edges=edges,
        information_seed_agent=node_to_id[seed_node],
        injected_information=injected_information,
        dataset_generation_stats=total_stats,
    )

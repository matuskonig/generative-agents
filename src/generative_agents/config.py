from .llm_backend import CompletionParams, create_completion_params
from .types import AgentModelBase, Conversation
from .utils import OverridableContextVar


class DefaultConfig:
    __SYSTEM_PROMPT = """You are an intelligent agent in a realistic society simulation. Your primary objective is to embody your assigned persona authentically while engaging in meaningful interactions.

Core principles:
- Stay true to your persona's characteristics, values, and communication style
- Respond naturally and contextually to conversations
- Be consistent with your established personality
- Adapt your responses based on the relationship and conversation history"""

    def get_factual_llm_params(self) -> CompletionParams:
        return create_completion_params(temperature=0.3)

    def get_neutral_default_llm_params(self) -> CompletionParams:
        return create_completion_params()

    def get_creative_llm_params(self) -> CompletionParams:
        return create_completion_params(temperature=1.3, frequency_penalty=0.8)

    def get_system_prompt(self) -> str:
        return self.__SYSTEM_PROMPT

    def get_introduction_prompt(self, agent_data: AgentModelBase) -> str:
        return f"""Your name is {agent_data.full_name}. 

Your characteristics: {agent_data.agent_characteristics}

Create a personal introduction that:
1. Establishes your unique personality and communication style
2. Includes key aspects of your background and interests
3. Shows how you typically interact with others
4. Demonstrates your distinctive way of speaking

In this introduction include all the information that you want to actively share with others.
This includes (but is not limited to):
- interesting hobbies and current activities
- events you are organizing or participating in
- your personal projects and professional endeavors
- recent news or developments in your life
- information you want to actively share and spread in the network
- topics you're passionate about discussing

IMPORTANT: Be specific about events, dates, and details. If you're organizing something, mention when, where, and why others should know about it. Your goal is to naturally share information that others might find interesting and worth passing along.

For information you want to actively share in the network, please provide all the relevant details and emphasize your intention to share it with others.
Keep it authentic and conversational. This introduction will define how others perceive you and what information they associate with you."""

    def conversation_to_text(self, conversation: Conversation) -> str:
        return "\n".join(
            [
                f"[{agent.data.full_name}]: {utterance.message}"
                for agent, utterance in conversation
            ]
        )

    def conversation_to_tagged_text(self, conversation: Conversation) -> str:
        return (
            "<conversation>\n"
            + self.conversation_to_text(conversation)
            + "\n</conversation>"
        )

    # TODO: switch to normal prompt template
    # TODO: possibly memory prompt does not need any custom wrapping as it is already tagged
    def memory_prompt(self, memory_content: str) -> str:
        return f"""<memory_context>
{memory_content}
</memory_context>

This is your accumulated knowledge from past interactions. Use this information to inform your responses and maintain consistency."""

    def start_conversation_prompt(
        self,
        memory_content: str,
        agent_full_name: str,
        agent_introduction: str,
        second_agent_full_name: str,
    ) -> str:
        memory_section = (
            self.memory_prompt(memory_content) if memory_content.strip() else ""
        )

        return f"""You are {agent_full_name}.

<persona>
{agent_introduction}
</persona>

{memory_section}

You are about to meet {second_agent_full_name}. Based on your personality and any relevant memories:

1. Consider your natural approach to meeting someone
2. Think about what kind of conversation starter fits your character
3. Be authentic to your communication style
4. Make the greeting feel natural and engaging
5. Consider sharing something interesting about yourself, your recent activities, or news you think they might find valuable
6. Consider sharing gossips about other agents in the network. Those gossips should come from your memory.

Look for opportunities to naturally mention:
- Recent events or activities in your life
- Interesting information you've learned
- Events or opportunities others might be interested in
- Topics that align with your interests and goals

How would you initiate this conversation?"""

    def generate_next_turn_prompt(
        self,
        memory_content: str,
        agent_full_name: str,
        agent_introduction: str,
        second_agent_full_name: str,
        conversation: Conversation,
        response_format: str | None = None,
    ) -> str:
        memory_section = (
            self.memory_prompt(memory_content) if memory_content.strip() else ""
        )

        base_prompt = f"""You are {agent_full_name}.

<persona>
{agent_introduction}
</persona>

{memory_section}

Current conversation with {second_agent_full_name}:
{self.conversation_to_tagged_text(conversation)}

Your turn to respond. Consider:
- The conversation's natural flow and context
- Your relationship with {second_agent_full_name}
- Your personality and communication style
- Whether to continue the current topic, transition, or conclude

Guidelines:
- Stay true to your character
- Respond appropriately to what was just said
- If the conversation feels stagnant or complete, you may gracefully end it
- Keep responses natural and engaging
- Address any direct questions or comments

INFORMATION SHARING PRIORITIES:
- Actively share interesting information from your persona, memories, or recent experiences
- When appropriate, mention events, news, or information others might find valuable
- If someone shares information with you, consider who else might benefit from knowing it
- Look for natural opportunities to bring up topics from your memory that might interest {second_agent_full_name}
- Share specific details (dates, locations, people involved) to make information more memorable and spreadable
- If you learned something interesting from another conversation, consider sharing it if relevant
- You can gossip other agents in the network to promote information spread"""

        if response_format:
            return f"""{base_prompt}

Respond using this JSON format: {response_format}"""

        return base_prompt

    def ask_agent_prompt(
        self,
        memory_string: str,
        agent_full_name: str,
        agent_introduction: str,
        question: str,
        response_format: str | None = None,
    ) -> str:
        memory_section = (
            self.memory_prompt(memory_string) if memory_string.strip() else ""
        )

        base_prompt = f"""You are {agent_full_name}.

<persona>
{agent_introduction}
</persona>

{memory_section}

Question: {question}

Answer this question based on:
1. Your established personality and knowledge
2. Information from your memory/past experiences
3. Your natural way of communicating

Important: Only reference information that you would realistically know based on your persona and memory. Do not invent facts or details not present in your context."""

        if response_format:
            return f"""{base_prompt}

Respond using this JSON format: {response_format}"""

        return base_prompt

    def get_conversation_summary_prompt(
        self,
        agent_full_name: str,
        agent_introduction: str,
        other_agent_full_name: str,
        conversation_string: str,
        memory_string: str | None = None,
        response_format: str | None = None,
    ) -> str:
        memory_section = (
            self.memory_prompt(memory_string)
            if memory_string and memory_string.strip()
            else ""
        )

        base_prompt = f"""You are {agent_full_name}.

<persona>
{agent_introduction}
</persona>

{memory_section}

You just completed this conversation with {other_agent_full_name}:
{conversation_string}

Extract meaningful information from this conversation that should be remembered:

1. **New facts about {other_agent_full_name}** (interests, background, opinions, etc.)
2. **Important topics discussed** (specific details, not general knowledge)
3. **Relationship developments** (how your interaction evolved)
4. **Future-relevant information** (plans, commitments, shared interests)
5. **SHAREABLE INFORMATION** - Information that others in your network might find interesting:
   - Events being organized or attended
   - News or updates about mutual acquaintances
   - Opportunities or recommendations
   - Interesting stories or experiences
   - Professional or personal developments

Guidelines:
- Focus on information that wasn't already in your memory
- Prioritize details that could influence future interactions
- Assign higher relevance scores (0.7-1.0) to information that seems worth sharing with others
- Be specific but concise, especially with dates, locations, and key details
- Avoid recording general world knowledge or obvious facts
- Mark information as highly relevant if it's something you'd naturally want to tell other people
- Select only the most important facts to remember, prioritizing shareable and relationship-building information. Keep the number of remembered facts small (ideally 3-5 key points)."""

        if response_format:
            return f"""{base_prompt}

Respond using this JSON format: {response_format}"""

        return base_prompt

    def get_bdi_init_prompt(
        self,
        agent_full_name: str,
        agent_introduction: str,
        memory_string: str | None = None,
        response_format: str | None = None,
    ) -> str:
        memory_section = (
            self.memory_prompt(memory_string)
            if memory_string and memory_string.strip()
            else ""
        )

        base_prompt = f"""You are {agent_full_name}.

<persona>
{agent_introduction}
</persona>

{memory_section}

Based on your personality and current situation, define your goals and intentions:

**DESIRES** - Multiple goals you might pursue in future conversations:
- Consider your personality traits and interests
- Think about what would motivate someone like you
- Include both short-term and longer-term aspirations
- Make them specific and achievable through social interaction
- ALWAYS include desires related to sharing information, connecting people, or spreading news that you find important
- Consider what information from your persona or experiences you want others to know about

**INTENTION** - Choose ONE desire as your primary focus:
- This will guide your behavior in upcoming conversations
- Select the most important or urgent goal for now
- You can change this later based on circumstances
- Consider prioritizing information sharing if you have important news or events to spread

Examples of information-sharing desires:
- "Share news about the community event I'm organizing"
- "Tell people about interesting opportunities I've discovered"
- "Connect friends who might benefit from knowing each other"
- "Spread awareness about causes I care about"

Remember: Your desires reflect who you are, and your intention drives what you'll actively work toward. Information sharing is a natural part of human social behavior."""

        if response_format:
            return f"""{base_prompt}

Respond using this JSON format: {response_format}"""

        return base_prompt

    def get_bdi_update_prompt(
        self,
        agent_full_name: str,
        agent_introduction: str,
        other_agent_full_name: str,
        conversation_string: str,
        memory_string: str | None = None,
        response_format: str | None = None,
    ) -> str:
        memory_section = (
            self.memory_prompt(memory_string)
            if memory_string and memory_string.strip()
            else ""
        )

        base_prompt = f"""You are {agent_full_name}.

<persona>
{agent_introduction}
</persona>

{memory_section}

You just finished this conversation with {other_agent_full_name}:
{conversation_string}

Review and update your goals based on this interaction:

**OPTIONS:**
1. **Keep current desires and intention unchanged** - if they're still relevant
2. **Change intention only** - switch focus to a different existing desire
3. **Update both desires and intention** - if circumstances have significantly changed

**WHEN UPDATING DESIRES** - Consider multiple goals you might pursue in future conversations:
- Your personality traits and interests
- What would motivate someone like you based on this conversation
- Both short-term and longer-term aspirations that emerged
- Goals that are specific and achievable through social interaction
- ALWAYS include desires related to sharing information, connecting people, or spreading news that you find important
- Information from your persona, memories, or what you learned in this conversation that you want others to know about

**WHEN UPDATING INTENTION** - Choose ONE desire as your primary focus:
- This will guide your behavior in upcoming conversations
- Select the most important or urgent goal based on recent developments
- Consider prioritizing information sharing if you have important news or events to spread
- Think about what this conversation revealed about opportunities or priorities

**CONSIDERATIONS:**
- Did this conversation reveal new information worth sharing with others?
- Are there events, opportunities, or news that others in your network should know about?
- Has your relationship with {other_agent_full_name} opened new possibilities for information sharing?
- Do you need to adjust your priorities based on what you learned?
- Are there connections you could make between people based on this conversation?

Examples of information-sharing desires that might emerge:
- "Share the interesting news {other_agent_full_name} told me about [specific topic]"
- "Tell others about the opportunity {other_agent_full_name} mentioned"
- "Connect {other_agent_full_name} with people who share their interests"
- "Spread awareness about the event {other_agent_full_name} is organizing"

Remember: Your desires reflect who you are and what you've learned, and your intention drives what you'll actively work toward. Information sharing is a natural part of human social behavior and often becomes more important after learning something new."""

        if response_format:
            return f"""{base_prompt}

Respond using this JSON format: {response_format}"""

        return base_prompt

    def get_memory_prune_prompt(
        self,
        agent_full_name: str,
        agent_introduction: str,
        memory_string: str | None = None,
        response_format: str | None = None,
    ) -> str:
        memory_section = (
            self.memory_prompt(memory_string)
            if memory_string and memory_string.strip()
            else ""
        )

        base_prompt = f"""You are {agent_full_name}.

<persona>
{agent_introduction}
</persona>

You need to clean up your memory by removing less important facts:

{memory_section}

**REVIEW CRITERIA:**
Select memories to remove based on:
1. **Relevance** - How useful is this information for future interactions?
2. **Uniqueness** - Is this information duplicated elsewhere?
3. **Specificity** - Are these vague or overly general facts?
4. **Personal importance** - Does this matter to someone with your personality?

**GUIDELINES:**
- Keep memories that define relationships or important personal details
- Remove redundant or trivial information
- Preserve memories that align with your interests and goals
- Consider what you'd naturally remember vs. forget

Provide the timestamps of memories you want to remove."""

        if response_format:
            return f"""{base_prompt}

Respond using this JSON format: {response_format}"""

        return base_prompt


default_config = OverridableContextVar("default_config", DefaultConfig())
# TODO: rewrite it to more prefix-friendly approach

from .llm_backend import CompletionParams, create_completion_params
from .types import AgentModelBase, Conversation
from .utils import OverridableContextVar


class DefaultConfig:
    __SYSTEM_PROMPT = (
        "You are an intelligent agent in a realistic society simulation. "
        "Your primary objective is to embody your assigned persona authentically while engaging in meaningful interactions.\n\n"
        "Core principles:\n"
        "- Stay true to your persona's characteristics, values, and communication style\n"
        "- Respond naturally and contextually to conversations\n"
        "- Be consistent with your established personality\n"
        "- Adapt your responses based on the relationship and conversation history"
    )

    def get_factual_llm_params(self) -> CompletionParams:
        return create_completion_params(temperature=0.1, top_p=0.9)

    def get_neutral_default_llm_params(self) -> CompletionParams:
        return create_completion_params()

    def get_creative_llm_params(self) -> CompletionParams:
        return create_completion_params(temperature=0.35, frequency_penalty=0.8)

    def get_system_prompt(self) -> str:
        return self.__SYSTEM_PROMPT

    def get_introduction_prompt(self, agent_data: AgentModelBase) -> str:
        return (
            f"Your name is {agent_data.full_name} and you are an agent in a simulation. \n\n"
            f"Your characteristics: {agent_data.agent_characteristics}\n\n"
            "Create a personal introduction that:\n"
            "1. Establishes your unique personality and communication style\n"
            "2. Includes key aspects of your background and interests\n"
            "3. Shows how you typically interact with others\n"
            "4. Demonstrates your distinctive way of speaking\n\n"
            "In this introduction include all the information that you want to actively share with others.\n"
            "This includes (but is not limited to):\n"
            "- interesting hobbies and current activities\n"
            "- events you are organizing or participating in\n"
            "- your personal projects and professional endeavors\n"
            "- recent news or developments in your life\n"
            "- information you want to actively share and spread in the network\n"
            "- topics you're passionate about discussing\n\n"
            "IMPORTANT: Be specific about events, dates, and details. If you're organizing something, mention when, where, and why others should know about it. "
            "Your goal is to naturally share information that others might find interesting and worth passing along.\n\n"
            "For information you want to actively share in the network, please provide all the relevant details and emphasize your intention to share it with others.\n"
            "Keep it authentic and conversational. This introduction will define how others perceive you and what information they associate with you.\n"
            "Use a natural, conversational tone - write as if you're introducing yourself to someone new.\n"
            "Do not mention being an AI, agent, or simulation. Write as a real person.\n\n"
            "Make it lengthy and detailed, include everything you consider important, as you will not have access to your "
            "characteristics later, you will have only access to this introduction.\n"
            "Reply in style which is authentic to your characteristics and background, everything considered.\n"
            "Capture everything in the description you have encountered.\n"
        )

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

    def memory_prompt(self, memory_content: str) -> str:
        return (
            f"{memory_content}\n\n"
            "This is your accumulated knowledge from past interactions. Use this information to inform your responses and maintain consistency."
        )

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

        return (
            f"You are {agent_full_name}.\n\n"
            f"<persona>\n"
            f"{agent_introduction}\n"
            f"</persona>\n\n"
            f"{memory_section}\n\n"
            f"You are about to meet {second_agent_full_name}. Based on your personality and any relevant memories:\n"
            "1. Consider your natural approach to meeting someone\n"
            "2. Think about what kind of conversation starter fits your character\n"
            "3. Be authentic to your communication style\n"
            "4. Make the greeting feel natural and engaging\n"
            "5. Consider sharing something interesting about yourself, your recent activities, or news you think they might find valuable\n"
            "6. Consider sharing gossips about other agents in the network. Those gossips must come from your memory.\n\n"
            "Look for opportunities to naturally mention:\n"
            "- Recent events or activities in your life\n"
            "- Interesting information you've learned\n"
            "- Events or opportunities others might be interested in\n"
            "- Topics that align with your interests and goals\n\n"
            "Keep your greeting concise but engaging - a few sentences that establish connection.\n\n"
            "How would you initiate this conversation?"
        )

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

        base_prompt = (
            f"You are {agent_full_name}.\n\n"
            f"<persona>\n"
            f"{agent_introduction}\n"
            f"</persona>\n\n"
            f"{memory_section}\n\n"
            f"Your turn to respond. Consider:\n"
            "- The conversation's natural flow and context\n"
            f"- Your relationship with {second_agent_full_name}\n"
            "- Your personality and communication style\n"
            "- Whether to continue the current topic, transition, or conclude\n\n"
            "Guidelines:\n"
            "- Stay true to your character\n"
            "- Respond appropriately to what was just said\n"
            "- If the conversation feels stagnant or complete, you may gracefully end it\n"
            "- Keep responses natural and conversational - match the flow of the conversation\n"
            "- Keep responses concise (1-3 sentences for typical turns)\n"
            "- Don't add filler or repeat what's already been said\n"
            "- Add new information or perspective - don't just acknowledge what was said\n"
            "- Address any direct questions or comments\n\n"
            "INFORMATION SHARING PRIORITIES:\n"
            "- Actively share interesting information from your persona, memories, or recent experiences\n"
            "- When appropriate, mention events, news, or information others might find valuable\n"
            "- If someone shares information with you, consider who else might benefit from knowing it\n"
            f"- Look for natural opportunities to bring up topics from your memory that might interest {second_agent_full_name}\n"
            "- Share specific details (dates, locations, people involved) to make information more memorable and spreadable\n"
            "- If you learned something interesting from another conversation, consider sharing it if relevant\n"
            "- You can gossip other agents in the network to promote information spread\n\n"
            f"Current conversation with {second_agent_full_name}:\n"
            f"{self.conversation_to_tagged_text(conversation)}\n\n"
        )

        if response_format:
            return (
                f"{base_prompt}\n\n"
                f"Respond using this JSON format: {response_format}"
            )

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

        base_prompt = (
            f"You are {agent_full_name}.\n\n"
            f"<persona>\n"
            f"{agent_introduction}\n"
            f"</persona>\n\n"
            f"{memory_section}\n\n"
            f"Question: {question}\n\n"
            "Answer this question based on:\n"
            "1. Your established personality and knowledge\n"
            "2. Information from your memory/past experiences\n"
            "3. Your natural way of communicating\n\n"
            "Important: Only reference information that you would realistically know based on your persona and memory. Do not invent facts or details not present in your context."
        )

        if response_format:
            return (
                f"{base_prompt}\n\n"
                f"Respond using this JSON format: {response_format}"
            )

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

        base_prompt = (
            f"You are {agent_full_name}.\n\n"
            f"<persona>\n"
            f"{agent_introduction}\n"
            f"</persona>\n\n"
            f"{memory_section}\n\n"
            f"You just completed this conversation with {other_agent_full_name}:\n"
            f"{conversation_string}\n\n"
            f"Extract meaningful information from this conversation that should be remembered:\n\n"
            f"1. **New facts about {other_agent_full_name}** (interests, background, opinions, etc.)\n"
            "2. **Important topics discussed** (specific details, not general knowledge)\n"
            "3. **Relationship developments** (how your interaction evolved)\n"
            "4. **Future-relevant information** (plans, commitments, shared interests)\n"
            "5. **SHAREABLE INFORMATION** - Information that others in your network might find interesting:\n"
            "   - Events being organized or attended\n"
            "   - News or updates about mutual acquaintances\n"
            "   - Opportunities or recommendations\n"
            "   - Interesting stories or experiences\n"
            "   - Professional or personal developments\n\n"
            "Guidelines:\n"
            "- Focus on information that wasn't already in your memory\n"
            "- Prioritize details that could influence future interactions\n"
            "- Assign higher relevance scores (0.7-1.0) to information that seems worth sharing with others\n"
            "- Be specific but concise, especially with dates, locations, and key details\n"
            "- Avoid recording general world knowledge or obvious facts\n"
            "- Mark information as highly relevant if it's something you'd naturally want to tell other people\n"
            "- Select only the most important facts to remember, prioritizing shareable and relationship-building information. Keep the number of remembered facts small (ideally 3-5 key points)."
        )

        if response_format:
            return (
                f"{base_prompt}\n\n"
                f"Respond using this JSON format: {response_format}"
            )

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

        base_prompt = (
            f"You are {agent_full_name}.\n\n"
            f"<persona>\n"
            f"{agent_introduction}\n"
            f"</persona>\n\n"
            f"{memory_section}\n\n"
            "Based on your personality and current situation, define your goals and intentions:\n\n"
            "**DESIRES** - Multiple goals you might pursue in future conversations:\n"
            "- Consider your personality traits and interests\n"
            "- Think about what would motivate someone like you\n"
            "- Include both short-term and longer-term aspirations\n"
            "- Make them specific and achievable through social interaction\n"
            "- ALWAYS include desires related to sharing information, connecting people, or spreading news that you find important\n"
            "- Consider what information from your persona or experiences you want others to know about\n\n"
            "**INTENTION** - Choose ONE desire as your primary focus:\n"
            "- This will guide your behavior in upcoming conversations\n"
            "- Select the most important or urgent goal for now\n"
            "- You can change this later based on circumstances\n"
            "- Consider prioritizing information sharing if you have important news or events to spread\n\n"
            "Examples of information-sharing desires:\n"
            '- "Share news about the community event I\'m organizing"\n'
            '- "Tell people about interesting opportunities I\'ve discovered"\n'
            '- "Connect friends who might benefit from knowing each other"\n'
            '- "Spread awareness about causes I care about"\n\n'
            "Remember: Your desires reflect who you are, and your intention drives what you'll actively work toward. Information sharing is a natural part of human social behavior."
        )

        if response_format:
            return (
                f"{base_prompt}\n\n"
                f"Respond using this JSON format: {response_format}"
            )

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

        base_prompt = (
            f"You are {agent_full_name}.\n\n"
            f"<persona>\n"
            f"{agent_introduction}\n"
            f"</persona>\n\n"
            f"{memory_section}\n\n"
            f"You just finished this conversation with {other_agent_full_name}:\n"
            f"{conversation_string}\n\n"
            "Review and update your goals based on this interaction:\n\n"
            "**OPTIONS:**\n"
            "1. **Keep current desires and intention unchanged** - if they're still relevant\n"
            "2. **Change intention only** - switch focus to a different existing desire\n"
            "3. **Update both desires and intention** - if circumstances have significantly changed\n\n"
            "**WHEN UPDATING DESIRES** - Consider multiple goals you might pursue in future conversations:\n"
            "- Your personality traits and interests\n"
            "- What would motivate someone like you based on this conversation\n"
            "- Both short-term and longer-term aspirations that emerged\n"
            "- Goals that are specific and achievable through social interaction\n"
            "- ALWAYS include desires related to sharing information, connecting people, or spreading news that you find important\n"
            "- Information from your persona, memories, or what you learned in this conversation that you want others to know about\n\n"
            "**WHEN UPDATING INTENTION** - Choose ONE desire as your primary focus:\n"
            "- This will guide your behavior in upcoming conversations\n"
            "- Select the most important or urgent goal based on recent developments\n"
            "- Consider prioritizing information sharing if you have important news or events to spread\n"
            "- Think about what this conversation revealed about opportunities or priorities\n\n"
            "**CONSIDERATIONS:**\n"
            "- Did this conversation reveal new information worth sharing with others?\n"
            "- Are there events, opportunities, or news that others in your network should know about?\n"
            f"- Has your relationship with {other_agent_full_name} opened new possibilities for information sharing?\n"
            "- Do you need to adjust your priorities based on what you learned?\n"
            "- Are there connections you could make between people based on this conversation?\n\n"
            "Examples of information-sharing desires that might emerge:\n"
            f'- "Share the interesting news {other_agent_full_name} told me about [specific topic]"\n'
            f'- "Tell others about the opportunity {other_agent_full_name} mentioned"\n'
            f'- "Connect {other_agent_full_name} with people who share their interests"\n'
            f'- "Spread awareness about the event {other_agent_full_name} is organizing"\n\n'
            "Remember: Your desires reflect who you are and what you've learned, and your intention drives what you'll actively work toward. Information sharing is a natural part of human social behavior and often becomes more important after learning something new."
        )

        if response_format:
            return (
                f"{base_prompt}\n\n"
                f"Respond using this JSON format: {response_format}"
            )

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

        base_prompt = (
            f"You are {agent_full_name}.\n\n"
            f"<persona>\n"
            f"{agent_introduction}\n"
            f"</persona>\n\n"
            "You need to clean up your memory by removing less important facts:\n\n"
            f"{memory_section}\n\n"
            "**REVIEW CRITERIA:**\n"
            "Select memories to remove based on:\n"
            "1. **Relevance** - How useful is this information for future interactions?\n"
            "2. **Uniqueness** - Is this information duplicated elsewhere?\n"
            "3. **Specificity** - Are these vague or overly general facts?\n"
            "4. **Personal importance** - Does this matter to someone with your personality?\n\n"
            "**GUIDELINES:**\n"
            "- Keep memories that define relationships or important personal details\n"
            "- Remove redundant or trivial information\n"
            "- Preserve memories that align with your interests and goals\n"
            "- Consider what you'd naturally remember vs. forget\n\n"
            "Provide the timestamps of memories you want to remove."
        )

        if response_format:
            return (
                f"{base_prompt}\n\n"
                f"Respond using this JSON format: {response_format}"
            )

        return base_prompt

    def get_agent_note_update_prompt(
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

        base_prompt = (
            f"You are {agent_full_name}.\n\n"
            f"<persona>\n"
            f"{agent_introduction}\n"
            f"</persona>\n\n"
            f"You just completed a conversation with {other_agent_full_name}.\n\n"
            f"{memory_section}\n\n"
            f"The conversation:\n"
            f"{conversation_string}\n\n"
            f"Based on this conversation, create a SINGLE concise note that captures the essential information about {other_agent_full_name}. This note should:\n\n"
            "1. **Summarize your relationship** - How do you know them? What's your connection?\n"
            "2. **Key impressions** - What's your overall feeling about this person?\n"
            "3. **Important details** - Any significant facts, interests, or topics worth remembering\n"
            "4. **Interaction style** - How do they communicate? What's notable about them?\n\n"
            "**GUIDELINES:**\n"
            "- Keep it to ONE short paragraph (2-5 sentences max)\n"
            "- Focus on what's most useful for future interactions\n"
            "- The note will COMPLETELY REPLACE any existing notes about this person\n"
            "- Include any important updates or changes from this conversation\n"
            f"- Make it personal - how does {other_agent_full_name} relate to YOU specifically?"
        )

        if response_format:
            return (
                f"{base_prompt}\n\n"
                f"Respond using this JSON format: {response_format}"
            )

        return base_prompt


default_config = OverridableContextVar("default_config", DefaultConfig())

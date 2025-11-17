from langchain.agents.middleware import AgentMiddleware

class MessageTrimmerMiddleware(AgentMiddleware):
    """
    控制读取的记忆，避免全量的记忆读取导致超出上下文限制
    """

    def __init__(self, max_messages: int = 20):
        self.max_messages = max_messages

    async def __call__(self, state, context, next_step):
        # 修剪输入消息
        if "messages" in state:
            messages = state["messages"]
            system_msgs = [m for m in messages if hasattr(m, 'type') and m.type == "system"]
            other_msgs = [m for m in messages if not (hasattr(m, 'type') and m.type == "system")]

            # 只保留最近的消息
            if len(other_msgs) > self.max_messages:
                trimmed_msgs = system_msgs + other_msgs[-self.max_messages:]
                state = {**state, "messages": trimmed_msgs}

        # 继续执行
        return await next_step(state, context)
"""The same outbound DeepSeek empty-reasoning fix used by the V6.2 run."""
from minisweagent.models.litellm_model import LitellmModel


class DeepSeekNativeModel(LitellmModel):
    def _prepare_messages_for_api(self, messages):
        prepared = super()._prepare_messages_for_api(messages)
        for message in prepared:
            if message.get("role") == "assistant" and message.get("reasoning_content") == "":
                message["reasoning_content"] = " "
        return prepared

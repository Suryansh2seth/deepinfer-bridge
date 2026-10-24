"""Request preparation: protocol JSON -> DeepSeek V4.1 prompt token IDs, via deepseek-recipe.

Adapted from deepseek-recipe's server-py example (MIT). The difference: DeepInfer consumes
token IDs, so the prompt is *encoded* with recipe's own tokenizer-aware encoder instead of
being returned as text. That keeps special-token handling identical to DeepSeek's API.
"""

from __future__ import annotations

from dataclasses import dataclass

from deepseek_recipe import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ConversationRequest,
    ConversionOptions,
    DeepseekV41Encoding,
    InferenceOptions,
    MessagesRequest,
    MessagesResponse,
    ModelVersion,
    ResponsesRequest,
    ResponsesResponse,
    SpecialToken,
    Tokenizer,
)

PROTOCOL_TYPES = {
    "chat_completions": (ChatCompletionRequest, ChatCompletionResponse),
    "responses": (ResponsesRequest, ResponsesResponse),
    "messages": (MessagesRequest, MessagesResponse),
}


class RequestError(Exception):
    """A failure with an HTTP status chosen by the bridge."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class PreparedRequest:
    protocol: str
    conversation_request: ConversationRequest
    token_ids: list[int]
    max_tokens: int  # already resolved against the context limit
    model: str

    @property
    def stream(self) -> bool:
        return self.conversation_request.stream

    @property
    def inference_options(self) -> InferenceOptions:
        return self.conversation_request.inference_options


def conversion_options(served_model_name: str) -> ConversionOptions:
    """Accept the served name plus the names recipe already knows for V4.1 Flash."""
    return (
        ConversionOptions()
        .with_model_alias(served_model_name, ModelVersion.DeepseekV41Flash)
        .with_model_alias("deepseek-flash", ModelVersion.DeepseekV41Flash)
    )


def make_encoding(tokenizer: Tokenizer) -> DeepseekV41Encoding:
    return DeepseekV41Encoding().with_tokenizer(tokenizer)


def prepare_request(
    protocol: str,
    body: bytes,
    *,
    options: ConversionOptions,
    encoding: DeepseekV41Encoding,
    context_limit: int,
) -> PreparedRequest:
    """Convert, render, encode and resolve max_tokens. Blocking (CPU); run in a threadpool.

    ConversionError (bad request, prompt too long, unknown model) propagates untouched so the
    HTTP layer can return recipe's protocol-shaped error body.
    """
    request_type, _ = PROTOCOL_TYPES[protocol]
    try:
        request = request_type(body)
    except ValueError as error:
        raise RequestError(str(error)) from error
    model = request.model
    converted = request.convert(options)
    if converted.model_version != ModelVersion.DeepseekV41Flash:
        raise RequestError("this bridge serves DeepSeek-V4.1-Flash only")
    if encoding.render_conversation(converted.conversation).image_sources:
        raise RequestError("image input is not supported by this bridge")
    token_ids = encoding.encode(converted.conversation)
    max_tokens = converted.resolve_max_tokens(len(token_ids), context_limit)
    return PreparedRequest(protocol, converted, token_ids, max_tokens, model)


def build_sampling(prepared: PreparedRequest, tokenizer: Tokenizer, *, constrained: bool) -> dict:
    """Map recipe's resolved InferenceOptions onto DeepInfer SamplingParams kwargs (plain dict,
    so the HTTP process never imports deepinfer/torch)."""
    opts = prepared.inference_options
    sp: dict = {
        "temperature": opts.temperature if opts.temperature is not None else 1.0,
        "top_p": opts.top_p if opts.top_p is not None else 1.0,
        "top_k": -1,
        "max_tokens": prepared.max_tokens,
    }
    if opts.seed is not None:
        sp["seq_seed"] = opts.seed
    constraint = opts.output_constraint
    # DeepInfer enforces json / regex / ebnf. recipe's tool_call* constraints have no DeepInfer
    # equivalent, so tool calls rely on the model + recipe's parser (same as unconstrained API use).
    if constrained and constraint is not None and constraint.kind in ("json", "regex"):
        so_args: dict = {
            "schema_type": constraint.kind,
            "schema": (constraint.schema or "{}") if constraint.kind == "json" else constraint.schema,
        }
        thinking = prepared.conversation_request.parsing_options.thinking_initial_stage is not None
        if thinking:
            # Let the model think freely; the grammar engages after the end-of-thinking token.
            so_args["trigger_token_ids"] = tokenizer.encode(SpecialToken.ThinkingEnd.default_str())
        sp["so_args"] = so_args
    return sp

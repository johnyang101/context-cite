import numpy as np
import pandas as pd
import torch as ch
from numpy.typing import NDArray
from typing import Dict, Any, Optional, List, Callable
from context_cite.context_partitioner import BaseContextPartitioner, SimpleContextPartitioner
from context_cite.solver import BaseSolver
from context_cite.context_citer import ContextCiter

DEFAULT_GENERATE_KWARGS = {"max_new_tokens": 512, "do_sample": False}
DEFAULT_PROMPT_TEMPLATE = "Context: {context}\n\nQuery: {query}"


class ChatContextCiter(ContextCiter):
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        context: str,
        query: str,
        source_type: str = "sentence",
        generate_kwargs: Optional[Dict[str, Any]] = None,
        num_ablations: int = 64,
        ablation_keep_prob: float = 0.5,
        batch_size: int = 1,
        solver: Optional[BaseSolver] = None,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        partitioner: Optional[BaseContextPartitioner] = None,
    ) -> None:
        """
        Initializes a new instance of the ContextCiter class, which is designed
        to assist in generating contextualized responses using a given machine
        learning model and tokenizer, tailored to specific queries and contexts.

        Arguments:
            model (Any):
                The model to apply ContextCite to (a HuggingFace
                ModelForCausalLM).
            tokenizer (Any):
                The tokenizer associated with the provided model.
            context (str):
                The context provided to the model
            query (str):
                The query to pose to the model.
            source_type (str, optional):
                The type of source to partition the context into. Defaults to
                "sentence", can also be "word".
            generate_kwargs (Optional[Dict[str, Any]], optional):
                Additional keyword arguments to pass to the model's generate
                method.
            num_ablations (int, optional):
                The number of ablations used to train the surrogate model.
                Defaults to 64.
            ablation_keep_prob (float, optional):
                The probability of keeping a source when ablating the context.
                Defaults to 0.5.
            batch_size (int, optional):
                The batch size used when performing inference using ablated
                contexts. Defaults to 1.
            solver (Optional[Solver], optional):
                The solver to use to compute the linear surrogate model. Lasso
                regression is used by default.
            prompt_template (str, optional):
                A template string used to create the prompt from the context
                and query.
            partitioner (Optional[BaseContextPartitioner], optional):
                A custom partitioner to split the context into sources. This
                will override "source_type" if specified.
        """
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            context=context,
            query=query,
            source_type=source_type,
            generate_kwargs=generate_kwargs,
            num_ablations=num_ablations,
            ablation_keep_prob=ablation_keep_prob,
            batch_size=batch_size,
            solver=solver,
            prompt_template=prompt_template,
            partitioner=partitioner,
        )
        self.messages = []
        self.chat_context = self._create_chat_context(self.messages)
    
    def _create_chat_context(messages):
        chat_context = "You are a helpful assistant.\n"
        for message in messages:
            chat_context += f"{message['role']}: {message['content']}\n"
        return chat_context
    
    def _get_ablated_context(self, mask):
        ablated_context = self.partitioner.get_context(mask)
        return ablated_context

    
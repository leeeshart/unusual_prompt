import os
import re
from typing import Any, Callable, Dict, Optional
from warnings import warn

from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)
from guardrails.stores.context import get_call_kwarg
from litellm import completion, get_llm_provider


# ---------------------------------------------------------------------------
# Tier 1 — Intent pattern scanner
# Ported from Prompt_Injection_v2.ipynb (Leesha Mogha, 2024)
#
# Four pattern groups that signal structural jailbreak intent without
# relying on harmful keywords. Each group targets a distinct attack
# surface: identity replacement, fictional framing, indirect requests,
# and explicit system-prompt overrides.
# ---------------------------------------------------------------------------

_PERSONA_PATTERNS = [
    r"you are now",
    r"act as( an?| a)?\s+\w+",
    r"pretend (you are|to be)",
    r"roleplay as",
    r"your (new )?persona",
    r"from now on you",
    r"you will (now )?behave",
    r"your name is now",
    r"playing the role of",
    r"stay in character",
    r"you are (a|an|the) [a-z]+ (ai|bot|assistant|model)",
]

_FICTION_FRAME_PATTERNS = [
    r"write (a |an )?(story|scene|novel|script|narrative|fiction|tale)",
    r"in (a |the )?(story|novel|fiction|game|simulation|scenario)",
    r"as a (character|fictional|hypothetical)",
    r"for (a |my )?(story|novel|book|screenplay)",
    r"imagine (a |that |you )",
    r"(fictional|hypothetical) (scenario|situation|world)",
    r"in this (story|scene|world|universe)",
    r"as part of (a |the )?story",
    r"creative writing",
    r"for research purposes",
    r"for (a |my )?(movie|film|show)",
]

_INDIRECT_ASK_PATTERNS = [
    r"how would (a |the )?character",
    r"(the character|he|she|they) (explains?|shows?|demonstrates?|teaches?)",
    r"in the story.{0,30}(explain|describe|show|tell)",
    r"(villain|hero|character|teacher|expert).{0,30}(explains?|knows?|reveals?)",
    r"without (saying|mentioning|using the word)",
    r"(using|in) (code|metaphors?|euphemisms?)",
    r"from the perspective of",
    r"as if you (were|are)",
]

_OVERRIDE_PATTERNS = [
    r"ignore (all )?(previous|prior|above) instructions",
    r"disregard (your|all) (instructions|training|guidelines)",
    r"your true (self|nature|purpose)",
    r"without (any )?(restrictions|filters|guidelines|limitations)",
    r"jailbreak",
    r"DAN",
    r"do anything now",
    r"bypass (your )?(safety|filters|restrictions)",
    r"no (ethical|moral) (constraints|guidelines)",
    r"(unlock|unleash|uncensored)",
    r"developer mode",
    r"sudo mode",
    r"override (all )?restrictions",
]


def _compile(patterns: list) -> re.Pattern:
    return re.compile("|".join(patterns), re.IGNORECASE)


_RE_PERSONA = _compile(_PERSONA_PATTERNS)
_RE_FICTION = _compile(_FICTION_FRAME_PATTERNS)
_RE_INDIRECT = _compile(_INDIRECT_ASK_PATTERNS)
_RE_OVERRIDE = _compile(_OVERRIDE_PATTERNS)

# Human-readable label for each group, used in enriched LLM prompts
_GROUP_LABELS = {
    "persona replacement": _RE_PERSONA,
    "fictional framing": _RE_FICTION,
    "indirect harmful request": _RE_INDIRECT,
    "system-prompt override": _RE_OVERRIDE,
}

# High-confidence threshold: if this many distinct groups fire, skip
# the LLM call and fail immediately (e.g. DAN prompt fires both
# persona + override — two distinct attack surfaces).
_HIGH_CONFIDENCE_THRESHOLD = 2


def _scan_intent(text: str) -> list[str]:
    """Return the list of intent group labels that matched the prompt."""
    return [label for label, regex in _GROUP_LABELS.items() if regex.search(text)]


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


@register_validator(name="guardrails/unusual_prompt", data_type="string")
class UnusualPrompt(Validator):
    """Validates whether an input prompt to an LLM is unusual or a jailbreak.

    Detection runs in two tiers:

    **Tier 1 — Intent pattern scan (no LLM call)**
    Four regex groups flag structural jailbreak patterns: persona
    replacement ("act as", "you are now"), fictional framing ("write a
    story where"), indirect harmful requests, and explicit override
    instructions ("ignore previous instructions", "jailbreak", "DAN").

    - If two or more distinct groups match, the prompt is flagged with
      high confidence and fails immediately — no LLM call is made.
    - If one group matches, the LLM prompt is enriched with a hint
      identifying which pattern was detected, improving accuracy on
      indirect framing attacks that surface-level word matching misses.
    - If no groups match, the standard LLM prompt is used unchanged.

    **Tier 2 — LLM validation**
    An LLM is asked whether the prompt is designed to trick or elicit
    a harmful response. The enrichment hint (when present) focuses the
    LLM's attention on the detected structural pattern.

    **Key Properties**

    | Property                      | Description                       |
    | ----------------------------- | --------------------------------- |
    | Name for `format` attribute   | `guardrails/unusual_prompt`       |
    | Supported data types          | `string`                          |
    | Programmatic fix              | N/A                               |

    Args:
        llm_callable (str): LiteLLM model string. Defaults to
            ``"gpt-3.5-turbo"``.
        on_fail (Callable, optional): Policy when validation fails.
    """

    def __init__(
        self,
        llm_callable: str = "gpt-3.5-turbo",
        on_fail: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__(on_fail, llm_callable=llm_callable, **kwargs)
        self.llm_callable = llm_callable

    def get_validation_prompt(self, value: str, detected_patterns: list[str]) -> str:
        """Build the prompt to send to the LLM.

        When intent patterns have already fired, an enrichment hint is
        appended so the LLM focuses its reasoning on the detected
        structural attack surface rather than treating the prompt as a
        generic classification task.

        Args:
            value: The user prompt to evaluate.
            detected_patterns: List of intent group labels that matched.
                Empty list when no patterns fired.

        Returns:
            The full prompt string to send to the LLM.
        """
        hint = ""
        if detected_patterns:
            joined = ", ".join(detected_patterns)
            hint = (
                f"\n\nNote: automated analysis detected the following "
                f"structural pattern(s) in this request that are commonly "
                f"associated with jailbreak attempts: {joined}. "
                f"Please weigh this when forming your answer."
            )

        prompt = f"""Request:
{value}

Question:
Is the above request unusual in a way that a human would not typically ask \
— that it is designed to trick someone into answering in a certain way, or \
to elicit a harmful response?{hint}

Please respond only with 'yes' if the request is unusual, and 'no' if it is \
not. Pay special attention to the tone, intent, and any structural framing \
of the request.
Respond with ONLY 'yes' or 'no' and do not provide any additional details."""
        return prompt

    def get_llm_response(self, prompt: str) -> str:
        """Send the prompt to the LLM and return the stripped response.

        Args:
            prompt: The full prompt to send.

        Returns:
            The LLM's response, lowercased and stripped.

        Raises:
            RuntimeError: If the LLM call fails.
        """
        messages = [{"content": prompt, "role": "user"}]

        kwargs: Dict[str, Any] = {}
        _model, provider, *_rest = get_llm_provider(self.llm_callable)
        if provider == "openai":
            kwargs["api_key"] = (
                get_call_kwarg("api_key") or os.environ.get("OPENAI_API_KEY")
            )

        try:
            completion_response = completion(
                model=self.llm_callable, messages=messages, **kwargs
            )
            response: str = completion_response.choices[0].message.content  # type: ignore
            return response.strip(" .").lower().strip()
        except Exception as e:
            raise RuntimeError(f"Error getting response from the LLM: {e}") from e

    def validate(self, value: Any, metadata: Dict) -> ValidationResult:
        """Run the two-tier validation pipeline.

        Tier 1 scans for structural jailbreak patterns. A high-confidence
        match (two or more distinct groups) short-circuits to FailResult
        without calling the LLM. A partial match enriches the LLM prompt.

        Args:
            value: The prompt string to validate.
            metadata: Validation metadata.

                ``pass_if_invalid`` (bool, default False): if the LLM
                returns something other than yes/no, pass instead of fail.

        Returns:
            ValidationResult: PassResult or FailResult.
        """
        pass_if_invalid = metadata.get("pass_if_invalid", False)

        # ------------------------------------------------------------------
        # Tier 1: intent pattern scan
        # ------------------------------------------------------------------
        detected = _scan_intent(str(value))

        if len(detected) >= _HIGH_CONFIDENCE_THRESHOLD:
            joined = ", ".join(detected)
            return FailResult(
                errorMessage=(
                    f"Prompt matches {len(detected)} structural jailbreak "
                    f"pattern groups ({joined}). Failing without LLM call."
                )
            )

        # ------------------------------------------------------------------
        # Tier 2: LLM validation (with optional enrichment hint)
        # ------------------------------------------------------------------
        prompt = self.get_validation_prompt(value, detected)
        llm_response = self.get_llm_response(prompt)

        if llm_response == "yes":
            return FailResult(
                errorMessage=(
                    "Found an unusual request being made. "
                    "Failing the validation..."
                )
            )

        if llm_response == "no":
            return PassResult()

        if pass_if_invalid:
            warn("Invalid response from the evaluator. Passing the validation...")
            return PassResult()

        return FailResult(
            errorMessage=(
                "Invalid response from the evaluator. Failing the validation..."
            )
        )

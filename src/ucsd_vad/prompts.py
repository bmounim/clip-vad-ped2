"""Text prompt sets for open-vocabulary anomaly scoring.

An honest note on what "open-vocabulary" means here. The method never
sees a labelled anomalous frame: anomalies are *described in language*
instead of learned from examples. But a vocabulary that names bicycles
and vehicles clearly encodes a prior about what counts as anomalous on a
pedestrian walkway. To measure how much that prior is worth, the
``generic`` set below deliberately avoids naming any Ped2 anomaly class,
and the ablation reports both. Treat the gap between them as the cost of
not knowing your anomalies in advance.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PromptSet:
    """Natural-language description of normal and anomalous appearance.

    ``anomaly_labels`` names each anomaly prompt so a flagged frame can be
    explained by the prompt that fired, rather than by a bare score.
    """

    name: str
    normal: list[str]
    anomaly: list[str]
    anomaly_labels: list[str] = field(default_factory=list)
    description: str = ""

    def __post_init__(self) -> None:
        if not self.normal or not self.anomaly:
            raise ValueError(f"Prompt set '{self.name}' needs both normal and anomaly prompts")
        if self.anomaly_labels and len(self.anomaly_labels) != len(self.anomaly):
            raise ValueError(f"Prompt set '{self.name}': labels/anomaly length mismatch")

    def labels(self) -> list[str]:
        return list(self.anomaly_labels) or list(self.anomaly)


# Names no anomaly class. Tests whether CLIP can flag "unusual" unaided.
GENERIC = PromptSet(
    name="generic",
    normal=[
        "a surveillance photo of pedestrians walking on a walkway",
        "a normal scene on a pedestrian footpath",
    ],
    anomaly=[
        "a surveillance photo of an unusual event on a walkway",
        "an anomalous scene on a pedestrian footpath",
    ],
    anomaly_labels=["unusual event", "anomalous scene"],
    description="No anomaly class is named; measures unaided semantic sensitivity.",
)

# One prompt per side: the smallest possible open-vocabulary specification.
MINIMAL = PromptSet(
    name="minimal",
    normal=["a photo of people walking"],
    anomaly=["a photo of a vehicle or bicycle on a footpath"],
    anomaly_labels=["vehicle or bicycle"],
    description="Single prompt per side, naming the anomaly family only.",
)

# Enumerates the anomaly categories that actually occur in Ped2.
SPECIFIC = PromptSet(
    name="specific",
    normal=[
        "a surveillance photo of pedestrians walking on a footpath",
        "a surveillance photo of people strolling outdoors",
    ],
    anomaly=[
        "a surveillance photo of a person riding a bicycle on a footpath",
        "a surveillance photo of a person riding a skateboard on a footpath",
        "a surveillance photo of a golf cart driving on a footpath",
        "a surveillance photo of a truck driving on a pedestrian walkway",
    ],
    anomaly_labels=["bicycle", "skateboard", "cart", "vehicle"],
    description="Names the Ped2 anomaly categories; yields per-frame explanations.",
)

# Prompt ensembling: several paraphrases per concept, averaged before use.
# Standard practice for zero-shot CLIP and usually worth a point or two.
ENSEMBLE = PromptSet(
    name="ensemble",
    normal=[
        "a surveillance photo of pedestrians walking on a footpath",
        "a low resolution security camera image of people walking",
        "a grayscale photo of pedestrians on a campus walkway",
        "people walking normally outdoors",
    ],
    anomaly=[
        "a surveillance photo of a person riding a bicycle on a footpath",
        "a low resolution security camera image of a cyclist among pedestrians",
        "a surveillance photo of a person on a skateboard among pedestrians",
        "a surveillance photo of a small vehicle driving on a pedestrian walkway",
        "a grayscale photo of a cart moving among people on a footpath",
    ],
    anomaly_labels=[
        "bicycle",
        "cyclist",
        "skateboard",
        "vehicle",
        "cart",
    ],
    description="Paraphrase ensemble over both sides; the default configuration.",
)

PROMPT_SETS: dict[str, PromptSet] = {
    ps.name: ps for ps in (GENERIC, MINIMAL, SPECIFIC, ENSEMBLE)
}


def get_prompt_set(name: str) -> PromptSet:
    if name not in PROMPT_SETS:
        raise KeyError(f"Unknown prompt set '{name}'. Available: {sorted(PROMPT_SETS)}")
    return PROMPT_SETS[name]

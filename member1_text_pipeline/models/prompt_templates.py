"""
Prompt Templates for Object-Conditioned Text Generation.

Generates normal-state text descriptions for each product class.
These templates define what a "defect-free" product looks like semantically,
and serve as input to the CLIP text encoder.

Based on the prompt engineering strategy from:
  "Text-Guided Multimodal Unified Industrial Anomaly Detection" (arXiv 2604.22899)
  Inspired by WinCLIP and AnomalyGPT prompt designs.
"""

from typing import List, Dict


# ──────────────────────────────────────────────
# Default tech product classes
# ──────────────────────────────────────────────
DEFAULT_CLASSES = [
    "bagel",
    "cable_gland",
    "carrot",
    "cookie",
    "dowel",
    "foam",
    "peach",
    "potato",
    "rope",
    "tire",
    "phone_screen",
    "car_metal",
    "pcb"
]

# ──────────────────────────────────────────────
# Normal-state descriptors
# ──────────────────────────────────────────────
# These describe a defect-free product from multiple linguistic angles.
# The model learns that anomalies are deviations from these descriptions.

STATE_DESCRIPTORS = [
    "{c}",                      # direct class label
    "flawless {c}",             # state enhancement
    "perfect {c}",
    "unblemished {c}",
    "pristine {c}",
    "intact {c}",
    "{c} without flaw",         # explicit exclusionary
    "{c} without defect",
    "{c} without damage",
    "{c} without scratch",
    "{c} without crack",
    "{c} without dent",
]

# ──────────────────────────────────────────────
# Contextual prompt wrappers
# ──────────────────────────────────────────────
# Embedding state descriptors into natural-language templates
# for compatibility with CLIP's pre-training distribution.

CONTEXT_TEMPLATES = [
    "a photo of a {s}.",
    "a photo of the {s}.",
    "a close-up photo of a {s}.",
    "a good photo of a {s}.",
    "a photo of a clean {s}.",
    "a photo of a normal {s}.",
    "an image of a {s} in good condition.",
]


class PromptGenerator:
    """
    Generates normal-state text prompts for each product class.

    For a given class name (e.g., "mobile phone"), it produces a list of
    diverse prompt strings like:
        - "a photo of a flawless mobile phone."
        - "a close-up photo of a mobile phone without scratch."
        - ...

    These prompts are then tokenized and encoded by the CLIP text encoder
    to produce text embeddings that anchor the "normal" semantic space.

    Args:
        classes: List of product class names. Defaults to tech product classes.
        state_descriptors: List of state descriptor templates with {c} placeholder.
        context_templates: List of context wrapper templates with {s} placeholder.
    """

    def __init__(
        self,
        classes: List[str] = None,
        state_descriptors: List[str] = None,
        context_templates: List[str] = None,
    ):
        self.classes = classes or DEFAULT_CLASSES
        self.state_descriptors = state_descriptors or STATE_DESCRIPTORS
        self.context_templates = context_templates or CONTEXT_TEMPLATES

    def generate_prompts(self, class_name: str) -> List[str]:
        """
        Generate all normal-state prompts for a single class.

        Args:
            class_name: Product class name (e.g., "mobile phone").

        Returns:
            List of prompt strings (len = num_states × num_contexts).

        Example:
            >>> gen = PromptGenerator()
            >>> prompts = gen.generate_prompts("mobile phone")
            >>> prompts[0]
            'a photo of a mobile phone.'
            >>> len(prompts)
            84
        """
        prompts = []
        for state_template in self.state_descriptors:
            # Fill in the class name: "flawless {c}" -> "flawless mobile phone"
            state = state_template.format(c=class_name)
            for context_template in self.context_templates:
                # Wrap in context: "a photo of a {s}." -> "a photo of a flawless mobile phone."
                prompt = context_template.format(s=state)
                prompts.append(prompt)
        return prompts

    def generate_all_prompts(self) -> Dict[str, List[str]]:
        """
        Generate prompts for all registered classes.

        Returns:
            Dictionary mapping class_name -> list of prompt strings.

        Example:
            >>> gen = PromptGenerator()
            >>> all_prompts = gen.generate_all_prompts()
            >>> list(all_prompts.keys())
            ['mobile phone', 'laptop', 'car body panel', ...]
        """
        return {cls: self.generate_prompts(cls) for cls in self.classes}

    def get_num_prompts_per_class(self) -> int:
        """Return the number of prompts generated per class."""
        return len(self.state_descriptors) * len(self.context_templates)

    def get_classes(self) -> List[str]:
        """Return the list of registered product classes."""
        return list(self.classes)

    def generate_state_prompts(self, class_name: str, is_anomaly: bool = False) -> List[str]:
        """Generate normal or anomaly state prompts for a specified category."""
        if not is_anomaly:
            return [
                f"a photo of a flawless intact {class_name}.",
                f"a photo of a perfect normal {class_name}.",
                f"a photo of a clean undamaged {class_name}."
            ]
        else:
            return [
                f"a photo of a damaged defective {class_name} with missing piece.",
                f"a photo of a broken {class_name} with cut, crack, or hole.",
                f"a photo of a contaminated bent {class_name} with defect."
            ]

    def __repr__(self) -> str:
        return (
            f"PromptGenerator("
            f"classes={len(self.classes)}, "
            f"states={len(self.state_descriptors)}, "
            f"contexts={len(self.context_templates)}, "
            f"prompts_per_class={self.get_num_prompts_per_class()})"
        )


# ──────────────────────────────────────────────
# Quick demo
# ──────────────────────────────────────────────
if __name__ == "__main__":
    gen = PromptGenerator()
    print(gen)
    print(f"\nClasses: {gen.get_classes()}")
    print(f"Prompts per class: {gen.get_num_prompts_per_class()}")

    # Show sample prompts for "mobile phone"
    prompts = gen.generate_prompts("mobile phone")
    print(f"\n--- Sample prompts for 'mobile phone' ({len(prompts)} total) ---")
    for i, p in enumerate(prompts[:10]):
        print(f"  [{i+1}] {p}")
    print(f"  ... and {len(prompts) - 10} more")

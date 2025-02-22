from typing import Any, Dict, Type

from src.frameworks.base import BaseFramework
from src.frameworks.outlines_framework import OutlinesFramework
from src.frameworks.pse_framework import PSEFramework
from src.frameworks.lm_format_enforcer_framework import LMFormatEnforcerFramework

# Registry of available frameworks
FRAMEWORK_REGISTRY: Dict[str, Type[BaseFramework]] = {
    "OutlinesFramework": OutlinesFramework,
    "PSEFramework": PSEFramework,
    "LMFormatEnforcerFramework": LMFormatEnforcerFramework,
}


def factory(class_name: str, *args: Any, **kwargs: Any) -> BaseFramework:
    """Create an instance of a framework class.

    This factory function instantiates a framework class based on its name.
    All frameworks must inherit from BaseFramework and implement its interface.

    Args:
        class_name: Name of the framework class to instantiate
        *args: Positional arguments to pass to the framework constructor
        **kwargs: Keyword arguments to pass to the framework constructor

    Returns:
        An instance of the requested framework class

    Raises:
        ValueError: If the class name is not found in the registry
        TypeError: If the class does not inherit from BaseFramework

    Examples:
        >>> framework = factory("PSEFramework", task="ner", device="cuda")
        >>> framework = factory("OutlinesFramework", task="multilabel_classification")
        >>> framework = factory("LMFormatEnforcerFramework", task="function_calling")
    """
    if class_name not in FRAMEWORK_REGISTRY:
        available = ", ".join(FRAMEWORK_REGISTRY.keys())
        raise ValueError(
            f"Invalid framework name: {class_name}. Available frameworks: {available}"
        )

    framework_class = FRAMEWORK_REGISTRY[class_name]

    # Type safety check - should never fail due to registry construction
    if not issubclass(framework_class, BaseFramework):
        raise TypeError(f"Framework {class_name} must inherit from BaseFramework")

    return framework_class(*args, **kwargs)


__all__ = [
    "factory",
    "OutlinesFramework",
    "PSEFramework",
    "LMFormatEnforcerFramework",
    "FRAMEWORK_REGISTRY",
]

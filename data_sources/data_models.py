import dataclasses
from typing import Any

from pydantic import BaseModel
from pydantic_core import PydanticUndefined


class UserAddress(BaseModel):
    """A model representing a user's physical address."""

    street: str
    """The street name and number"""
    city: str
    """The city name"""
    country: str
    """The country name"""


class User(BaseModel):
    """A model representing a user with their basic information and address."""

    name: str
    """The user's full name"""
    age: int
    """The user's age in years"""
    address: UserAddress
    """The user's physical address details"""

    @staticmethod
    def calculate_diversity_score(users: list["User"]) -> float:
        """
        Calculate a diversity score for a list of users (0.0 to 1.0).
        Considers age distribution, name uniqueness, and geographic diversity.
        """
        if not users:
            return 0.0

        # Name uniqueness (ratio of unique names)
        unique_names = len(set(user.name for user in users))
        name_diversity = unique_names / len(users)

        # Geographic diversity (unique locations)
        unique_locations = len(
            set((user.address.city, user.address.country) for user in users)
        )
        location_diversity = unique_locations / len(users)

        # Combined score (equal weights)
        return (name_diversity + location_diversity) / 2.0


class FunctionCall(BaseModel):
    name: str
    arguments: dict[str, Any]

    @staticmethod
    def compare(
        expected: "FunctionCall",
        generated: "FunctionCall",
    ) -> tuple[float, float]:
        # Score between 0.0 and 1.0
        name_match = float(expected.name == generated.name)

        if not name_match:
            return 0.0, 0.0

        # Compare arguments presence
        expected_keys = set(expected.arguments.keys())
        generated_keys = set(generated.arguments.keys())
        common_keys = expected_keys & generated_keys

        # Argument presence score - handle both empty cases
        args_presence = (
            len(common_keys) / max(len(expected_keys), len(generated_keys))
            if (expected_keys or generated_keys)
            else 1.0
        )

        return name_match, args_presence


def pydantic_to_dataclass(
    klass: type[BaseModel],
    classname: str | None = None,
) -> Any:
    """
    Dataclass from Pydantic model

    Transferred entities:
        * Field names
        * Type annotations, except of Annotated etc
        * Default factory or default value

    Validators are not transferred.

    Order of fields may change due to dataclass's positional arguments.

    """

    dataclass_args = []
    for name, info in klass.model_fields.items():
        if info.default_factory is not None:
            dataclass_field = dataclasses.field(
                default_factory=info.default_factory,  # type: ignore
            )
            dataclass_arg = (name, info.annotation, dataclass_field)
        elif info.default is not PydanticUndefined:
            dataclass_field = dataclasses.field(
                default=info.get_default(),
            )
            dataclass_arg = (name, info.annotation, dataclass_field)
        else:
            dataclass_arg = (name, info.annotation)
        dataclass_args.append(dataclass_arg)
    dataclass_args.sort(key=lambda arg: len(arg) > 2)
    return dataclasses.make_dataclass(
        classname or f"{klass.__name__}",
        dataclass_args,
    )

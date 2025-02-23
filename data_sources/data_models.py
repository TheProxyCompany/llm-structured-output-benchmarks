import dataclasses
from enum import Enum
from typing import Any

from pydantic import BaseModel
from pydantic_core import PydanticUndefined


class UserAddress(BaseModel):
    street: str
    city: str
    country: str


class User(BaseModel):
    name: str
    age: int
    address: UserAddress

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
    ) -> tuple[float, float, float]:
        # Score between 0.0 and 1.0
        name_match = float(expected.name == generated.name)

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

        # Argument correctness score (for present arguments)
        args_correctness = 0.0
        if common_keys:
            correct_args = sum(
                1
                for key in common_keys
                if expected.arguments[key] == generated.arguments[key]
            )
            args_correctness = correct_args / len(common_keys)

        return name_match, args_presence, args_correctness


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

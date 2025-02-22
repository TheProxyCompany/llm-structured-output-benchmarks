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

class FunctionCall(BaseModel):
    name: str
    arguments: dict[str, Any]


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

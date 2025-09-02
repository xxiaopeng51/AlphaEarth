from typing import Callable, Dict, Any


class DatasetRegistry:
	_builders: Dict[str, Callable[..., Any]] = {}

	@classmethod
	def register(cls, name: str, builder: Callable[..., Any]) -> None:
		cls._builders[name] = builder

	@classmethod
	def build(cls, name: str, **kwargs: Any) -> Any:
		if name not in cls._builders:
			raise KeyError(f"Dataset '{name}' is not registered")
		return cls._builders[name](**kwargs)
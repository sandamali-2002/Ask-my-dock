import os
from pathlib import Path


def _load_env_file() -> None:
	"""Load key=value pairs from a local .env file into os.environ."""
	env_path = Path(__file__).resolve().parent / ".env"
	if not env_path.exists():
		return

	for line in env_path.read_text(encoding="utf-8").splitlines():
		line = line.strip()
		if not line or line.startswith("#") or "=" not in line:
			continue
		key, value = line.split("=", 1)
		key = key.strip()
		value = value.strip().strip('"').strip("'")
		if key and key not in os.environ:
			os.environ[key] = value


_load_env_file()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
	raise RuntimeError(
		"OPENAI_API_KEY is not set. Add it to system env vars or .env in project root."
	)
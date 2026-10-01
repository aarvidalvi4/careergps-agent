"""Force the mock provider for every test, even if .env configures a real one.

load_dotenv() never overrides variables that are already set, so pinning
LLM_PROVIDER here keeps tests offline regardless of the developer's .env.
"""

import os

os.environ["LLM_PROVIDER"] = "mock"

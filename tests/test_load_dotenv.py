import os
from dotenv import load_dotenv

def test_load_dotenv() ->None:
    """Asserts that after loading env variables of this project the total number of env variables available via os is increased. This proxies that the .env import was succesful.
    """    
    env_vars_before = len(os.environ)
    load_dotenv('.env.md')
    env_vars_after = len(os.environ)
    assert env_vars_after > env_vars_before
    
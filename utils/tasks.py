def get_task_title(env_name: str) -> str:
    if env_name.startswith("antsoccer"):
        return "Antsoccer"
    elif env_name.startswith("cube"):
        return "Cube"
    else:
        return "Unknown Task"
    
def get_task_filename(env_name: str) -> str:
    if env_name.startswith("antsoccer"):
        return "antsoccer"
    elif env_name.startswith("cube"):
        return "cube"
    else:
        return "unknown"

class Tracer:
    def __init__(self):
        self.state = []
        
    def append_log(self, step_info: str):
        """Append logs to a state dictionary."""
        self.state.append(step_info)
        
    @property
    def logs(self) -> list:
        return list(self.state)

    @property
    def steps(self) -> list:
        return list(self.state)

    def get_trace(self) -> dict:
        """Packages the log to be sent back, providing both 'steps' and 'logs' for full compatibility."""
        return {
            "steps": list(self.state),
            "logs": list(self.state),
            "summary": " | ".join(self.state) if self.state else ""
        }


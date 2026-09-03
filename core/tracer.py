class Tracer:
    def __init__(self):
        self.state = []
        
    def append_log(self, step_info: str):
        """Append logs to a state dictionary."""
        self.state.append(step_info)
        
    def get_trace(self) -> dict:
        """Packages the log (e.g. 'step 1: aligned... confidence: 89%') to be sent back."""
        return {
            "steps": self.state,
            "summary": " | ".join(self.state) if self.state else ""
        }

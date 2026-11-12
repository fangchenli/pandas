"""Optional diagnostics contain numbers, never references to live storage."""


class Metrics:
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.events = []

    def record(self, kind, **details):
        if self.enabled:
            self.events.append({"kind": kind, **details})

    def count(self, kind):
        return sum(event["kind"] == kind for event in self.events)

    def clear(self):
        self.events.clear()

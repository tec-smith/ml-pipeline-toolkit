"""Log setup for the command line: readable text, or JSON for log pipelines."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime

LEVELS = ("debug", "info", "warning", "error")
FORMATS = ("text", "json")


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "time": datetime.fromtimestamp(record.created, UTC).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry)


def configure(level: str = "info", fmt: str = "text") -> None:
    handler = logging.StreamHandler(sys.stderr)
    if fmt == "json":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S")
        )
    handler.set_name("mlpipeline")
    root = logging.getLogger()
    # Replace our own handler on repeat calls, but leave anyone else's alone.
    root.handlers[:] = [h for h in root.handlers if h.get_name() != "mlpipeline"]
    root.addHandler(handler)
    root.setLevel(level.upper())


# ***
#   @COPYRIGHT (c) 2026. Thomas EC. Smith (https://www.TECSmith.info). Licensed under the MIT License.
# ****

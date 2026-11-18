"""vLLM's native scheduler payload with a worker-local index invalidation."""
from dataclasses import dataclass, fields

from vllm.v1.core.sched.output import SchedulerOutput


@dataclass
class RepairSchedulerOutput(SchedulerOutput):
    bfla_repaired_req_ids: tuple[str, ...] = ()
    bfla_rewound_req_ids: tuple[str, ...] = ()

    @classmethod
    def from_step(cls, step, affected, rewound):
        # Preserve all native fields, including opaque connector metadata.
        # Native worker copies made with dataclasses.replace retain both fields.
        native = {field.name: getattr(step, field.name) for field in fields(step)}
        return cls(**native, bfla_repaired_req_ids=tuple(sorted(affected)),
                   bfla_rewound_req_ids=tuple(sorted(rewound)))

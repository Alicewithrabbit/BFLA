"""Adapter environment handling; operator configuration remains explicit."""
from dataclasses import dataclass, fields
from bfla import BFLAConfig


@dataclass(frozen=True)
class Settings:
    config: BFLAConfig = BFLAConfig()
    execution: str = 'eager'
    enabled: bool = True

    @classmethod
    def from_env(cls, env=None):
        if env is None:
            import os
            env = os.environ
        execution = env.get('BFLA_EXECUTION', 'eager')
        if execution not in ('eager', 'localgraph'):
            raise ValueError('BFLA_EXECUTION must be eager or localgraph')
        enabled = env.get('BFLA_ENABLED', '1')
        if enabled not in ('0', '1'):
            raise ValueError('BFLA_ENABLED must be 0 or 1')
        values = {}
        for field in fields(BFLAConfig):
            raw = env.get('BFLA_' + field.name.upper())
            if raw is not None:
                if field.name == 'k_max' and raw.lower() == 'none':
                    values[field.name] = None
                else:
                    values[field.name] = raw if field.name == 'index_dtype' else int(raw)
        return cls(BFLAConfig(**values), execution, enabled == '1')

"""Offline joint-angle clipping for model chunks; no robot/network operations."""
import hashlib
import json
import math
from pathlib import Path


class JointLimits:
    def __init__(self, path):
        raw = Path(path).read_bytes()
        cfg = json.loads(raw)
        if cfg.get('schema') != 'model-joint-limits/1' or cfg.get('units') != 'rad':
            raise ValueError('Unsupported joint-limit schema or units')
        if not isinstance(cfg.get('version'), str) or not cfg['version']:
            raise ValueError('Joint-limit version is required')
        if cfg.get('robot_host') != '192.168.50.226' or cfg.get('robot_model') != 'G2_t2v2_crsB':
            raise ValueError('Joint-limit robot identity mismatch')
        self.bounds = {}
        for arm, side, first in [('left_arm', 'l', 21), ('right_arm', 'r', 61)]:
            joints = cfg[arm]['joints']
            if len(joints) != 7:
                raise ValueError('Expected seven ordered joints: ' + arm)
            bounds = []
            for index, joint in enumerate(joints):
                name = f'idx{first + index}_arm_{side}_joint{index + 1}'
                if joint['name'] != name:
                    raise ValueError('Joint order/name mismatch: ' + name)
                ranges = []
                for field in ('position_hardware', 'position', 'position_ik'):
                    if field == 'position_ik' and field not in joint:
                        continue
                    pair = joint[field]
                    if (not isinstance(pair, list) or len(pair) != 2
                            or any(type(v) not in (int, float) or not math.isfinite(v) for v in pair)
                            or pair[0] >= pair[1]):
                        raise ValueError('Invalid range: ' + name + '/' + field)
                    ranges.append(pair)
                lower, upper = max(p[0] for p in ranges), min(p[1] for p in ranges)
                if lower >= upper:
                    raise ValueError('Empty limit intersection: ' + name)
                bounds.append({'name': name, 'lower': lower, 'upper': upper})
            self.bounds[arm] = bounds
        self.reference = {
            'schema': cfg['schema'], 'version': cfg['version'],
            'config_sha256': hashlib.sha256(raw).hexdigest(),
            'robot_host': cfg['robot_host'], 'robot_model': cfg['robot_model'],
            'units': 'rad', 'method': 'clip_intersection_position_hardware_position_optional_ik',
            'effective_limits': self.bounds,
            'source': {arm: cfg[arm]['source'] for arm in self.bounds},
        }

    def clip(self, chunk):
        # Return a fresh copy, preserving both grippers and all raw input values.
        if hasattr(chunk, 'tolist'):
            chunk = chunk.tolist()
        if len(chunk) != 30 or any(len(row) != 16 for row in chunk):
            raise ValueError('Model chunk must be 30x16')
        if any(type(v) not in (int, float) or not math.isfinite(v) for row in chunk for v in row):
            raise ValueError('Model chunk must contain finite numeric values')
        result = [list(row) for row in chunk]
        changes = []
        for arm, offset in [('left_arm', 0), ('right_arm', 8)]:
            for j, limit in enumerate(self.bounds[arm]):
                col = offset + j
                for row, values in enumerate(result):
                    before = values[col]
                    after = min(limit['upper'], max(limit['lower'], before))
                    if before != after:
                        values[col] = after
                        changes.append({'row_index': row, 'action_column': col,
                                        'joint': limit['name'], 'raw_rad': before,
                                        'prepared_target_rad': after, 'delta_rad': after - before})
        report = {**self.reference, 'applied': bool(changes), 'changed_values': len(changes),
                  'changed_rows': len({c['row_index'] for c in changes}),
                  'max_abs_delta_rad': max((abs(c['delta_rad']) for c in changes), default=0.0),
                  'index_base': 0, 'changes': changes}
        return result, report

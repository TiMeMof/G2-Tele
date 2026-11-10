"""Read-only decoding and candidate takeover observation; never robot control."""
import math
import struct

SCHEMA = "rh_msgs_v4.RobotFrameData"
PARSER_VERSION = "a2-20260922.1"
SCHEMA_FILE_SHA256 = "6efc0c66e6c71e02856313b8dc6f9fbddab58c0a030e740b1feeb6658e89f00e"


def varint(data, offset):
    value = 0
    for shift in range(0, 70, 7):
        if offset >= len(data):
            raise ValueError("truncated varint")
        part = data[offset]
        offset += 1
        if shift == 63 and part > 1:
            raise ValueError("varint overflow")
        value |= (part & 127) << shift
        if part < 128:
            return value, offset
    raise ValueError("long varint")


def fields(data):
    offset = 0
    while offset < len(data):
        tag, offset = varint(data, offset)
        number, kind = tag >> 3, tag & 7
        if not 0 < number < (1 << 29):
            raise ValueError("invalid field number")
        if kind == 0:
            value, offset = varint(data, offset)
        elif kind in (1, 2, 5):
            if kind == 2:
                size, offset = varint(data, offset)
            else:
                size = 8 if kind == 1 else 4
            if offset + size > len(data):
                raise ValueError("truncated field")
            value = data[offset:offset + size]
            offset += size
        else:
            raise ValueError("unsupported wire type")
        yield number, kind, value


def decode_vr(payload):
    """Uses verified installed descriptor. Scalar omission follows the descriptor defaults."""
    timestamp = frame = 0
    devices = []
    for number, kind, value in fields(payload):
        if number in (1, 2):
            if kind != 0:
                raise ValueError("invalid frame scalar")
            if number == 1:
                timestamp = value
            else:
                frame = value
        elif number == 103:
            if kind != 2:
                raise ValueError("invalid device_inputs wire type")
            device = {"device_type": 0, "axis_click": False, "axis_x": 0.0,
                      "axis_y": 0.0, "index_trig": 0.0, "hand_trig": 0.0,
                      "key_one": False, "key_two": False}
            for n, w, v in fields(value):
                if n in (1, 2, 7, 8):
                    if w != 0:
                        raise ValueError("invalid device scalar")
                    key = {1: "device_type", 2: "axis_click", 7: "key_one", 8: "key_two"}[n]
                    if n != 1 and v not in (0, 1):
                        raise ValueError("invalid button value")
                    device[key] = v if n == 1 else bool(v)
                elif n in (3, 4, 5, 6):
                    if w != 5:
                        raise ValueError("invalid device float")
                    key = {3: "axis_x", 4: "axis_y", 5: "index_trig", 6: "hand_trig"}[n]
                    device[key] = struct.unpack("<f", v)[0]
                    if not math.isfinite(device[key]):
                        raise ValueError("nonfinite input")
            for key in ("index_trig", "hand_trig"):
                if not 0 <= device[key] <= 1:
                    raise ValueError("trigger outside [0,1]")
            device["side"] = {0: "left", 1: "right", 2: "head"}.get(device["device_type"], "unknown")
            devices.append(device)
    ids = [d["device_type"] for d in devices]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate device IDs")
    return {"schema": SCHEMA, "timestamp_raw_u64": str(timestamp),
            "frame_index_u64": str(frame), "devices": devices,
            "both_hands_present": 0 in ids and 1 in ids}


class Observer:
    """Candidate button state, not verified control ownership. No resume action exists."""
    def __init__(self, on=0.2, off=0.2, stale_ms=250):
        if not (math.isfinite(on) and math.isfinite(off) and 0 <= off <= on <= 1):
            raise ValueError("thresholds must satisfy 0 <= off <= on <= 1")
        if not math.isfinite(stale_ms) or stale_ms <= 0:
            raise ValueError("stale_ms must be positive")
        self.on, self.off, self.stale_ns = on, off, int(stale_ms * 1e6)
        self.last_rx = None
        self.frame = None
        self.source_ts = None
        self.state = {"left": None, "right": None}
        self.error = "no_vr_frames"
        self.pause_latched = False
        self.last_identity = None
        self.publisher = None
        self.unique_frames = 0
        self.last_decoded = None
        self.events = []

    def fault(self, reason, now_ns):
        self.error = reason
        self.state = {"left": None, "right": None}
        self.events.append({"kind": "unknown", "reason": reason, "mono_ns": str(now_ns)})

    def update(self, decoded, now_ns, publisher, sequence):
        identity = (publisher, sequence)
        if identity == self.last_identity:
            return False
        # An old or duplicate transport frame is not fresh evidence.
        if self.publisher == publisher and self.last_identity is not None and sequence <= self.last_identity[1]:
            self.fault("transport_sequence_not_increasing", now_ns)
            return False
        changed_publisher = self.publisher is not None and self.publisher != publisher
        self.last_identity, self.publisher = identity, publisher
        frame = int(decoded["frame_index_u64"])
        source_ts = int(decoded["timestamp_raw_u64"])
        # The native forwarder can repeat a VR frame with a new DDS sequence.
        # Retain it in raw storage, but do not refresh input freshness or emit edges.
        if not changed_publisher and frame == self.frame and source_ts == self.source_ts:
            if decoded != self.last_decoded:
                self.fault("conflicting_duplicate_device_frame", now_ns)
            return False
        advancing = self.frame is None or (frame > self.frame and source_ts >= self.source_ts)
        if source_ts == 0:
            self.fault("missing_source_timestamp", now_ns)
            return False
        self.frame, self.source_ts = frame, source_ts
        self.last_decoded = decoded
        self.unique_frames += 1
        if changed_publisher or not advancing:
            self.fault("source_restart_or_nonadvancing_frame", now_ns)
            return True
        if not decoded["both_hands_present"]:
            self.fault("missing_hand", now_ns)
            return True
        was_unknown = self.error is not None or self.last_rx is None or now_ns - self.last_rx > self.stale_ns
        self.last_rx = now_ns
        self.error = None
        for d in decoded["devices"]:
            side = d["side"]
            if side not in self.state:
                continue
            previous = None if was_unknown else self.state[side]
            value = d["hand_trig"]
            # Comparator is explicitly provisional until runtime/physical calibration.
            active = value > (self.off if previous else self.on)
            self.state[side] = active
            if active:
                self.pause_latched = True
            if previous is None or previous != active:
                self.events.append({"kind": "initial_or_reacquired_state" if previous is None else "candidate_edge",
                                    "side": side, "active": active, "value": value,
                                    "mono_ns": str(now_ns), "frame_index_u64": str(frame)})
        return True

    def snapshot(self, now_ns):
        fresh = self.last_rx is not None and now_ns - self.last_rx <= self.stale_ns and self.error is None
        reason = self.error or (None if fresh else "vr_stale")
        state = dict(self.state) if fresh else {"left": None, "right": None}
        return {"fresh": fresh, "reason": reason, "candidate_arm_input": state,
                "age_ms": None if self.last_rx is None else (now_ns-self.last_rx)/1e6,
                "pause_latched": self.pause_latched,
                "resume_allowed": False, "control_ownership": "unverified",
                "control_ready": False, "unique_frames": self.unique_frames,
                "threshold": {"on": self.on, "off": self.off, "comparison": ">",
                              "status": "config_candidate_not_calibrated"}}


def pbdat_records(path):
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError("truncated container")
        count, = struct.unpack("<Q", raw)
        for _ in range(count):
            raw = f.read(40)
            if len(raw) != 40:
                raise ValueError("truncated record header")
            header = struct.unpack("<5Q", raw)
            if header[4] > 8_000_000:
                raise ValueError("oversized payload")
            data = f.read(header[4])
            if len(data) != header[4]:
                raise ValueError("truncated payload")
            yield header, data
        if f.read(1):
            raise ValueError("trailing data")



def decode_network(payload):
    """Installed VRNetwork descriptor; presence is retained rather than filling zeros."""
    mapping = {491747779: ("is_connected", 0), 109035493: ("frequency", 5),
               389577674: ("latency", 5), 13278980: ("packet_loss_rate", 5),
               387630747: ("device_name", 2), 50322782: ("ip_address", 2),
               512328240: ("remote_ip_address", 2)}
    result = {}
    for n, w, v in fields(payload):
        if n not in mapping:
            continue
        key, expected = mapping[n]
        if w != expected:
            raise ValueError("invalid network field")
        if w == 0:
            if v not in (0,1):
                raise ValueError("invalid network boolean")
            result[key] = bool(v)
        elif w == 5:
            result[key] = struct.unpack("<f",v)[0]
            if not math.isfinite(result[key]):
                raise ValueError("invalid network float")
        else:
            result[key] = v.decode("utf-8")
    return result


def decode_operator_event(payload):
    """An input REQUEST is intent, never proof that robot accepted control."""
    result = {"semantics": "operator_request_not_acceptance"}
    for n,w,v in fields(payload):
        if n in (1,2,3):
            if w != 0:
                raise ValueError("invalid operator scalar")
            result[{1:"timestamp_raw_u64",2:"operator_event_code",3:"operator_event_detail_code"}[n]] = str(v) if n==1 else v
        elif n in (4,5,6,7):
            if w != 2:
                raise ValueError("invalid operator string")
            result[{4:"operator_device",5:"robot_aid",6:"vr_id",7:"task_uuid"}[n]]=v.decode("utf-8")
    return result

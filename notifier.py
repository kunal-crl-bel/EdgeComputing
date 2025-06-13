import time
import socket
import struct


class Notifier:
    HEADER_FMT = "<BBBHBB"  # src_csci, dest_csci, msg_id, msg_len, src_unit, dest_unit
    NOOBJ_FMT = "<B"
    OBJECT_FMT = "<BB"

    def __init__(self, cfg: dict):
        self.cooldown = cfg['cooldown_seconds']
        self.last_sent = 0
        self.pending: list[int] = []
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.dst = (cfg['udp_ip'], cfg['udp_port'])
        self.src_csci_id = cfg['src_csci_id']
        self.dest_csci_id = cfg['dest_csci_id']
        self.src_unit_id = cfg['src_unit_id']
        self.dest_unit_id = cfg['dest_unit_id']
        self.msg_id = 0

    def add(self, track_id: int) -> None:
        self.pending.append(track_id)
        self._try_send()

    def _try_send(self) -> None:
        if (time.time() - self.last_sent) >= self.cooldown and self.pending:
            self._send()
            self.pending.clear()
            self.last_sent = time.time()

    def _send(self) -> None:
        no_of_objects = len(self.pending)
        objs = b''.join(
            struct.pack(self.OBJECT_FMT, t_id & 0xFF, 1)
            for t_id in self.pending
        )
        payload = struct.pack(self.NOOBJ_FMT, no_of_objects) + objs
        msg_len = len(payload)
        header = struct.pack(
            self.HEADER_FMT,
            self.src_csci_id,
            self.dest_csci_id,
            self.msg_id & 0xFF,
            msg_len,
            self.src_unit_id,
            self.dest_unit_id
        )
        self.sock.sendto(header + payload, self.dst)
        print(f"[Notifier] Sent msg_id={self.msg_id}, objects={self.pending}")
        self.msg_id += 1

    def close(self) -> None:
        self.sock.close()
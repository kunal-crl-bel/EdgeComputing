import time
import socket
import struct
import threading
from concurrent.futures import ThreadPoolExecutor

class Notifier:
    def __init__(self, cfg: dict):
        # Read all parameters from cfg
        self.cooldown = cfg['cooldown_seconds']
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.dst = (cfg['udp_ip'], cfg['udp_port'])

        # Header fields from cfg
        self.src_csci_id = cfg['src_csci_id']
        self.dst_csci_id = cfg['dest_csci_id']
        self.src_unit_id = cfg['src_unit_id']
        self.dst_unit_id = cfg['dest_unit_id']

        # Initial message ID and thread pool size
        self.msg_id = cfg.get('msg_id', 0) & 0xFF
        max_workers = cfg.get('parallel_workers_udp_msg', 4)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # Pending entries buffer
        self.pending: list[tuple[int, int]] = []
        self.lock = threading.Lock()
        self.timer: threading.Timer | None = None

    def add(self, code: int, count: int) -> None:
        """
        Buffer entries and send all pending after cooldown elapses.
        If no new add() calls arrive during cooldown, pending will flush via timer.

        code: numeric object type code (e.g., SRS value of object type)
        count: number of objects of this type
        """
        with self.lock:
            entry = (code & 0xFF, count & 0xFF)
            self.pending.append(entry)

            # If no timer is active, start one to flush after cooldown
            if not self.timer or not self.timer.is_alive():
                self.timer = threading.Timer(self.cooldown, self._flush)
                self.timer.start()

    def _flush(self) -> None:
        """Flush pending entries: send them in one packet."""
        with self.lock:
            if not self.pending:
                return
            entries_to_send = list(self.pending)
            self.pending.clear()
        # Dispatch send outside lock
        self.executor.submit(self._send, entries_to_send)

    def _send(self, entries: list[tuple[int, int]]) -> None:
        tmp_map = dict()
        for i in entries:
            tmp_map[i[0]] = i[1]
        if tmp_map.get(0,None):
            tmp_map[0]=None
        
        if not tmp_map:
            return 
        
        msg = self._create_detected_object_message(tmp_map)
        self.sock.sendto(msg, self.dst)
        # print(f"[Notifier] Sent msg_id={self.msg_id}, entries={list(map(lambda x: f"{x}:{tmp_map[x]}", tmp_map))}")
        del tmp_map

        self.msg_id = (self.msg_id + 1) & 0xFF

    def _create_message_header(self, msg_len: int) -> bytes:
        return struct.pack(
            '<BBBHBB',
            self.src_csci_id,
            self.dst_csci_id,
            self.msg_id,
            msg_len,
            self.src_unit_id,
            self.dst_unit_id
        )

    def _create_object_data_entry(self, object_type_code: int, count: int) -> bytes:
        return struct.pack('<BB', object_type_code, count)

    def _create_detected_object_message(self, entries: dict[tuple[int, int]]) -> bytes:
        # entries: dict of (code, count)
        payload = struct.pack('<B', len(entries))
        for (code, cnt) in entries.items():
            payload += self._create_object_data_entry(code, cnt)
        
        header = self._create_message_header(len(payload))
        return header + payload

    def close(self) -> None:
        # Cancel pending timer
        if self.timer and self.timer.is_alive():
            self.timer.cancel()
        # Flush any remaining pending entries
        self._flush()
        self.executor.shutdown(wait=True)
        self.sock.close()
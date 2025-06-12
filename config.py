import xml.etree.ElementTree as ET


def load_config(config_path: str) -> dict:
    tree = ET.parse(config_path)
    root = tree.getroot()
    cfg = {}
    for child in root:
        if child.text is not None:
            cfg[child.tag] = child.text.strip()

    # Convert to proper types
    converters = {
        'similarity_threshold': float,
        'cooldown_seconds': int,
        'max_db_size_mb': float,
        'track_threshold': float,
        'parallel_workers': int,
        'udp_port': int,
        'src_csci_id': int,
        'dest_csci_id': int,
        'src_unit_id': int,
        'dest_unit_id': int
    }
    for key, conv in converters.items():
        if key in cfg:
            cfg[key] = conv(cfg[key])

    # Defaults
    cfg.setdefault('udp_ip', '127.0.0.1')
    return cfg
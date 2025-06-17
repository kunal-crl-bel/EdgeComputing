import xml.etree.ElementTree as ET


def parse_element(element) -> dict | str:
    # If element has no child, return stripped text
    if len(element) == 0:
        return element.text.strip() if element.text else ""
    
    # Otherwise, return a nested dictionary
    result = {}
    for child in element:
        result[child.tag] = parse_element(child)
    return result


def load_config(config_path: str) -> dict:
    tree = ET.parse(config_path)
    root = tree.getroot()
    cfg = {}

    for child in root:
        cfg[child.tag] = parse_element(child)
        # print(child.tag, cfg[child.tag])  # Debug

    # Convert to proper types where applicable
    converters = {
        'video_source': str,
        'model_path': str,
        'detect_confidence': float,
        'db_dir': str,
        'max_db_size_mb': float,
        'similarity_threshold': float,
        'cooldown_seconds': int,
        
        'parallel_workers_similarity': int,
        'parallel_workers_io': int,
        'parallel_workers_udp_msg': int,
        
        'prefetch_input_stream_frame': int,
        'saved_frame_dir': str,
        'udp_ip': str,
        'udp_port': int,
        
        'live_stream_dst_ip':str,
        'live_stream_dst_port':int,
        'live_stream_bitrate_kbps':int,
        'live_stream_fps':int,
        'live_stream_height':int,
        'live_stream_width':int,
               
        'src_csci_id': int,
        'dest_csci_id': int,
        'src_unit_id': int,
        'dest_unit_id': int,
        'msg_id': int,
    }

    for key, conv in converters.items():
        if key in cfg:
            cfg[key] = conv(cfg[key])

    # Optional: Convert nested values in srs_values_of_classes
    if 'srs_values_of_classes' in cfg:
        cfg['srs_values_of_classes'] = {
            k: int(v) for k, v in cfg['srs_values_of_classes'].items()
        }

    # Defaults
    cfg.setdefault('udp_ip', '127.0.0.1')

    return cfg

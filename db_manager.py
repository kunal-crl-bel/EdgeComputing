from pathlib import Path
from PIL import Image


class DBManager:
    def __init__(self, db_dir: str, max_size_mb: float):
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size_mb * 1024 * 1024

    def add_image(self, crop_img: Image.Image, id_hash: str) -> None:
        file_path = self.db_dir / f"{id_hash}.jpg"
        crop_img.save(file_path)
        self._cleanup()

    def _current_size(self) -> int:
        """  
            need optimization here, why checking the size again and again,
            use some variable to calculate the size and then just update 
            based on the value of variable, summing again and again is time consuming.
        """
        return sum(f.stat().st_size for f in self.db_dir.glob("*.jpg"))

    def _cleanup(self) -> None:
        files = sorted(self.db_dir.glob("*.jpg"),
                       key=lambda x: x.stat().st_mtime)
        while self._current_size() > self.max_size and files:
            files.pop(0).unlink()

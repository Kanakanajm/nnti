from os.path import join as path_join

class Cacheable:
    def __init__(self, cache_dir="cache/") -> None:
        self.cache_dir = cache_dir

    def cache_dir_sub(self, sub):
        return path_join(self.cache_dir, sub)
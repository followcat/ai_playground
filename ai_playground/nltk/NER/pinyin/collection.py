import json

class Collection:
    def __init__(self, items=None):
        if items is None:
            items = []
        self.items = items

    def join(self, separator=' '):
        return separator.join(str(item) for item in self.items)

    def map(self, callback):
        return Collection(list(map(callback, self.items)))

    def all(self):
        return self.items

    def to_array(self):
        return self.all()

    def to_json(self, options=0):
        return json.dumps(self.to_array(), ensure_ascii=False)

    def __str__(self):
        return self.join()

    def __getitem__(self, key):
        return self.items[key]

    def __setitem__(self, key, value):
        self.items[key] = value

    def __delitem__(self, key):
        del self.items[key]

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)

    def json_serialize(self):
        return self.items


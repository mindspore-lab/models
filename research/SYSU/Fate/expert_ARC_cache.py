class ARC_Cache:
    def __init__(self, max_size):
        self.max_size = max_size
        self.p = 0  # 自适应参数，用于调整T1和T2的大小

        self.T1 = []  # 最近使用列表，保存专家ID
        self.T2 = []  # 频繁使用列表，保存专家ID
        self.B1 = []  # T1的Ghost列表，保存专家ID
        self.B2 = []  # T2的Ghost列表，保存专家ID

        for expert_id in range(max_size):
            self.update(expert_id)
    
    def is_evicted(self, expert_id):
        if (expert_id in self.T1) or (expert_id in self.T2):
            return False
        else:
            return True
    
    def update_list(self, expert_list):
        evicted_list = []
        for expert_id in expert_list:
            evicted_id = self.update(expert_id)
            if (evicted_id is not None) and (evicted_id not in expert_list) and (evicted_id not in evicted_list):
                evicted_list.append(evicted_id)
        return evicted_list

    def update(self, expert_id):
        if expert_id in self.T1:
            self.T1.remove(expert_id)
            self.T2.append(expert_id)
        elif expert_id in self.T2:
            self.T2.remove(expert_id)
            self.T2.append(expert_id)
        elif expert_id in self.B1:
            self._adjust_p(min(len(self.T1), self.max_size))
            evicted_expert_id = self._replace(expert_id)
            self.B1.remove(expert_id)
            self.T2.append(expert_id)
            return evicted_expert_id
        elif expert_id in self.B2:
            self._adjust_p(-min(len(self.T2), self.max_size))
            evicted_expert_id = self._replace(expert_id)
            self.B2.remove(expert_id)
            self.T2.append(expert_id)
            return evicted_expert_id
        else:
            evicted_expert_id = None
            if len(self.T1) + len(self.B1) == self.max_size:
                if len(self.T1) < self.max_size:
                    self.B1.pop(0)
                    evicted_expert_id = self._replace(expert_id)
                else:
                    evicted_expert_id = self.T1.pop(0)
            elif len(self.T1) + len(self.T2) + len(self.B1) + len(self.B2) >= self.max_size:
                if len(self.T1) + len(self.T2) + len(self.B1) + len(self.B2) >= 2 * self.max_size:
                    if len(self.B1) > 0:
                        self.B1.pop(0)
                    else:
                        self.B2.pop(0)
                evicted_expert_id = self._replace(expert_id)
            self.T1.append(expert_id)
            return evicted_expert_id

    def _adjust_p(self, delta):
        self.p = min(self.max_size, max(0, self.p + delta))

    def _replace(self, expert_id):
        if (len(self.T1) > 0) and ((expert_id in self.B2 and len(self.T1) > self.p) or len(self.T1) > self.p):
            evicted_expert_id = self.T1.pop(0)
            self.B1.append(evicted_expert_id)
        else:
            if len(self.T2) > 0:
                evicted_expert_id = self.T2.pop(0)
                self.B2.append(evicted_expert_id)
            else:
                evicted_expert_id = self.T1.pop(0)
                self.B1.append(evicted_expert_id)
        return evicted_expert_id


if __name__ == "__main__":
    arc_cache = ARC_Cache(max_size=3)
    arc_cache.initialize_cache([1,2,3])

    evicted_id = arc_cache.update(4)
    print(f"Evicted Expert: {evicted_id}")

    evicted_id = arc_cache.update(2)
    print(f"Evicted Expert: {evicted_id}")

    evicted_id = arc_cache.update(5)
    print(f"Evicted Expert: {evicted_id}")

    evicted_id = arc_cache.update(3)
    print(f"Evicted Expert: {evicted_id}")

    evicted_id = arc_cache.update(6)
    print(f"Evicted Expert: {evicted_id}")

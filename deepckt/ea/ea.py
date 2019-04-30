
class EA:

    def get_next_generation_candidates(self, *args, **kwargs):
        raise NotImplementedError

    def prepare_for_generation(self, db, n):
        raise NotImplementedError

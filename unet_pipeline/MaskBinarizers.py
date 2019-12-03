class MaskBinarization():
    def __init__(self):
        self.thresholds = 0.5
    def transform(self, predicted):
        yield predicted > self.thresholds
    
class SimpleMaskBinarization(MaskBinarization):
    def __init__(self, score_thresholds):
        super().__init__()
        self.thresholds = score_thresholds
    def transform(self, predicted):
        for thr in self.thresholds:
            yield predicted > thr

class DupletMaskBinarization(MaskBinarization):
    def __init__(self, duplets, with_channels=True):
        super().__init__()
        self.thresholds = duplets
        self.dims = (2,3) if with_channels else (1,2)
    def transform(self, predicted):
        for score_threshold, area_threshold in self.thresholds:
            mask = predicted > score_threshold
            mask[mask.sum(dim=self.dims) < area_threshold] = 0
            yield mask

class TripletMaskBinarization(MaskBinarization):
    def __init__(self, triplets, with_channels=True):
        super().__init__()
        self.thresholds = triplets
        self.dims = (2,3) if with_channels else (1,2)
    def transform(self, predicted):
        for top_score_threshold, area_threshold, bottom_score_threshold in self.thresholds:     
            clf_mask = predicted > top_score_threshold
            pred_mask = predicted > bottom_score_threshold
            pred_mask[clf_mask.sum(dim=self.dims) < area_threshold] = 0
            yield pred_mask
